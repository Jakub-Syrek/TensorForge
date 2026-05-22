"""Text-to-video and image-to-video generation.

Two backends share one entry point:

* **LTX-Video** (Lightricks) — small + fast. ~6-9 GB VRAM, ~30 s for a
  5 s clip. Lower fidelity but enables interactive iteration.
* **Wan 2.2** (Alibaba) — bigger + slower. ~10-14 GB VRAM, ~3 min for a
  5 s clip. Best open-source quality as of late 2025, strong motion
  coherence.

Both backends support two subtypes:

* ``t2v`` — text only -> video.
* ``i2v`` — text + a single reference image -> video. The reference
  becomes (or animates from) the first frame.

VRAM coexistence
----------------
Video pipelines are large; they cannot share VRAM with a warm FluxEditor
on a 16 GB card. The worker calls ``FluxEditor._release_intermediate_memory()``
before dispatching a video task. Switching between LTX and Wan inside
this module also releases the previous backend's pipeline before loading
the next one — LRU=1 across the two video pipelines.

Output
------
Each ``generate(...)`` call returns the encoded ``.mp4`` bytes. The
worker then routes those bytes through ``storage.save_variant_output``
with ``ext=".mp4"``. We use ``diffusers.utils.export_to_video`` for
encoding, which fans out to ``imageio[ffmpeg]`` under the hood and
produces an H.264 MP4 readable by ``<video>`` tags.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from backend.imgutils import ensure_rgb

log = logging.getLogger(__name__)

# Default model IDs — pinned to what's stable as of the 2026 freeze. Users
# can swap via env vars if they want a different revision.
LTX_T2V_MODEL = "Lightricks/LTX-Video"
LTX_I2V_MODEL = "Lightricks/LTX-Video"  # same checkpoint, separate pipeline class
# Wan 2.2 has two SKUs on HuggingFace:
#
#   * A14B — two separate checkpoints (T2V / I2V), each a MoE with two
#     14B experts (transformer + transformer_2). ~56 GB on disk per
#     checkpoint. At load time diffusers mmaps BOTH experts before
#     freeing the first, so peak virtual-memory commit reaches ~120 GB.
#     Below that, Windows raises os error 1455 ("page file too small")
#     from safetensors.safe_open during the second expert's shards. We
#     burned multiple hours trying to crank Windows page file high
#     enough; at 128 GB it still crashes mid-load.
#
#   * TI2V-5B — UNIFIED text+image-to-video, smaller MoE (5B+5B), uses
#     the same WanPipeline class for both subtypes. ~20 GB total
#     working set. Fits on a 16 GB / 32 GB RAM machine without
#     thrashing the page file. Quality below A14B but well above LTX.
#
# We default to TI2V-5B because A14B is effectively unreachable on a
# single-consumer-GPU + 32 GB RAM box. Users with workstation-class
# RAM (128+ GB) can swap to A14B via env override.
WAN_T2V_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_I2V_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"  # unified — same checkpoint as T2V

VideoBackend = Literal["ltx", "wan"]
VideoSubtype = Literal["t2v", "i2v"]


@dataclass
class VideoRequest:
    """One video-generation request.

    Attributes:
        backend: Which model family ("ltx" for fast iteration, "wan" for
            best quality).
        subtype: "t2v" for text-only, "i2v" for text + reference image.
        prompt: Description of the scene/motion.
        ref_image: Required for ``subtype="i2v"`` — the starting frame
            (or strong visual anchor; depends on the backend).
        num_frames: How many frames to render. LTX defaults 97 (~4 s
            @24fps); Wan defaults 81 (~3.4 s @24fps).
        fps: Output framerate. Most backends train at 24; lowering it
            stretches motion, raising it loses fidelity.
        width / height: Output resolution. Must be /16 for LTX, /16 for
            Wan; backends will clamp internally.
        steps: Denoising steps. LTX-distilled works in 4-8; Wan needs
            25-30 for good quality.
        guidance: CFG scale. LTX distilled = 1.0 (CFG-free); Wan = 5.0.
        seed: PRNG seed. ``None`` means random.
    """

    backend: VideoBackend
    subtype: VideoSubtype
    prompt: str
    ref_image: Image.Image | None = None
    num_frames: int = 81
    fps: int = 24
    width: int = 768
    height: int = 512
    steps: int = 28
    guidance: float = 5.0
    seed: int | None = None


class VideoGenerator:
    """Lazy holder for video diffusion pipelines.

    Keeps at most one heavy pipeline resident (LRU=1 across LTX + Wan).
    Public API is purely ``generate(request) -> bytes``; ``release()``
    drops everything from VRAM (called by the worker before falling back
    to image tasks).
    """

    def __init__(self) -> None:
        self._active_key: tuple[VideoBackend, VideoSubtype] | None = None
        self._pipe = None  # diffusers pipeline instance for the active backend
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def generate(self, req: VideoRequest) -> bytes:  # pragma: no cover — GPU + multi-GB load
        """Run one video generation. Returns encoded MP4 bytes."""
        self._validate(req)
        self._ensure_pipe(req.backend, req.subtype)
        frames = self._invoke_pipe(req)
        return self._encode_mp4(frames, fps=req.fps)

    def release(self) -> None:  # pragma: no cover — GPU
        """Drop any resident pipeline + adapters from VRAM. Idempotent."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._active_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(req: VideoRequest) -> None:
        """Reject obviously bad requests before we burn time loading
        multi-GB checkpoints."""
        if req.backend not in ("ltx", "wan"):
            raise ValueError(f"backend must be 'ltx' or 'wan', got {req.backend!r}")
        if req.subtype not in ("t2v", "i2v"):
            raise ValueError(f"subtype must be 't2v' or 'i2v', got {req.subtype!r}")
        if req.subtype == "i2v" and req.ref_image is None:
            raise ValueError("i2v requires ref_image")
        if not req.prompt or not req.prompt.strip():
            raise ValueError("prompt is empty")
        if req.num_frames < 8 or req.num_frames > 257:
            raise ValueError(f"num_frames must be between 8 and 257, got {req.num_frames}")
        if req.fps < 8 or req.fps > 60:
            raise ValueError(f"fps must be between 8 and 60, got {req.fps}")

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

    def _ensure_pipe(
        self, backend: VideoBackend, subtype: VideoSubtype
    ) -> None:  # pragma: no cover — GPU
        """Load the pipeline matching (backend, subtype). Drops the
        previous one if the key changed."""
        key = (backend, subtype)
        if self._active_key == key and self._pipe is not None:
            return
        # LRU=1: release the previous pipeline before loading the next.
        self.release()
        log.info("loading video pipeline backend=%s subtype=%s", backend, subtype)
        if backend == "ltx":
            self._pipe = self._load_ltx(subtype)
        else:
            self._pipe = self._load_wan(subtype)
        self._active_key = key
        if torch.cuda.is_available():
            # VRAM strategy:
            #  - LTX-Video weights ~8-9 GB resident; fits comfortably on a
            #    16 GB card with full residency. cpu_offload would swap
            #    every block over PCIe (240 W spike -> 27 W idle pattern),
            #    making inference 3-5x slower than necessary. Force .to("cuda").
            #  - Wan 2.2 A14B is ~28 GB across two experts + T5 + VAE; it
            #    cannot stay fully resident. enable_model_cpu_offload keeps
            #    activations in VRAM while parking idle blocks on CPU.
            try:
                if backend == "ltx":
                    self._pipe.to("cuda")
                else:
                    self._pipe.enable_model_cpu_offload()
            except Exception:
                self._pipe.enable_model_cpu_offload()
        log.info("video pipeline ready")

    def _load_ltx(self, subtype: VideoSubtype):  # pragma: no cover — GPU + downloads
        """Instantiate the LTX-Video pipeline for the requested subtype."""
        if subtype == "t2v":
            from diffusers import LTXPipeline  # type: ignore[import-not-found]

            return LTXPipeline.from_pretrained(  # nosec B615
                LTX_T2V_MODEL, torch_dtype=self._dtype
            )
        from diffusers import LTXImageToVideoPipeline  # type: ignore[import-not-found]

        return LTXImageToVideoPipeline.from_pretrained(  # nosec B615
            LTX_I2V_MODEL, torch_dtype=self._dtype
        )

    def _load_wan(self, subtype: VideoSubtype):  # pragma: no cover — GPU + downloads
        """Instantiate the Wan 2.2 pipeline for the requested subtype.

        Both subtypes share the same TI2V-5B checkpoint (``WAN_T2V_MODEL``
        and ``WAN_I2V_MODEL`` point at the same repo on purpose). The
        pipeline *class* differs:

        * ``t2v`` -> ``WanPipeline`` (no ``image`` kwarg at call time)
        * ``i2v`` -> ``WanImageToVideoPipeline`` (requires ``image``)

        Diffusers' ``WanImageToVideoPipeline`` keeps ``image_encoder`` as
        an optional component and only calls it when
        ``transformer.config.image_dim`` is not None — which it is on
        Wan 2.1 I2V but **not** on Wan 2.2 TI2V-5B. So we can construct
        the I2V pipeline directly from the TI2V-5B checkpoint without
        the CLIP image encoder; the unified DiT consumes the reference
        frame via VAE latents instead of CLIP embeddings.
        """
        if subtype == "t2v":
            from diffusers import WanPipeline  # type: ignore[import-not-found]

            return WanPipeline.from_pretrained(  # nosec B615
                WAN_T2V_MODEL, torch_dtype=self._dtype
            )
        from diffusers import WanImageToVideoPipeline  # type: ignore[import-not-found]

        return WanImageToVideoPipeline.from_pretrained(  # nosec B615
            WAN_I2V_MODEL, torch_dtype=self._dtype
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _invoke_pipe(self, req: VideoRequest):  # pragma: no cover — GPU
        """Call the active pipeline. Returns the frames object (PIL list
        or numpy stack — diffusers' export_to_video handles both)."""
        if self._pipe is None:
            raise RuntimeError("_invoke_pipe called before _ensure_pipe")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = (
            torch.Generator(device).manual_seed(int(req.seed)) if req.seed is not None else None
        )
        call_kwargs = {
            "prompt": req.prompt,
            "num_frames": req.num_frames,
            "width": req.width,
            "height": req.height,
            "num_inference_steps": req.steps,
            "guidance_scale": req.guidance,
            "generator": generator,
        }
        if req.subtype == "i2v":
            ref = ensure_rgb(req.ref_image)
            call_kwargs["image"] = ref
        result = self._pipe(**call_kwargs)
        # All current video pipelines return ``.frames`` — a list of one
        # batch's frame list. We unwrap the batch dim (always size 1 in
        # our single-prompt requests).
        frames = result.frames[0] if hasattr(result, "frames") else result
        return frames

    # ------------------------------------------------------------------
    # MP4 encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_mp4(frames, fps: int) -> bytes:  # pragma: no cover — depends on imageio
        """Encode a frame stack to H.264 MP4 bytes via diffusers'
        ``export_to_video`` helper (which fans out to imageio[ffmpeg]).
        """
        from diffusers.utils import export_to_video  # type: ignore[import-not-found]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            export_to_video(frames, str(tmp_path), fps=fps)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)


__all__ = [
    "LTX_I2V_MODEL",
    "LTX_T2V_MODEL",
    "WAN_I2V_MODEL",
    "WAN_T2V_MODEL",
    "VideoBackend",
    "VideoGenerator",
    "VideoRequest",
    "VideoSubtype",
]
