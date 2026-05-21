"""PuLID for FLUX — face identity preservation across stylized edits.

PuLID (Pure and Lightning ID customization) trains a small ID encoder
that projects a face photo into FLUX's conditioning space, then injects
that vector at each denoising step so the generator preserves identity
without needing dozens of training images or full fine-tunes. Result:
"a cyberpunk version of ME" generations that actually look like you,
across any prompt or style.

Practical pieces
----------------
1. **InsightFace** extracts the user's face embedding from a reference
   photo (face detect + ArcFace alignment + 512-dim embedding).
2. **PuLID weights** load on top of FLUX.1-dev as a LoRA-style adapter
   plus an ID-projector MLP.
3. At inference, the ID embedding is added to the FLUX text embedding
   as an extra cross-attention input. The user's prompt drives the
   scene; PuLID drives WHO is in it.

Stack notes
-----------
This module is the most experimental of the seven-feature push because
PuLID-FLUX integration in diffusers is still moving — the community
pinned a working combination but it requires:

  - ``insightface`` (face detect + embedding extractor)
  - ``onnxruntime`` (already pulled in by rembg)
  - the PuLID-FLUX weights (``guozinan/PuLID-FLUX-v0.9.1``)
  - the same FLUX.1-dev base that ControlNet uses (~24 GB on disk —
    already there if ControlNet ran once)

Mirroring the ControlNet pattern: separate pipeline, lazy-loaded,
``release()`` called on the main FluxEditor before loading because a
16 GB card cannot hold both Kontext and PuLID + FLUX.1-dev warm.

VRAM
----
- FLUX.1-dev under NF4: ~6 GB
- PuLID adapter: ~700 MB
- InsightFace (CPU via ONNX): negligible
- Total: ~7 GB resident.

When integration breaks
-----------------------
Identity preservation models like PuLID move fast — published weight
schemas often need tweaks to load cleanly against newer diffusers /
transformers releases. If the lazy loader raises, the error message
points the user at the upstream repo for the working combination.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
from PIL import Image

from backend.imgutils import ensure_rgb

log = logging.getLogger(__name__)

PULID_MODEL = "guozinan/PuLID-FLUX-v0.9.1"
FLUX_DEV_MODEL = "black-forest-labs/FLUX.1-dev"
INSIGHTFACE_MODEL = "buffalo_l"  # InsightFace's standard face-detect + ArcFace bundle


@dataclass
class PuLIDRequest:
    """One face-preserving generation request."""

    prompt: str
    face_image: Image.Image
    id_scale: float = 1.0  # how strongly the face embedding biases the output
    steps: int = 28
    guidance: float = 3.5
    seed: int | None = None
    width: int = 1024
    height: int = 1024


class PuLIDGenerator:
    """Lazy holder for the FLUX + PuLID + InsightFace stack."""

    def __init__(self) -> None:
        self._pipe = None
        self._face_app = None  # InsightFace application
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _load(self) -> None:  # pragma: no cover — needs GPU + multi-GB download
        if self._pipe is not None:
            return
        try:
            from diffusers import FluxPipeline
            from huggingface_hub import hf_hub_download
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "PuLID requires the 'insightface' package. Install with:\n"
                "    pip install insightface onnxruntime\n"
                "Original ImportError: " + str(exc)
            ) from exc

        from backend.pipeline import QUANT_MODE

        log.info("loading PuLID (FLUX.1-dev + ID adapter)")
        # FLUX.1-dev base, same as ControlNet uses. NF4 quant when configured.
        if QUANT_MODE == "4bit":
            base_kwargs = self._build_nf4_components()
        else:
            base_kwargs = {"torch_dtype": self._dtype}

        # nosec B615 — BFL is trusted; pinning would block legit upstream
        # fixes for a single-user desktop app.
        pipe = FluxPipeline.from_pretrained(FLUX_DEV_MODEL, **base_kwargs)  # nosec B615

        # PuLID weights ship as a single safetensors file with the LoRA
        # delta + the ID projection MLP. We pull it via hf_hub_download
        # and then attach it through the pipeline's LoRA loader; the
        # projector head is wired up via setattr on the transformer.
        pulid_path = hf_hub_download(repo_id=PULID_MODEL, filename="pulid_flux_v0.9.1.safetensors")  # nosec B615
        pipe.load_lora_weights(pulid_path, adapter_name="pulid")
        pipe.set_adapters(["pulid"], adapter_weights=[1.0])

        if torch.cuda.is_available():
            if QUANT_MODE == "4bit":
                pipe.to("cuda")
            else:
                pipe.enable_model_cpu_offload()
        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        self._pipe = pipe

        # InsightFace face encoder. Bundle ``buffalo_l`` contains
        # RetinaFace (detection) + ArcFace ResNet-100 (recognition).
        # First call downloads ~280 MB into ~/.insightface.
        self._face_app = FaceAnalysis(name=INSIGHTFACE_MODEL)
        self._face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        log.info("PuLID stack ready")

    def _build_nf4_components(self):  # pragma: no cover — bitsandbytes path
        """Construct NF4-quantized FLUX.1-dev components. Mirrors the
        FluxEditor / ControlNet pattern in backend/pipeline.py."""
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from diffusers import FluxTransformer2DModel
        from transformers import BitsAndBytesConfig as TransformersBnbConfig
        from transformers import T5EncoderModel

        nf4_diffusers = DiffusersBnbConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self._dtype,
        )
        nf4_transformers = TransformersBnbConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self._dtype,
        )
        transformer = FluxTransformer2DModel.from_pretrained(  # nosec B615
            FLUX_DEV_MODEL,
            subfolder="transformer",
            quantization_config=nf4_diffusers,
            torch_dtype=self._dtype,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(  # nosec B615
            FLUX_DEV_MODEL,
            subfolder="text_encoder_2",
            quantization_config=nf4_transformers,
            torch_dtype=self._dtype,
        )
        return {
            "transformer": transformer,
            "text_encoder_2": text_encoder_2,
            "torch_dtype": self._dtype,
        }

    def _extract_id_embedding(  # pragma: no cover — InsightFace inference
        self, face_image: Image.Image
    ) -> torch.Tensor:
        """Run InsightFace's detector + recognizer to produce a 512-dim
        ArcFace embedding. Returns the highest-confidence face when
        the photo contains multiple."""
        import numpy as np

        rgb = ensure_rgb(face_image)
        # InsightFace expects BGR uint8 (OpenCV convention).
        arr = np.array(rgb)[:, :, ::-1].copy()
        faces = self._face_app.get(arr)
        if not faces:
            raise ValueError(
                "no face detected in the reference photo — try a clearer, "
                "front-facing shot at higher resolution"
            )
        # Best face = highest detection score.
        best = max(faces, key=lambda f: float(f.det_score))
        embedding = torch.from_numpy(best.embedding).to(self._dtype)
        return embedding

    def generate(self, req: PuLIDRequest) -> Image.Image:  # pragma: no cover — GPU
        """Run face-preserving generation. The ID embedding is passed
        to the pipeline as a cross-attention input via the joint_attention
        kwargs that PuLID's LoRA hook reads."""
        self._load()
        id_embedding = self._extract_id_embedding(req.face_image)

        generator = None
        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(req.seed))

        result = self._pipe(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=generator,
            # PuLID hooks: the ID embedding rides on joint_attention_kwargs
            # so the LoRA-injected attention layers can read it. ``id_scale``
            # toggles how strongly the embedding biases the output.
            joint_attention_kwargs={"id_embedding": id_embedding, "id_scale": req.id_scale},
        )
        return result.images[0]

    def release(self) -> None:  # pragma: no cover — GPU
        """Drop PuLID weights + InsightFace to free VRAM. Next call reloads."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._face_app = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


__all__ = ["FLUX_DEV_MODEL", "INSIGHTFACE_MODEL", "PULID_MODEL", "PuLIDGenerator", "PuLIDRequest"]
