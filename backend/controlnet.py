"""ControlNet for FLUX — compositional control via condition images.

ControlNet adds spatial conditioning on top of FLUX text-to-image
generation. Where the prompt says WHAT to draw, the control image says
WHERE — give it a pose skeleton, a depth map, or a canny edge map, and
the generator follows that geometry while filling in style and texture
from the prompt.

We use ``Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro`` — a single
checkpoint that handles multiple condition types via a ``union_mode``
integer selector at inference time. Saves having one ControlNet per
condition. Trained on the base ``black-forest-labs/FLUX.1-dev`` weights,
so we pull those alongside (separately from Kontext's fine-tune).

Resource budget
---------------
- Disk: ~24 GB (FLUX.1-dev base) + ~6 GB (Union-Pro ControlNet),
  one-time download into the HF cache.
- VRAM (NF4): ~3 GB transformer + ~3 GB T5 + ~1.5 GB ControlNet + VAE
  = ~8 GB resident. Cannot coexist with a warm Kontext/Fill on a 16 GB
  card; ``release()`` is called on the FluxEditor before loading
  ControlNet, and vice versa.
- VRAM (bf16): ~22 GB resident — requires ``cpu_offload`` on 16 GB.

This file deliberately mirrors the patterns in backend/pipeline.py
(NF4 quant, lazy load, accel-LoRA hook) so the runtime profile is
predictable for users who know how the main editor behaves.

Pre-processing
--------------
We do NOT preprocess user images into canny/depth/pose maps server-side.
That keeps the dependency surface light (no opencv-python, no
DepthAnything, no DWPose) and gives the user full control over the
condition. Industry-standard tools (the controlnet_aux package, the
ComfyUI preprocessor nodes, online generators) produce these maps; the
user drops the resulting image into our control_image slot.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
from PIL import Image

from backend.imgutils import ensure_rgb

log = logging.getLogger(__name__)

CONTROLNET_MODEL = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
FLUX_DEV_MODEL = "black-forest-labs/FLUX.1-dev"

# Union-Pro's published mode table. We expose the three most useful for
# fantasy / sci-fi work (composition, depth, pose); the others (tile,
# blur, grayscale, pidi) are reachable via the env override below for
# power users without us cluttering the UI dropdown.
UNION_MODES = {
    "canny": 0,
    "depth": 2,
    "pose": 4,
}
UNION_DEFAULT = "canny"


@dataclass
class ControlRequest:
    """One-shot ControlNet generation request."""

    prompt: str
    control_image: Image.Image
    control_type: str = UNION_DEFAULT  # key into UNION_MODES
    control_scale: float = 0.7
    steps: int = 28
    guidance: float = 3.5
    seed: int | None = None
    width: int = 1024
    height: int = 1024


class ControlNetGenerator:
    """Lazy holder for the FLUX-ControlNet generation pipeline."""

    def __init__(self) -> None:
        self._pipe = None
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _load(self) -> None:  # pragma: no cover — needs GPU + ~30 GB download
        if self._pipe is not None:
            return
        # Lazy imports — bitsandbytes / diffusers don't load unless the
        # user clicks the ControlNet path.
        from diffusers import FluxControlNetModel, FluxControlNetPipeline

        from backend.pipeline import QUANT_MODE

        log.info("loading FLUX ControlNet (Union-Pro + FLUX.1-dev base)")
        if QUANT_MODE == "4bit":
            controlnet, base_kwargs = self._build_nf4_components()
        else:
            controlnet = FluxControlNetModel.from_pretrained(  # nosec B615
                CONTROLNET_MODEL, torch_dtype=self._dtype
            )
            base_kwargs = {"torch_dtype": self._dtype}

        # nosec B615 — black-forest-labs / Shakker-Labs are trusted
        # upstreams; pinning a revision would block legit fixes for a
        # single-user desktop app.
        pipe = FluxControlNetPipeline.from_pretrained(  # nosec B615
            FLUX_DEV_MODEL,
            controlnet=controlnet,
            **base_kwargs,
        )
        self._apply_memory_savers(pipe)
        self._pipe = pipe
        log.info("ControlNet pipeline ready")

    def _build_nf4_components(self):  # pragma: no cover — bitsandbytes path
        """Build the NF4-quantized transformer + T5 + ControlNet trio.

        Same pattern as backend/pipeline.py's ``_from_pretrained`` but
        targeted at FLUX.1-dev (not the Kontext fine-tune) and with the
        ControlNet model quantized alongside.
        """
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from diffusers import FluxControlNetModel, FluxTransformer2DModel
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
        # The ControlNet itself does NOT go through NF4 — it's already
        # only ~3 GB in bf16 and the union head is sensitive to
        # quantization noise. Run it in compute dtype.
        controlnet = FluxControlNetModel.from_pretrained(  # nosec B615
            CONTROLNET_MODEL, torch_dtype=self._dtype
        )
        return controlnet, {
            "transformer": transformer,
            "text_encoder_2": text_encoder_2,
            "torch_dtype": self._dtype,
        }

    def _apply_memory_savers(self, pipe) -> None:  # pragma: no cover — GPU
        from backend.pipeline import QUANT_MODE

        if torch.cuda.is_available():
            if QUANT_MODE == "4bit":
                pipe.to("cuda")
            else:
                pipe.enable_model_cpu_offload()
        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

    def generate(self, req: ControlRequest) -> Image.Image:  # pragma: no cover — GPU
        """Run ControlNet-guided generation. Returns a single PIL image."""
        if req.control_type not in UNION_MODES:
            raise ValueError(
                f"control_type must be one of {list(UNION_MODES)}, got {req.control_type!r}"
            )
        self._load()

        control_image = ensure_rgb(req.control_image).resize((req.width, req.height))
        generator = None
        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(req.seed))

        result = self._pipe(
            prompt=req.prompt,
            control_image=control_image,
            control_mode=UNION_MODES[req.control_type],
            controlnet_conditioning_scale=req.control_scale,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=generator,
        )
        return result.images[0]

    def release(self) -> None:  # pragma: no cover — GPU
        """Drop the pipeline from VRAM. Call before loading a different
        FLUX flavor (Kontext / Fill / schnell) to avoid OOM on 16 GB."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


__all__ = [
    "CONTROLNET_MODEL",
    "FLUX_DEV_MODEL",
    "UNION_DEFAULT",
    "UNION_MODES",
    "ControlNetGenerator",
    "ControlRequest",
]
