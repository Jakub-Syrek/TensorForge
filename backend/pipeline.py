"""FLUX edit pipelines: global (Kontext) + masked inpaint (Fill).

Sized for a 16 GB Blackwell card. Flux in fp16 is ~24 GB, so model CPU offload
plus VAE slicing/tiling are required — they're not optional polish.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

from backend.imgutils import ensure_l, ensure_rgb, image_to_png_bytes  # noqa: F401  (re-export)

KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
FILL_MODEL = "black-forest-labs/FLUX.1-Fill-dev"

Mode = Literal["kontext", "inpaint"]


@dataclass
class EditRequest:
    mode: Mode
    prompt: str
    image: Image.Image
    mask: Image.Image | None = None  # required for inpaint, ignored for kontext
    steps: int = 28
    guidance: float = 3.5
    seed: int | None = None


class FluxEditor:
    """Lazy holder for both pipelines. Each is loaded on first use."""

    def __init__(self) -> None:
        self._kontext = None
        self._fill = None
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _load_kontext(self):
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained(KONTEXT_MODEL, torch_dtype=self._dtype)
        self._apply_memory_savers(pipe)
        return pipe

    def _load_fill(self):
        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained(FILL_MODEL, torch_dtype=self._dtype)
        self._apply_memory_savers(pipe)
        return pipe

    def _apply_memory_savers(self, pipe) -> None:
        if torch.cuda.is_available():
            # Required on 16 GB: keeps non-active submodules on CPU.
            pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    @property
    def kontext(self):
        if self._kontext is None:
            self._kontext = self._load_kontext()
        return self._kontext

    @property
    def fill(self):
        if self._fill is None:
            self._fill = self._load_fill()
        return self._fill

    def edit(self, req: EditRequest) -> Image.Image:
        generator = None
        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(req.seed))

        image = ensure_rgb(req.image)

        if req.mode == "kontext":
            result = self.kontext(
                prompt=req.prompt,
                image=image,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance,
                generator=generator,
            )
            return result.images[0]

        if req.mode == "inpaint":
            if req.mask is None:
                raise ValueError("inpaint mode requires a mask image")
            mask = ensure_l(req.mask).resize(image.size)
            result = self.fill(
                prompt=req.prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance,
                generator=generator,
            )
            return result.images[0]

        raise ValueError(f"unknown mode: {req.mode}")
