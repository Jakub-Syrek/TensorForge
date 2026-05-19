"""FLUX edit pipelines: global (Kontext) + masked inpaint (Fill).

Sized for a 16 GB Blackwell card. Flux in fp16 is ~24 GB, so model CPU offload
plus VAE slicing/tiling are required — they're not optional polish.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

from backend.imgutils import ensure_l, ensure_rgb, image_to_png_bytes  # noqa: F401  (re-export)

KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
FILL_MODEL = "black-forest-labs/FLUX.1-Fill-dev"

Mode = Literal["kontext", "inpaint"]


@dataclass(frozen=True)
class AccelConfig:
    """Optional acceleration LoRA (e.g. Hyper-SD, Flux-Turbo) to drop step count.

    Set via env vars at process start — read once by `AccelConfig.from_env()`.
    The LoRA is fused into both Kontext and Fill pipelines on first load.

    Caveat: Hyper-SD / Flux-Turbo are trained on base FLUX.1-dev. Kontext and
    Fill share the transformer architecture so the weights load cleanly, but
    quality at low step counts on those variants is not officially validated —
    verify visually before relying on it.
    """
    repo: str
    weight_name: str
    scale: float = 1.0

    @classmethod
    def from_env(cls) -> AccelConfig | None:
        repo = os.environ.get("FLUX_ACCEL_REPO")
        weight = os.environ.get("FLUX_ACCEL_WEIGHT")
        if not repo or not weight:
            return None
        scale = float(os.environ.get("FLUX_ACCEL_SCALE", "1.0"))
        return cls(repo=repo, weight_name=weight, scale=scale)


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

    def __init__(self, accel: AccelConfig | None = None) -> None:
        self._kontext = None
        self._fill = None
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.accel = accel if accel is not None else AccelConfig.from_env()

    def _load_kontext(self):
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained(KONTEXT_MODEL, torch_dtype=self._dtype)
        self._apply_accel(pipe)
        self._apply_memory_savers(pipe)
        return pipe

    def _load_fill(self):
        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained(FILL_MODEL, torch_dtype=self._dtype)
        self._apply_accel(pipe)
        self._apply_memory_savers(pipe)
        return pipe

    def _apply_accel(self, pipe) -> None:
        """Load and fuse the acceleration LoRA, if configured.

        Fusing (rather than keeping it as a separate adapter) bakes the delta
        into the base weights — slightly faster inference, no per-call adapter
        switching, and `enable_model_cpu_offload` doesn't have to chase the
        adapter modules separately.
        """
        if self.accel is None:
            return
        pipe.load_lora_weights(self.accel.repo, weight_name=self.accel.weight_name)
        pipe.fuse_lora(lora_scale=self.accel.scale)
        pipe.unload_lora_weights()

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
