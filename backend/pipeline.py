"""FLUX edit pipelines: global (Kontext) + masked inpaint (Fill).

Sized for a 16 GB Blackwell card. Flux in fp16 is ~24 GB, so model CPU offload
plus VAE slicing/tiling are required — they're not optional polish.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

from backend.imgutils import ensure_l, ensure_rgb
from backend.progress import job_progress

KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
FILL_MODEL = "black-forest-labs/FLUX.1-Fill-dev"

# Set FLUX_QUANT=4bit to load the transformer and T5 text encoder in NF4
# (bitsandbytes). Whole pipeline drops from ~21 GB to ~10 GB, fits in 16 GB
# without cpu_offload — eliminates PCIe streaming, GPU actually computes
# instead of waiting on host RAM. NF4 is community-standard for Flux; visible
# quality loss is limited to fine textures / smooth gradients / image text.
_VALID_QUANT_MODES = frozenset({"4bit"})


def _read_quant_mode() -> str | None:
    """Read FLUX_QUANT env var; return canonical mode string or None if disabled."""
    raw = os.environ.get("FLUX_QUANT", "").strip().lower()
    return raw if raw in _VALID_QUANT_MODES else None


QUANT_MODE: str | None = _read_quant_mode()

Mode = Literal["kontext", "inpaint"]


class EditAborted(RuntimeError):
    """Raised from the diffusers step callback when job_progress.aborted is set.
    Distinct exception type so the server can return a dedicated status code."""


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
    # When True and an accel LoRA is configured, enable the adapter for this
    # edit; when False, disable it (lets the UI flip between fast 8-step
    # and full 28-step quality without reloading the pipeline).
    use_accel: bool = True


class FluxEditor:
    """Lazy holder for both pipelines. Each is loaded on first use."""

    def __init__(self, accel: AccelConfig | None = None) -> None:
        self._kontext = None
        self._fill = None
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.accel = accel if accel is not None else AccelConfig.from_env()

    def _load_kontext(self):  # pragma: no cover — requires GPU + 24 GB model
        from diffusers import FluxKontextPipeline

        pipe = self._from_pretrained(FluxKontextPipeline, KONTEXT_MODEL)
        self._apply_accel(pipe)
        self._apply_memory_savers(pipe)
        return pipe

    def _load_fill(self):  # pragma: no cover — requires GPU + 24 GB model
        from diffusers import FluxFillPipeline

        pipe = self._from_pretrained(FluxFillPipeline, FILL_MODEL)
        self._apply_accel(pipe)
        self._apply_memory_savers(pipe)
        return pipe

    def _from_pretrained(self, pipeline_cls, repo_id):  # pragma: no cover — GPU
        """Load `repo_id` either in bf16 (default) or NF4 4-bit (FLUX_QUANT=4bit).

        When quantized, the transformer and T5 text encoder are constructed
        separately with BitsAndBytesConfig and injected into the pipeline —
        diffusers needs the components built before pipeline assembly because
        from_pretrained doesn't accept a per-component quant config.
        """
        if not QUANT_MODE:
            return pipeline_cls.from_pretrained(repo_id, torch_dtype=self._dtype)  # nosec B615 - BFL is trusted upstream; pinning a revision would block legit upstream fixes for a single-user desktop app

        # Lazy imports — these pull bitsandbytes only when the user opts in.
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

        # nosec B615 below x3 — BFL is trusted upstream; see comment in the bf16 branch above.
        transformer = FluxTransformer2DModel.from_pretrained(  # nosec B615
            repo_id,
            subfolder="transformer",
            quantization_config=nf4_diffusers,
            torch_dtype=self._dtype,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(  # nosec B615
            repo_id,
            subfolder="text_encoder_2",
            quantization_config=nf4_transformers,
            torch_dtype=self._dtype,
        )

        return pipeline_cls.from_pretrained(  # nosec B615
            repo_id,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=self._dtype,
        )

    ACCEL_ADAPTER_NAME = "accel"

    def _apply_accel(self, pipe) -> None:  # pragma: no cover — needs diffusers pipe
        """Load the acceleration LoRA as a toggleable adapter (NOT fused).

        Fused would be slightly faster but bakes the LoRA permanently into
        weights — incompatible with per-edit on/off toggling from the UI.
        We start with the adapter active by default; edit() flips it per
        request via _set_accel_active().
        """
        if self.accel is None:
            return
        pipe.load_lora_weights(
            self.accel.repo,
            weight_name=self.accel.weight_name,
            adapter_name=self.ACCEL_ADAPTER_NAME,
        )
        pipe.set_adapters([self.ACCEL_ADAPTER_NAME], adapter_weights=[self.accel.scale])

    def _set_accel_active(self, pipe, enabled: bool) -> None:  # pragma: no cover — GPU pipe
        if self.accel is None:
            return
        if enabled:
            pipe.set_adapters([self.ACCEL_ADAPTER_NAME], adapter_weights=[self.accel.scale])
        else:
            pipe.set_adapters([])

    def _apply_memory_savers(self, pipe) -> None:  # pragma: no cover — GPU branches
        if torch.cuda.is_available():
            if QUANT_MODE:
                # NF4 brings the whole pipeline under VRAM; offload would
                # only add PCIe round-trips for no gain.
                pipe.to("cuda")
            else:
                # bf16 + 16 GB doesn't fit — stream components in/out.
                pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    @property
    def kontext(self):  # pragma: no cover — triggers _load_kontext (GPU)
        if self._kontext is None:
            self._kontext = self._load_kontext()
        return self._kontext

    @property
    def fill(self):  # pragma: no cover — triggers _load_fill (GPU)
        if self._fill is None:
            self._fill = self._load_fill()
        return self._fill

    def _release_intermediate_memory(self) -> None:  # pragma: no cover — GPU only
        """Drop intermediate activation buffers held after the call returns.

        Diffusers leaves attention KV cache + workspace tensors allocated in
        CUDA's caching allocator between calls. Without this, repeated edits
        slowly grow resident VRAM until OOM under prolonged use."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def edit(self, req: EditRequest) -> Image.Image:  # pragma: no cover — GPU inference
        generator = None
        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(req.seed))

        image = ensure_rgb(req.image)

        # diffusers calls this after each denoising step. We update the
        # shared progress state so /api/progress can report N/total live.
        # Raising here is the only point where we can interrupt the loop —
        # diffusers does not expose mid-step cancellation.
        def _on_step_end(pipe, step, timestep, callback_kwargs):
            if job_progress.aborted:
                raise EditAborted(f"aborted by user at step {step + 1}/{req.steps}")
            job_progress.advance(step)
            return callback_kwargs

        try:
            pipe = self.kontext if req.mode == "kontext" else self.fill
            self._set_accel_active(pipe, req.use_accel)

            if req.mode == "kontext":
                result = self.kontext(
                    prompt=req.prompt,
                    image=image,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance,
                    generator=generator,
                    callback_on_step_end=_on_step_end,
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
                    callback_on_step_end=_on_step_end,
                )
                return result.images[0]

            raise ValueError(f"unknown mode: {req.mode}")
        finally:
            self._release_intermediate_memory()
