"""FLUX edit pipelines: global (Kontext) + masked inpaint (Fill).

Sized for a 16 GB Blackwell card. Flux in fp16 is ~24 GB, so model CPU offload
plus VAE slicing/tiling are required — they're not optional polish.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

from backend.imgutils import ensure_l, ensure_rgb
from backend.loras import StyleLoRA
from backend.loras import get as get_style_lora
from backend.loras import is_compatible as style_compatible
from backend.progress import job_progress

log = logging.getLogger(__name__)

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

Mode = Literal["kontext", "inpaint", "qwen", "generate"]

QWEN_EDIT_MODEL = "Qwen/Qwen-Image-Edit"
SCHNELL_MODEL = "black-forest-labs/FLUX.1-schnell"

# IP-Adapter for FLUX — image-as-prompt augmentation. Drop a reference
# image, the model inherits its style and composition without needing
# verbal description. Hosted by InstantX, ~1 GB additional VRAM, works
# on both schnell and kontext (and the base FLUX.1-dev that ControlNet
# uses — but that's a separate pipeline).
IP_ADAPTER_REPO = "InstantX/FLUX.1-dev-IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter.bin"


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
    image: Image.Image | None = None  # None for generate mode (text-to-image)
    mask: Image.Image | None = None  # required for inpaint, ignored otherwise
    steps: int = 28
    guidance: float = 3.5
    seed: int | None = None
    # Output dimensions for generate mode; ignored for image-conditioned modes
    # (those derive size from the input image).
    width: int = 1024
    height: int = 1024
    # When True and an accel LoRA is configured, enable the adapter for this
    # edit; when False, disable it (lets the UI flip between fast 8-step
    # and full 28-step quality without reloading the pipeline).
    use_accel: bool = True
    # Optional style LoRA id (from backend.loras.STYLE_LORAS). When set, the
    # corresponding adapter is loaded (once) and combined with the accel
    # adapter via diffusers' multi-adapter mechanism. None = no style bias.
    style_lora_id: str | None = None
    style_lora_scale: float = 1.0
    # Optional IP-Adapter reference image. When set, FLUX inherits visual
    # style / composition from this image alongside the prompt — useful
    # when an aesthetic is hard to describe in words. Loaded lazily on
    # first use; the per-request scale (0-2, ~0.7 typical) controls how
    # strongly the reference biases the output.
    ip_adapter_image: Image.Image | None = None
    ip_adapter_scale: float = 0.7


class FluxEditor:
    """Lazy holder for both pipelines. Each is loaded on first use."""

    def __init__(self, accel: AccelConfig | None = None) -> None:
        self._kontext = None
        self._fill = None
        self._qwen = None
        self._schnell = None
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.accel = accel if accel is not None else AccelConfig.from_env()
        # Per-pipeline set of style LoRA adapter names already loaded onto
        # that pipe. Keyed by ``id(pipe)`` so pipes are tracked even though
        # they're plain objects (no hashable identity beyond that).
        self._loaded_style_loras: dict[int, set[str]] = {}
        # Per-pipeline flag tracking whether the IP-Adapter has been loaded
        # onto that pipe. Loading is one-shot and ~1 GB; toggling per
        # request is done via ``set_ip_adapter_scale(0)`` instead.
        self._ip_adapter_loaded: set[int] = set()

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

    def _load_schnell(self):  # pragma: no cover — requires GPU + ~13 GB model
        """Load Flux schnell — 4-step distilled text-to-image generator.

        Shares the Flux DiT transformer with Kontext/Fill, so our NF4 path
        works here too: same FluxTransformer2DModel + T5EncoderModel pieces.
        guidance_scale is unused (schnell was distilled with cfg baked in).
        """
        from diffusers import FluxPipeline

        pipe = self._from_pretrained(FluxPipeline, SCHNELL_MODEL)
        # No accel LoRA — schnell is already step-distilled to 4 steps;
        # layering Hyper-SD on top would be redundant and probably degrade.
        self._apply_memory_savers(pipe)
        return pipe

    def _load_qwen(self):  # pragma: no cover — requires GPU + ~20 GB model
        """Load Qwen-Image-Edit as an alternative instructive-edit backend.

        Different architecture than Flux: Qwen2-VL text encoder + DiT
        transformer. Our Flux-specific NF4 path (`_from_pretrained` with
        T5EncoderModel + FluxTransformer2DModel) doesn't apply — Qwen
        runs as bf16 with cpu_offload. Hyper-SD LoRA also doesn't transfer
        (Flux-trained, different attention shapes), so no _apply_accel here.
        """
        from diffusers import QwenImageEditPipeline

        pipe = QwenImageEditPipeline.from_pretrained(  # nosec B615 - upstream Qwen team trusted
            QWEN_EDIT_MODEL,
            torch_dtype=self._dtype,
        )
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
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

    def _ensure_ip_adapter_loaded(self, pipe) -> None:  # pragma: no cover — GPU + download
        """Load the IP-Adapter weights onto ``pipe`` on first use.

        Subsequent requests reuse the warm adapter — toggling between
        IP-Adapter-on and IP-Adapter-off is done by setting the scale
        to 0 (effectively a no-op), not by unloading.
        """
        pipe_id = id(pipe)
        if pipe_id in self._ip_adapter_loaded:
            return
        log.info("loading IP-Adapter weights onto %s", type(pipe).__name__)
        # nosec B615 — InstantX is a trusted upstream; pinning a revision
        # would block legit fixes for a single-user desktop app.
        pipe.load_ip_adapter(  # nosec B615
            IP_ADAPTER_REPO,
            weight_name=IP_ADAPTER_WEIGHT,
        )
        self._ip_adapter_loaded.add(pipe_id)

    def _apply_ip_adapter(  # pragma: no cover — GPU pipe
        self,
        pipe,
        ip_image: Image.Image | None,
        scale: float,
    ) -> Image.Image | None:
        """Configure the IP-Adapter for one request and return the image
        to pass into the pipe call (or None if disabled).

        scale=0 OR ip_image=None -> set adapter scale to 0 (effectively
        disables conditioning without unloading) and return None.
        Otherwise load the adapter (idempotent), set the scale, return
        the image so the caller passes it to ``pipe(..., ip_adapter_image=img)``.
        """
        if ip_image is None or scale <= 0:
            # Only zero the scale if the adapter was previously loaded —
            # calling set_ip_adapter_scale on a pipe that never loaded
            # one raises. Tracking via _ip_adapter_loaded covers this.
            if id(pipe) in self._ip_adapter_loaded:
                pipe.set_ip_adapter_scale(0.0)
            return None
        self._ensure_ip_adapter_loaded(pipe)
        pipe.set_ip_adapter_scale(scale)
        return ensure_rgb(ip_image)

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
        """Legacy single-adapter setter — kept for callers that don't deal
        with style LoRAs. New code should use ``_set_active_adapters``."""
        if self.accel is None:
            return
        if enabled:
            pipe.set_adapters([self.ACCEL_ADAPTER_NAME], adapter_weights=[self.accel.scale])
        else:
            pipe.set_adapters([])

    def _style_adapter_name(self, lora: StyleLoRA) -> str:
        return f"style_{lora.id}"

    def _ensure_style_lora_loaded(self, pipe, lora: StyleLoRA) -> str:  # pragma: no cover — GPU
        """Load `lora` onto `pipe` once. Returns the diffusers adapter name."""
        adapter_name = self._style_adapter_name(lora)
        loaded = self._loaded_style_loras.setdefault(id(pipe), set())
        if adapter_name in loaded:
            return adapter_name
        pipe.load_lora_weights(
            lora.repo,
            weight_name=lora.weight_name,
            adapter_name=adapter_name,
        )
        loaded.add(adapter_name)
        return adapter_name

    def _set_active_adapters(  # pragma: no cover — GPU pipe
        self,
        pipe,
        *,
        use_accel: bool,
        style: StyleLoRA | None,
        style_scale: float,
    ) -> None:
        """Compose the adapter stack for one request.

        Accel + style are independent layers. Diffusers' multi-adapter API
        takes parallel name/weight lists and blends linearly. An empty list
        disables every adapter on the pipe.
        """
        names: list[str] = []
        weights: list[float] = []
        if self.accel is not None and use_accel:
            names.append(self.ACCEL_ADAPTER_NAME)
            weights.append(self.accel.scale)
        if style is not None:
            adapter_name = self._ensure_style_lora_loaded(pipe, style)
            names.append(adapter_name)
            weights.append(style_scale)
        if names:
            pipe.set_adapters(names, adapter_weights=weights)
            return

        # No adapters requested. Only worth zeroing the stack if SOMETHING
        # was previously loaded onto this pipe — otherwise diffusers'
        # set_adapters([]) walks an empty ``peft_config`` and raises
        # ``KeyError: 'transformer'`` (or AttributeError on older versions).
        # We know what was loaded via our per-pipe trackers; if all are
        # empty for this pipe, the stack is already clean and the call
        # is unnecessary.
        pipe_id = id(pipe)
        ever_loaded = (
            self.accel is not None
            or pipe_id in self._loaded_style_loras
            or pipe_id in self._ip_adapter_loaded
        )
        if not ever_loaded:
            return
        with contextlib.suppress(ValueError, RuntimeError, KeyError, AttributeError):
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

    @property
    def qwen(self):  # pragma: no cover — triggers _load_qwen (GPU)
        if self._qwen is None:
            self._qwen = self._load_qwen()
        return self._qwen

    @property
    def schnell(self):  # pragma: no cover — triggers _load_schnell (GPU)
        if self._schnell is None:
            self._schnell = self._load_schnell()
        return self._schnell

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

        image = ensure_rgb(req.image) if req.image is not None else None

        # diffusers calls this after each denoising step. We update the
        # shared progress state so /api/progress can report N/total live.
        # Raising here is the only point where we can interrupt the loop —
        # diffusers does not expose mid-step cancellation.
        def _on_step_end(pipe, step, timestep, callback_kwargs):
            if job_progress.aborted:
                raise EditAborted(f"aborted by user at step {step + 1}/{req.steps}")
            job_progress.advance(step)
            return callback_kwargs

        # Resolve optional style LoRA. Only kontext + generate accept one
        # (fill / qwen ignored by design — see backend/loras.py docstring).
        style = get_style_lora(req.style_lora_id)
        if style is not None and not style_compatible(style, req.mode):
            style = None

        try:
            if req.mode == "kontext":
                self._set_active_adapters(
                    self.kontext,
                    use_accel=req.use_accel,
                    style=style,
                    style_scale=req.style_lora_scale,
                )
                ip_img = self._apply_ip_adapter(
                    self.kontext, req.ip_adapter_image, req.ip_adapter_scale
                )
                kontext_kwargs = {} if ip_img is None else {"ip_adapter_image": ip_img}
                result = self.kontext(
                    prompt=req.prompt,
                    image=image,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance,
                    generator=generator,
                    callback_on_step_end=_on_step_end,
                    **kontext_kwargs,
                )
                return result.images[0]

            if req.mode == "inpaint":
                if req.mask is None:
                    raise ValueError("inpaint mode requires a mask image")
                mask = ensure_l(req.mask).resize(image.size)
                self._set_accel_active(self.fill, req.use_accel)
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

            if req.mode == "qwen":
                # Qwen uses true_cfg_scale (default 4.0), not guidance_scale.
                # Map req.guidance -> true_cfg_scale; req.steps -> num_inference_steps.
                # No accel LoRA — Hyper-SD is Flux-architecture specific.
                result = self.qwen(
                    prompt=req.prompt,
                    image=image,
                    num_inference_steps=req.steps,
                    true_cfg_scale=req.guidance,
                    generator=generator,
                    callback_on_step_end=_on_step_end,
                )
                return result.images[0]

            if req.mode == "generate":
                # Flux schnell: 4-step distilled t2i. No input image. The
                # accel LoRA toggle still applies if configured (rare but
                # not harmful — Hyper-SD pushes schnell to ~1-2 steps).
                self._set_active_adapters(
                    self.schnell,
                    use_accel=req.use_accel,
                    style=style,
                    style_scale=req.style_lora_scale,
                )
                ip_img = self._apply_ip_adapter(
                    self.schnell, req.ip_adapter_image, req.ip_adapter_scale
                )
                schnell_kwargs = {} if ip_img is None else {"ip_adapter_image": ip_img}
                # schnell's guidance is baked in via distillation; passing
                # guidance_scale=0 is the recommended sentinel.
                result = self.schnell(
                    prompt=req.prompt,
                    height=req.height,
                    width=req.width,
                    num_inference_steps=req.steps,
                    guidance_scale=0.0,
                    generator=generator,
                    callback_on_step_end=_on_step_end,
                    **schnell_kwargs,
                )
                return result.images[0]

            raise ValueError(f"unknown mode: {req.mode}")
        finally:
            self._release_intermediate_memory()
