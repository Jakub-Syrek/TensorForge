"""IP-Adapter for FLUX — image-as-prompt generation via InstantX's stack.

Standard ``pipe.load_ip_adapter()`` does not work with the InstantX FLUX
IP-Adapter weights — the upstream ships custom inference code instead.
We embed those modules in ``backend/ipa_vendored/`` and wrap their
``IPAdapter`` helper class here in a TensorForge-friendly lazy generator
mirroring ``backend/controlnet.py`` and ``backend/pulid.py``.

How it differs from ControlNet / PuLID:
  - ControlNet: spatial conditioning (pose / depth / canny → boxes pixels)
  - PuLID: identity conditioning (face vector → keep recognizable face)
  - **IP-Adapter: visual-style conditioning (reference image → inherit
    aesthetic, colour palette, composition without verbal description)**

Stack
-----
- FLUX.1-dev base (~24 GB, shared with ControlNet / PuLID — paid once)
- SigLIP-SO400M-Patch14-384 image encoder (~1.5 GB)
- InstantX/FLUX.1-dev-IP-Adapter weights (``ip-adapter.bin``, ~1 GB)
- num_tokens=128 image-projection tokens (matches InstantX example)

VRAM (NF4 transformer + bf16 SigLIP + bf16 attn processors): ~8 GB
resident. Cannot coexist with warm Kontext / Fill on a 16 GB card —
the worker releases the FluxEditor's intermediate buffers before
loading.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
from PIL import Image

from backend.imgutils import ensure_rgb

log = logging.getLogger(__name__)

IPA_REPO = "InstantX/FLUX.1-dev-IP-Adapter"
IPA_WEIGHT = "ip-adapter.bin"
SIGLIP_REPO = "google/siglip-so400m-patch14-384"
FLUX_DEV_MODEL = "black-forest-labs/FLUX.1-dev"
# Number of image-projection tokens. 128 matches the InstantX example
# (their default in infer_flux_ipa_siglip.py).
IPA_NUM_TOKENS = 128


@dataclass
class IPARequest:
    """One image-conditioned generation request."""

    prompt: str
    ref_image: Image.Image
    ip_scale: float = 0.7
    steps: int = 24
    guidance: float = 3.5
    seed: int | None = None
    width: int = 1024
    height: int = 1024


class IPAdapterGenerator:
    """Lazy holder for InstantX's FLUX IP-Adapter stack."""

    def __init__(self) -> None:
        self._ip_model = None  # backend.ipa_vendored ``IPAdapter`` wrapper instance
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _load(self) -> None:  # pragma: no cover — needs GPU + multi-GB downloads
        if self._ip_model is not None:
            return
        from huggingface_hub import hf_hub_download

        from backend.ipa_vendored.attention_processor import (  # type: ignore[import-not-found]
            IPAFluxAttnProcessor2_0,
        )
        from backend.ipa_vendored.pipeline_flux_ipa import (  # type: ignore[import-not-found]
            FluxPipeline as IPAFluxPipeline,
        )
        from backend.ipa_vendored.transformer_flux import (  # type: ignore[import-not-found]
            FluxTransformer2DModel as IPAFluxTransformer2DModel,
        )
        from backend.pipeline import QUANT_MODE

        log.info("loading FLUX IP-Adapter (InstantX) — base + SigLIP + adapter")

        # Build the transformer with the same NF4 quant path the main
        # editor uses when QUANT_MODE=4bit. InstantX's custom
        # FluxTransformer2DModel inherits from ``ModelMixin`` so it
        # accepts ``quantization_config`` like the stock class.
        if QUANT_MODE == "4bit":
            from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
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
            transformer = IPAFluxTransformer2DModel.from_pretrained(  # nosec B615
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
            pipe = IPAFluxPipeline.from_pretrained(  # nosec B615
                FLUX_DEV_MODEL,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=self._dtype,
            )
        else:
            transformer = IPAFluxTransformer2DModel.from_pretrained(  # nosec B615
                FLUX_DEV_MODEL,
                subfolder="transformer",
                torch_dtype=self._dtype,
            )
            pipe = IPAFluxPipeline.from_pretrained(  # nosec B615
                FLUX_DEV_MODEL, transformer=transformer, torch_dtype=self._dtype
            )

        if torch.cuda.is_available():
            if QUANT_MODE == "4bit":
                pipe.to("cuda")
            else:
                pipe.enable_model_cpu_offload()
        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        # Download the adapter weights to a local path so the InstantX
        # IPAdapter helper can ``torch.load`` from disk.
        # nosec B615 — InstantX is trusted upstream; pinning a revision
        # would block legit fixes for a single-user desktop app.
        ip_ckpt_path = hf_hub_download(  # nosec B615
            repo_id=IPA_REPO, filename=IPA_WEIGHT
        )

        # InstantX's IPAdapter wrapper builds the image-projection MLP,
        # attaches the custom attention processors to the transformer
        # blocks, and loads the adapter weights.
        ip_model = _InstantXIPAdapter(
            pipe=pipe,
            image_encoder_path=SIGLIP_REPO,
            ip_ckpt=ip_ckpt_path,
            num_tokens=IPA_NUM_TOKENS,
            ipa_attn_processor_cls=IPAFluxAttnProcessor2_0,
            dtype=self._dtype,
        )
        self._ip_model = ip_model
        log.info("FLUX IP-Adapter ready")

    def generate(self, req: IPARequest) -> Image.Image:  # pragma: no cover — GPU
        """Run one IP-Adapter-conditioned generation. Returns a single PIL image."""
        self._load()
        ref = ensure_rgb(req.ref_image)
        images = self._ip_model.generate(
            pil_image=ref,
            prompt=req.prompt,
            scale=req.ip_scale,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            seed=req.seed,
            width=req.width,
            height=req.height,
        )
        return images[0]

    def release(self) -> None:  # pragma: no cover — GPU
        """Drop the pipeline + image encoder + attention adapters from
        VRAM. Useful before loading a different FLUX flavor (Kontext /
        Fill / schnell / ControlNet / PuLID) on a 16 GB card."""
        if self._ip_model is not None:
            del self._ip_model
            self._ip_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class _InstantXIPAdapter:  # pragma: no cover — GPU + multi-GB load
    """Adapted from InstantX's ``IPAdapter`` class in
    ``infer_flux_ipa_siglip.py``. Kept verbatim in spirit but inlined
    here so we don't need to depend on a file outside our package
    that imports siblings via implicit-relative imports.

    Responsibilities:
      - Hold the FLUX pipeline and run inference through it
      - Build the image-projection MLP (id_embeddings_dim=1152 ← SigLIP
        ``pooler_output``, cross_attention_dim=4096 ← FLUX joint_attention_dim)
      - Replace stock attention processors on transformer blocks with
        ``IPAFluxAttnProcessor2_0`` — the ones that read the image
        prompt embeddings
      - Load ``ip-adapter.bin`` weights into the MLP + attention layers
      - Provide ``generate(pil_image, prompt, scale, ...)`` matching
        the upstream API
    """

    def __init__(
        self,
        *,
        pipe,
        image_encoder_path: str,
        ip_ckpt: str,
        num_tokens: int,
        ipa_attn_processor_cls,
        dtype: torch.dtype,
    ) -> None:
        from transformers import AutoProcessor, SiglipVisionModel

        self.pipe = pipe
        self.num_tokens = num_tokens
        self.dtype = dtype
        self.device = next(pipe.transformer.parameters()).device
        self._ipa_attn_processor_cls = ipa_attn_processor_cls

        # SigLIP image encoder — InstantX uses pooler_output (1152-dim).
        self.image_encoder = SiglipVisionModel.from_pretrained(  # nosec B615
            image_encoder_path
        ).to(self.device, dtype=dtype)
        self.clip_image_processor = AutoProcessor.from_pretrained(  # nosec B615
            image_encoder_path
        )

        self.image_proj_model = self._init_proj()
        self._set_ip_attn_processors()
        self._load_adapter_weights(ip_ckpt)

    def _init_proj(self):
        model = _MLPProjModel(
            cross_attention_dim=self.pipe.transformer.config.joint_attention_dim,
            id_embeddings_dim=1152,  # SigLIP pooler_output dim
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.dtype)
        return model

    def _set_ip_attn_processors(self):
        transformer = self.pipe.transformer
        new_procs = {}
        for name in transformer.attn_processors:
            if name.startswith(("transformer_blocks.", "single_transformer_blocks")):
                new_procs[name] = self._ipa_attn_processor_cls(
                    hidden_size=transformer.config.num_attention_heads
                    * transformer.config.attention_head_dim,
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.dtype)
            else:
                new_procs[name] = transformer.attn_processors[name]
        transformer.set_attn_processor(new_procs)

    def _load_adapter_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        ip_layers = torch.nn.ModuleList(self.pipe.transformer.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def _embed_reference(self, pil_image: Image.Image) -> torch.Tensor:
        pixel_values = self.clip_image_processor(
            images=[pil_image], return_tensors="pt"
        ).pixel_values
        clip_embeds = self.image_encoder(
            pixel_values.to(self.device, dtype=self.image_encoder.dtype)
        ).pooler_output
        clip_embeds = clip_embeds.to(dtype=self.dtype)
        return self.image_proj_model(clip_embeds)

    def _set_scale(self, scale: float) -> None:
        for proc in self.pipe.transformer.attn_processors.values():
            if isinstance(proc, self._ipa_attn_processor_cls):
                proc.scale = scale

    def generate(
        self,
        *,
        pil_image: Image.Image,
        prompt: str,
        scale: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        width: int,
        height: int,
    ):
        self._set_scale(scale)
        image_prompt_embeds = self._embed_reference(pil_image)
        generator = (
            torch.Generator(self.device).manual_seed(int(seed)) if seed is not None else None
        )
        result = self.pipe(
            prompt=prompt,
            image_emb=image_prompt_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images


class _MLPProjModel(torch.nn.Module):  # pragma: no cover — GPU
    """Image-projection MLP from InstantX's ``infer_flux_ipa_siglip.py``.
    Projects the SigLIP pooler_output (1152-dim) into ``num_tokens``
    embeddings of size ``cross_attention_dim`` (4096 for FLUX)."""

    def __init__(
        self,
        cross_attention_dim: int = 4096,
        id_embeddings_dim: int = 1152,
        num_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


__all__ = [
    "FLUX_DEV_MODEL",
    "IPA_NUM_TOKENS",
    "IPA_REPO",
    "IPA_WEIGHT",
    "SIGLIP_REPO",
    "IPARequest",
    "IPAdapterGenerator",
]
