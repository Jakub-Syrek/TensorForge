"""Style LoRA registry for the FLUX pipelines.

A style LoRA is a small adapter (typically 50-300 MB) that biases the base
FLUX.1 model toward a specific aesthetic — fantasy painting, retro sci-fi,
anime, etc. The base checkpoint is shared; only the LoRA delta swaps in.

How loading works at runtime
----------------------------
- LoRAs are loaded lazily on first use into the target pipeline via
  ``pipe.load_lora_weights(...)`` and tracked per pipeline instance.
- The acceleration LoRA (Hyper-SD / Flux-Turbo) is a separate adapter —
  style + accel are combined by ``set_adapters([...], [weights])`` so you
  can run "8-step fantasy" without rebuilding the pipe.
- Switching between styles is a cheap ``set_adapters`` call once both have
  been loaded.

Compatibility
-------------
- ``kontext`` (FLUX.1-Kontext-dev) and ``generate`` (FLUX.1-schnell) share
  the Flux DiT transformer, so dev-trained LoRAs load on both. Quality on
  Kontext is unofficial but usually fine — the LoRA biases style, Kontext
  still preserves composition.
- ``inpaint`` (FLUX.1-Fill-dev) and ``qwen`` (Qwen-Image-Edit) are out of
  scope: Fill is rarely used with style LoRAs, and Qwen is a different
  architecture.

Adding more LoRAs
-----------------
Append a ``StyleLoRA(...)`` entry to ``STYLE_LORAS`` below. The minimum is
``repo`` (HF id) + ``weight_name`` (the .safetensors filename inside the
repo). Set ``trigger`` to a short prompt prefix the model expects (some
LoRAs need a keyword like "ohwx style" to activate fully); the UI surfaces
it as a hint, it is NOT auto-injected into the prompt.

The registry intentionally ships small. Picking LoRAs is a taste call —
browse Hugging Face / CivitAI and drop the ones you like in here.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StyleLoRA:
    id: str
    """Short, URL-safe identifier sent from the UI."""

    label: str
    """Display name shown in the dropdown."""

    repo: str
    """Hugging Face repo id, e.g. 'XLabs-AI/flux-RealismLora'."""

    weight_name: str
    """Safetensors filename inside the repo."""

    scale: float = 1.0
    """Default adapter weight. UI may override per-request."""

    modes: tuple[str, ...] = ("generate", "kontext")
    """Pipelines this LoRA is allowed to bind to."""

    trigger: str = ""
    """Optional prompt prefix the LoRA was trained with. Shown as a hint."""

    description: str = ""
    """Short human description for the UI tooltip."""


# Starter set. Most are well-known FLUX.1-dev LoRAs from Hugging Face;
# the schnell pipeline accepts them because it shares the same DiT.
# Edit / extend freely.
_BUILTIN: tuple[StyleLoRA, ...] = (
    StyleLoRA(
        id="realism",
        label="XLabs Realism",
        repo="XLabs-AI/flux-RealismLora",
        weight_name="lora.safetensors",
        scale=1.0,
        trigger="",
        description="Photoreal portraits / scenes. Tames Flux's painterly drift.",
    ),
    StyleLoRA(
        id="koda",
        label="Koda (analog film)",
        repo="alvdansen/flux-koda",
        weight_name="araminta_k_flux_koda.safetensors",
        scale=1.0,
        trigger="flmft style",
        description="Soft analog-film color palette, vintage Kodak vibe.",
    ),
    StyleLoRA(
        id="tarot",
        label="Tarot card",
        repo="multimodalart/flux-tarot-v1",
        weight_name="flux_tarot_v1_lora.safetensors",
        scale=1.0,
        trigger="in the style of TOK a trtcrd tarot style",
        description="Ornate medieval-tarot framing — strong fantasy bias.",
    ),
    StyleLoRA(
        id="ghibsky",
        label="Ghibsky (Ghibli-esque)",
        repo="aleksa-codes/flux-ghibsky-illustration",
        weight_name="lora.safetensors",
        scale=1.0,
        trigger="GHIBSKY style",
        description="Studio-Ghibli-flavored illustration, lush backgrounds.",
    ),
    StyleLoRA(
        id="anime",
        label="Synthetic Anime",
        repo="dataautogpt3/FLUX-AestheticAnime",
        weight_name="Flux_1_Dev_LoRA_AestheticAnime.safetensors",
        scale=1.0,
        trigger="",
        description="Clean anime/manga lineart aesthetic.",
    ),
)


def _load_user_registry() -> tuple[StyleLoRA, ...]:
    """Optional user-supplied JSON registry merged with the built-in set.

    Path resolution:
      1. ``AIPIC_LORA_REGISTRY`` env var (absolute or repo-relative)
      2. ``loras.json`` next to this file

    Format: JSON list of objects with the same fields as ``StyleLoRA``.
    Entries override built-ins of the same ``id``.
    """
    candidates: list[Path] = []
    env_path = os.environ.get("AIPIC_LORA_REGISTRY")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).with_name("loras.json"))

    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("lora registry %s unreadable: %s", path, exc)
            return ()
        if not isinstance(raw, list):
            log.warning("lora registry %s must be a JSON array", path)
            return ()
        out: list[StyleLoRA] = []
        for i, entry in enumerate(raw):
            try:
                out.append(StyleLoRA(**entry))
            except TypeError as exc:
                log.warning("lora registry %s entry %d invalid: %s", path, i, exc)
        log.info("loaded %d user-defined LoRAs from %s", len(out), path)
        return tuple(out)

    return ()


def _build_registry() -> dict[str, StyleLoRA]:
    out: dict[str, StyleLoRA] = {lora.id: lora for lora in _BUILTIN}
    for lora in _load_user_registry():
        out[lora.id] = lora  # user override wins
    return out


STYLE_LORAS: dict[str, StyleLoRA] = _build_registry()


def get(lora_id: str | None) -> StyleLoRA | None:
    """Look up a LoRA by id. None/empty returns None (no style)."""
    if not lora_id:
        return None
    return STYLE_LORAS.get(lora_id)


def is_compatible(lora: StyleLoRA, mode: str) -> bool:
    return mode in lora.modes


def as_public_list() -> list[dict]:
    """Serializable form for the ``GET /api/loras`` endpoint."""
    return [asdict(lora) for lora in STYLE_LORAS.values()]


__all__ = ["STYLE_LORAS", "StyleLoRA", "as_public_list", "get", "is_compatible"]
