"""Background removal — isolate the subject from its surroundings.

Pairs naturally with the existing inpaint / Kontext flow: rip the
subject out of one scene, drop a different prompt-generated background
behind it. Also doubles as a generic alpha-mask producer for the auto-
mask pipeline ("everything except the subject" mask).

Implementation uses the ``rembg`` package with the BiRefNet "general"
model. Rembg avoids the transformers ``trust_remote_code`` route that
the upstream RMBG-2.0 / BiRefNet repos require (which gave us trouble
with Florence-2 earlier); models ship as ONNX files loaded by
``onnxruntime``, which is a separate stack from PyTorch. ONNX runtime
runs on CPU by default — for a single-user desktop tool this is plenty
fast (~1-2 s on a 1024² image) and keeps the FLUX VRAM budget intact.

The session caches the model in memory after the first call; no need
to roll our own loader.
"""

from __future__ import annotations

import logging
import os

from PIL import Image

log = logging.getLogger(__name__)

# BiRefNet "general" is rembg's top-quality general-purpose model — clean
# alpha edges on hair, fur, fabric. Slower than u2net but the quality
# difference is dramatic. Override via env var if a user wants something
# specific (e.g. "u2netp" for the lightweight ~5 MB model).
BG_REMOVE_MODEL = os.environ.get("AIPIC_BG_REMOVE_MODEL", "birefnet-general")


class BackgroundRemover:
    """Lazy holder for the rembg session."""

    def __init__(self) -> None:
        self._session = None

    def _load(self) -> None:  # pragma: no cover — needs the rembg model file download
        if self._session is not None:
            return
        from rembg import new_session

        log.info("loading background-removal session (%s)", BG_REMOVE_MODEL)
        self._session = new_session(BG_REMOVE_MODEL)
        log.info("background remover ready")

    def remove(  # pragma: no cover — exercises ONNX runtime
        self, image: Image.Image
    ) -> Image.Image:
        """Return ``image`` with its background made transparent.

        Output is RGBA at the input's native resolution. The alpha
        channel encodes per-pixel subject probability (continuous values,
        not binary — feathered edges around hair / fabric).
        """
        from rembg import remove

        self._load()
        # rembg.remove accepts PIL directly and returns PIL with alpha.
        return remove(image, session=self._session)

    def release(self) -> None:  # pragma: no cover — trivial
        """Drop the ONNX session. Frees the ~150 MB BiRefNet model from
        process memory; next call reloads."""
        self._session = None


__all__ = ["BG_REMOVE_MODEL", "BackgroundRemover"]
