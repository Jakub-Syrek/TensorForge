"""Pure-Python image helpers. No torch — safe to import in CI without CUDA."""

from __future__ import annotations

import io
from typing import Literal

from PIL import Image, ImageFilter

SharpenLevel = Literal["off", "light", "medium", "strong"]

# Tuned for our pipeline's LANCZOS upscale (typically 1024 -> original size).
# Stronger upscales (e.g. 1024 -> 4032) soften more, so higher levels exist;
# 'light' is conservative, 'strong' will produce visible halos on hard edges
# and should be reserved for content with smooth tones.
_SHARPEN_PARAMS = {
    "light": {"radius": 1, "percent": 100, "threshold": 3},
    "medium": {"radius": 2, "percent": 150, "threshold": 3},
    "strong": {"radius": 3, "percent": 200, "threshold": 2},
}


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def ensure_l(img: Image.Image) -> Image.Image:
    return img if img.mode == "L" else img.convert("L")


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sharpen(img: Image.Image, level: SharpenLevel) -> Image.Image:
    """Apply PIL UnsharpMask at one of three intensities; no-op when 'off'.

    Edge enhancement only — doesn't invent detail. Counteracts the softening
    introduced by LANCZOS upscale back to the original input dimensions. For
    real super-resolution use a learned upscaler (Real-ESRGAN etc.) instead;
    that's a separate post-process not in scope here.
    """
    if level == "off" or level not in _SHARPEN_PARAMS:
        return img
    params = _SHARPEN_PARAMS[level]
    return img.filter(ImageFilter.UnsharpMask(**params))


def fit_long_edge(img: Image.Image, max_edge: int, multiple_of: int = 16) -> Image.Image:
    """Downscale `img` so its longest edge is at most `max_edge`, rounded to
    a multiple of `multiple_of` (Flux VAE wants dims divisible by 16).

    Never upscales — if the image is already small, returns it unchanged.
    """
    w, h = img.size
    longest = max(w, h)
    scale = min(1.0, max_edge / longest) if longest > 0 else 1.0

    def _round(v: int) -> int:
        return max(multiple_of, (round(v * scale) // multiple_of) * multiple_of)

    new_w, new_h = _round(w), _round(h)
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
