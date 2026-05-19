"""Pure-Python image helpers. No torch — safe to import in CI without CUDA."""

from __future__ import annotations

import io

from PIL import Image


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def ensure_l(img: Image.Image) -> Image.Image:
    return img if img.mode == "L" else img.convert("L")


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
