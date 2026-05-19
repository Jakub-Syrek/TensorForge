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
