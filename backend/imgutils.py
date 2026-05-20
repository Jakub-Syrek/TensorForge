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


# Flux Kontext / Flux-dev training aspect buckets at the 1024-long-edge tier.
# When the user's input doesn't land on a bucket, the model internally
# snaps to the nearest one — and that snap CROPS content from the wide
# axis. We do the snap ourselves explicitly so the model has nothing left
# to bucket and the output's content stays in frame (it just gets a small
# aspect-ratio drift that the LANCZOS round-trip then absorbs).
FLUX_BUCKETS_1024 = (
    (1024, 1024),  # 1:1
    (1152, 896),  # 4:3 landscape (~1.29)
    (896, 1152),  # 3:4 portrait  (~0.78)
    (1216, 832),  # 3:2 landscape (~1.46)
    (832, 1216),  # 2:3 portrait  (~0.68)
    (1344, 768),  # 16:9 landscape (~1.75)
    (768, 1344),  # 9:16 portrait  (~0.57)
    (1536, 640),  # 12:5 landscape (2.40)
    (640, 1536),  # 5:12 portrait  (~0.42)
)


def fit_to_flux_bucket(img: Image.Image, max_edge: int = 1024) -> Image.Image:
    """Snap input dimensions to the closest Flux training-aspect bucket,
    scaled proportionally to `max_edge` (default 1024 = bucket baseline).

    The Flux Kontext pipeline buckets aspect ratios internally during
    inference; if the input doesn't match a bucket exactly, the model
    side-crops content to fit. Pre-bucketing on the server keeps every
    pixel in frame at the cost of a small explicit aspect adjustment
    (typically a few percent), which the round-trip LANCZOS upscale to
    the original input dimensions reverses cleanly.

    Returns the image unchanged if it's already at the bucket size.
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    aspect = w / h

    # Closest bucket by aspect-ratio L1 distance.
    bw, bh = min(FLUX_BUCKETS_1024, key=lambda b: abs(b[0] / b[1] - aspect))

    # Scale the chosen bucket to `max_edge` proportionally. We treat 1024 as
    # the bucket baseline (it's the long-edge of the 1:1 / 4:3 family).
    scale = max_edge / 1024 if max_edge > 0 else 1.0
    new_w = max(16, (round(bw * scale) // 16) * 16)
    new_h = max(16, (round(bh * scale) // 16) * 16)

    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


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
