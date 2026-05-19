from io import BytesIO

from PIL import Image

from backend.imgutils import ensure_l, ensure_rgb, image_to_png_bytes


def test_ensure_rgb_passthrough():
    img = Image.new("RGB", (4, 4), color=(10, 20, 30))
    assert ensure_rgb(img) is img


def test_ensure_rgb_converts_rgba():
    img = Image.new("RGBA", (4, 4), color=(10, 20, 30, 255))
    out = ensure_rgb(img)
    assert out.mode == "RGB"
    assert out.size == img.size


def test_ensure_l_passthrough():
    img = Image.new("L", (4, 4), color=128)
    assert ensure_l(img) is img


def test_ensure_l_converts_rgb():
    img = Image.new("RGB", (4, 4), color=(255, 255, 255))
    out = ensure_l(img)
    assert out.mode == "L"


def test_image_to_png_bytes_roundtrip():
    img = Image.new("RGB", (8, 8), color=(200, 100, 50))
    blob = image_to_png_bytes(img)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"

    decoded = Image.open(BytesIO(blob))
    assert decoded.size == (8, 8)
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (200, 100, 50)
