from io import BytesIO

from PIL import Image

from backend.imgutils import ensure_l, ensure_rgb, fit_long_edge, image_to_png_bytes


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


def test_fit_long_edge_passthrough_when_already_small():
    img = Image.new("RGB", (512, 384), color=(0, 0, 0))
    out = fit_long_edge(img, max_edge=1024)
    assert out is img


def test_fit_long_edge_downscales_to_cap_keeping_aspect():
    img = Image.new("RGB", (4000, 3000), color=(0, 0, 0))
    out = fit_long_edge(img, max_edge=1024)
    w, h = out.size
    assert max(w, h) <= 1024
    assert w % 16 == 0 and h % 16 == 0
    # 4000:3000 = 4:3 → 1024:768 rounded to /16 → 1024:768
    assert (w, h) == (1024, 768)


def test_fit_long_edge_portrait_orientation():
    img = Image.new("RGB", (3000, 4000), color=(0, 0, 0))
    out = fit_long_edge(img, max_edge=1024)
    w, h = out.size
    assert max(w, h) <= 1024
    assert w % 16 == 0 and h % 16 == 0
    assert (w, h) == (768, 1024)


def test_fit_long_edge_rounds_to_multiple_of_16():
    img = Image.new("RGB", (1500, 1500), color=(0, 0, 0))
    out = fit_long_edge(img, max_edge=1000)
    w, h = out.size
    assert w % 16 == 0 and h % 16 == 0
    assert max(w, h) <= 1000  # never exceed the cap


def test_image_to_png_bytes_roundtrip():
    img = Image.new("RGB", (8, 8), color=(200, 100, 50))
    blob = image_to_png_bytes(img)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"

    decoded = Image.open(BytesIO(blob))
    assert decoded.size == (8, 8)
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (200, 100, 50)
