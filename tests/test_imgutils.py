from io import BytesIO

from PIL import Image

from backend.imgutils import (
    FLUX_BUCKETS_1024,
    ensure_l,
    ensure_rgb,
    fit_long_edge,
    fit_to_flux_bucket,
    image_to_png_bytes,
    sharpen,
)


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


def test_fit_to_flux_bucket_square_input_snaps_to_1024_square():
    img = Image.new("RGB", (1024, 1024))
    out = fit_to_flux_bucket(img, max_edge=1024)
    assert out is img  # already a bucket


def test_fit_to_flux_bucket_portrait_phone_aspect_snaps_to_3to4():
    """3024x4032 phone shot (aspect ~0.75) -> closest bucket is 896x1152 (~0.78)."""
    img = Image.new("RGB", (3024, 4032))
    out = fit_to_flux_bucket(img, max_edge=1024)
    assert out.size == (896, 1152)


def test_fit_to_flux_bucket_landscape_3to2_snaps():
    """4032x2688 (aspect 1.5) -> closest 1216x832 (1.46)."""
    img = Image.new("RGB", (4032, 2688))
    out = fit_to_flux_bucket(img, max_edge=1024)
    assert out.size == (1216, 832)


def test_fit_to_flux_bucket_scales_with_max_edge():
    """max_edge=512 should halve all bucket dimensions (proportional scale)."""
    img = Image.new("RGB", (1024, 1024))
    out = fit_to_flux_bucket(img, max_edge=512)
    assert out.size == (512, 512)


def test_fit_to_flux_bucket_output_dims_divisible_by_16():
    """Pick odd-ish aspects and verify the /16 alignment holds at lower
    max_edge where rounding can land off."""
    for src in [(1933, 2895), (2480, 3508), (3840, 2160), (1080, 1920)]:
        img = Image.new("RGB", src)
        out = fit_to_flux_bucket(img, max_edge=640)
        w, h = out.size
        assert w % 16 == 0, f"{src} -> {out.size}"
        assert h % 16 == 0, f"{src} -> {out.size}"


def test_fit_to_flux_bucket_all_baseline_buckets_are_divisible_by_16():
    """Constant-time sanity on the bucket list itself."""
    for w, h in FLUX_BUCKETS_1024:
        assert w % 16 == 0, (w, h)
        assert h % 16 == 0, (w, h)


def test_sharpen_off_returns_same_instance():
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    assert sharpen(img, "off") is img


def test_sharpen_unknown_level_is_noop():
    """Unknown levels are treated as 'off' — defensive, since the form-field
    value reaches us as an arbitrary string."""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    assert sharpen(img, "extreme") is img


def test_sharpen_light_modifies_image_on_edge_content():
    """UnsharpMask enhances mid-tone edges; 0/255 binary edges are already
    saturated and have no headroom to amplify. Use a grey-to-grey edge so
    we can detect the filter actually ran."""
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    # paste a slightly brighter grey to create a soft edge with headroom
    img.paste(Image.new("RGB", (32, 32), color=(160, 160, 160)), (16, 16))

    out = sharpen(img, "light")
    assert out is not img
    assert out.size == img.size
    # Tobytes() comparison: same content -> same bytes, modified -> different.
    assert img.tobytes() != out.tobytes()


def test_image_to_png_bytes_roundtrip():
    img = Image.new("RGB", (8, 8), color=(200, 100, 50))
    blob = image_to_png_bytes(img)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"

    decoded = Image.open(BytesIO(blob))
    assert decoded.size == (8, 8)
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (200, 100, 50)
