"""Minimal coverage for the bits of backend.vision that don't need
GPU / network — the constructor, the dataclass, and the polygon
rasterizer. Anything that touches a real model lives behind
``# pragma: no cover`` in the module."""

from __future__ import annotations

from PIL import Image

from backend.vision import (
    BLIP_DEEP_MODEL,
    BLIP_FAST_MODEL,
    CLIPSEG_MODEL,
    DETR_MODEL,
    OWLV2_MODEL,
    DetectedObject,
    VisionAnalyzer,
    _draw_polygons,
)


def test_model_ids_are_huggingface_repo_strings():
    """Sanity: each MODEL constant points at the expected upstream repo."""
    assert CLIPSEG_MODEL.startswith("CIDAS/")
    assert DETR_MODEL.startswith("facebook/")
    assert OWLV2_MODEL.startswith("google/")
    assert BLIP_FAST_MODEL.startswith("Salesforce/")
    assert BLIP_DEEP_MODEL.startswith("Salesforce/")
    assert BLIP_FAST_MODEL != BLIP_DEEP_MODEL


def test_detected_object_default_score_is_none():
    o = DetectedObject(label="cat", box=(0.0, 0.0, 10.0, 10.0))
    assert o.label == "cat"
    assert o.box == (0.0, 0.0, 10.0, 10.0)
    assert o.score is None


def test_detected_object_with_score():
    o = DetectedObject(label="dog", box=(1.0, 2.0, 3.0, 4.0), score=0.87)
    assert o.score == 0.87


def test_vision_analyzer_constructor_does_not_load_models():
    """Instantiating must NOT trigger any download — model fields stay None
    until the corresponding _load_* is called for the first time."""
    v = VisionAnalyzer()
    assert v._clipseg is None
    assert v._detr is None
    assert v._owlv2 is None
    assert v._blip_fast is None
    assert v._blip_deep is None


def test_draw_polygons_empty_input_returns_blank_mask():
    mask = _draw_polygons((32, 32), [])
    assert mask.size == (32, 32)
    assert mask.mode == "L"
    # All pixels black — getextrema returns (min, max).
    assert mask.getextrema() == (0, 0)


def test_draw_polygons_ignores_too_short_polygons():
    """Polygons with fewer than 3 points (6 coords) are silently skipped."""
    mask = _draw_polygons((32, 32), [[1.0, 2.0], [1.0, 2.0, 3.0]])
    assert mask.getextrema() == (0, 0)


def test_draw_polygons_rasterizes_a_triangle():
    triangle = [10.0, 5.0, 25.0, 5.0, 17.0, 25.0]
    mask = _draw_polygons((32, 32), [triangle])
    # At least the centroid pixel should be filled.
    assert mask.getpixel((17, 10)) == 255


def test_vision_analyzer_release_idempotent():
    """release() on a fresh instance with no loaded models must not error."""
    v = VisionAnalyzer()
    v.release()
    v.release()
    assert v._clipseg is None


def test_segment_rejects_empty_text():
    """Empty / whitespace-only text raises before any model load — verifies
    the early-exit guard without touching CLIPSeg weights."""
    import pytest

    v = VisionAnalyzer()
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    with pytest.raises(ValueError, match="text prompt is required"):
        v.segment(img, "   ")
