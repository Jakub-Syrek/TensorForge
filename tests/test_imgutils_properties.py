"""Property-based tests for backend.imgutils.fit_long_edge.

Hypothesis generates hundreds of (w, h, max_edge) tuples and verifies that
the function holds the same invariants across the whole input space, not
just the few example points covered by the explicit unit tests.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from PIL import Image

from backend.imgutils import fit_long_edge

# Bounded ranges keep test runtime sane while still covering realistic inputs:
# images up to 8K, caps up to 1024.
dims = st.integers(min_value=16, max_value=8192)
caps = st.integers(min_value=64, max_value=1024)

# Disable per-example deadline. LANCZOS resize of an 8K dummy image can exceed
# Hypothesis' default 200 ms — that's PIL's runtime, not a code-under-test bug.
_resize_settings = settings(deadline=None, max_examples=100)


@_resize_settings
@given(w=dims, h=dims, cap=caps)
def test_long_edge_never_exceeds_cap(w, h, cap):
    img = Image.new("RGB", (w, h))
    out = fit_long_edge(img, max_edge=cap)
    assert max(out.size) <= cap


@_resize_settings
@given(w=dims, h=dims, cap=caps)
def test_output_dims_divisible_by_16(w, h, cap):
    img = Image.new("RGB", (w, h))
    out = fit_long_edge(img, max_edge=cap)
    ow, oh = out.size
    assert ow % 16 == 0
    assert oh % 16 == 0


@_resize_settings
@given(w=dims, h=dims, cap=caps)
def test_never_upscales(w, h, cap):
    img = Image.new("RGB", (w, h))
    out = fit_long_edge(img, max_edge=cap)
    assert out.size[0] <= w
    assert out.size[1] <= h


@_resize_settings
@given(
    # AR-preservation only holds when both dims are well above the /16 floor;
    # at small dims the rounding-down-to-multiple-of-16 dominates AR drift by
    # design. Use a tighter strategy so the invariant is meaningful.
    w=st.integers(min_value=256, max_value=8192),
    h=st.integers(min_value=256, max_value=8192),
    cap=st.integers(min_value=256, max_value=1024),
)
def test_aspect_ratio_preserved_within_rounding_tolerance(w, h, cap):
    """AR drift is bounded by the /16 rounding: each side can lose up to 15
    px, so combined relative drift is ~30/min(ow, oh). At dim=256 that's
    ~12% tolerance; at dim=1024 it's ~3%. The tighter tests
    (output_dims_divisible_by_16, never_exceeds_cap, never_upscales) cover
    the strict structural invariants — this one just ensures we don't
    catastrophically distort the image."""
    img = Image.new("RGB", (w, h))
    out = fit_long_edge(img, max_edge=cap)
    ow, oh = out.size
    assume(ow > 0 and oh > 0)
    ar_ratio = (ow / oh) / (w / h)
    tolerance = 32.0 / min(ow, oh)  # generous: ~2x the per-side rounding loss
    assert abs(ar_ratio - 1.0) <= tolerance


@_resize_settings
@given(w=dims, h=dims, cap=caps)
def test_already_small_returns_same_instance(w, h, cap):
    """When input is already at or below the cap (and aligned to /16),
    output should be the same object — no needless copy."""
    if max(w, h) <= cap and w % 16 == 0 and h % 16 == 0:
        img = Image.new("RGB", (w, h))
        assert fit_long_edge(img, max_edge=cap) is img
