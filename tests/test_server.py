"""Routing-level tests for the FastAPI app.

We don't load Flux — that's a multi-GB GPU operation. Instead we replace
backend.server.editor with a stub that records the request and returns a
canned PIL image. This catches regressions in routing, validation, and
multipart handling without touching the GPU.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.server as srv
from backend.pipeline import EditRequest


class _StubEditor:
    def __init__(self):
        self.accel = None
        self.last_request: EditRequest | None = None

    def edit(self, req: EditRequest) -> Image.Image:
        self.last_request = req
        # Encode the mode into a recognizable color so the test can verify
        # which branch ran without parsing the request log.
        color = (255, 0, 0) if req.mode == "kontext" else (0, 255, 0)
        return Image.new("RGB", req.image.size, color=color)


@pytest.fixture
def stub_editor(monkeypatch):
    stub = _StubEditor()
    monkeypatch.setattr(srv, "editor", stub)
    return stub


@pytest.fixture
def client():
    return TestClient(srv.app)


def _png_bytes(size=(32, 32), color=(50, 50, 50)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_health_reports_stub_torch(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["torch"] == "stub"
    assert body["cuda_available"] is False
    assert body["device"] == "cpu"


def test_edit_kontext_returns_png(client, stub_editor):
    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "make it sunset", "steps": "8", "guidance": "3.5"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"

    out = Image.open(BytesIO(r.content))
    assert out.getpixel((0, 0)) == (255, 0, 0)  # kontext branch

    req = stub_editor.last_request
    assert req is not None
    assert req.mode == "kontext"
    assert req.prompt == "make it sunset"
    assert req.steps == 8
    assert req.mask is None


def test_edit_inpaint_requires_mask(client, stub_editor):
    r = client.post(
        "/api/edit",
        data={"mode": "inpaint", "prompt": "fill it"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400
    assert "mask" in r.text.lower()


def test_edit_inpaint_accepts_mask(client, stub_editor):
    r = client.post(
        "/api/edit",
        data={"mode": "inpaint", "prompt": "fill it", "steps": "8"},
        files={
            "image": ("in.png", _png_bytes(), "image/png"),
            "mask": ("m.png", _png_bytes(color=(255, 255, 255)), "image/png"),
        },
    )
    assert r.status_code == 200, r.text
    out = Image.open(BytesIO(r.content))
    assert out.getpixel((0, 0)) == (0, 255, 0)  # inpaint branch
    assert stub_editor.last_request.mask is not None


def test_edit_rejects_unknown_mode(client, stub_editor):
    r = client.post(
        "/api/edit",
        data={"mode": "doodle", "prompt": "x"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400


def test_edit_round_trips_to_original_size(client, stub_editor):
    """Server downscales input to MAX_EDGE for inference, then restores
    the original canvas before returning. The stub editor echoes whatever
    size it received — so the response should be back at the upload size."""
    from backend.server import MAX_EDGE

    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "x", "steps": "1"},
        files={"image": ("big.png", _png_bytes(size=(2048, 2048)), "image/png")},
    )
    assert r.status_code == 200, r.text
    out = Image.open(BytesIO(r.content))
    assert out.size == (2048, 2048)
    # And the editor saw the downscaled size, not the upload size.
    assert max(stub_editor.last_request.image.size) == MAX_EDGE


def test_edit_rejects_empty_prompt(client, stub_editor):
    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "   "},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400


def test_progress_endpoint_returns_idle_shape(client):
    from backend import server as srv

    # Reset job state — previous tests in the same session may have left it active.
    srv.job_progress.finish()
    srv.job_progress.active = False
    srv.job_progress.step = 0
    srv.job_progress.total = 0

    r = client.get("/api/progress")
    assert r.status_code == 200
    body = r.json()
    assert "job" in body
    assert body["job"]["active"] is False
    assert body["job"]["percent"] == 0.0
    # gpu is None in CI (no nvidia-smi); on the user's machine it'd be populated
    assert "gpu" in body


def test_abort_when_idle_returns_409(client):
    from backend import server as srv

    srv.job_progress.active = False
    r = client.post("/api/abort")
    assert r.status_code == 409
    body = r.json()
    assert body["aborted"] is False
    assert body["active"] is False


def test_abort_when_active_returns_200_and_sets_flag(client):
    from backend import server as srv

    srv.job_progress.start(total=28, mode="kontext")
    try:
        r = client.post("/api/abort")
        assert r.status_code == 200
        body = r.json()
        assert body["aborted"] is True
        assert srv.job_progress.aborted is True
    finally:
        srv.job_progress.finish()


def test_edit_passes_use_accel_through_to_request(client, stub_editor):
    """The use_accel form field should land in EditRequest.use_accel and
    default to True if the caller omits it."""
    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "x", "steps": "1", "use_accel": "false"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert stub_editor.last_request.use_accel is False

    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "x", "steps": "1"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert stub_editor.last_request.use_accel is True


def test_edit_returns_used_seed_header_when_caller_supplies_seed(client, stub_editor):
    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "x", "steps": "1", "seed": "12345"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert r.headers.get("X-Used-Seed") == "12345"
    assert stub_editor.last_request.seed == 12345


def test_edit_generates_seed_when_caller_omits(client, stub_editor):
    """No seed in form → server picks one and surfaces it in the header so
    the client can pin it for the next run (A/B iteration workflow)."""
    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "x", "steps": "1"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    used = r.headers.get("X-Used-Seed")
    assert used is not None
    seed_int = int(used)
    assert 0 <= seed_int < 2**31
    # Same value made it into the EditRequest, not None.
    assert stub_editor.last_request.seed == seed_int


def test_edit_marks_progress_finished_after_success(client, stub_editor):
    from backend import server as srv

    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "x", "steps": "4"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert srv.job_progress.active is False
    assert srv.job_progress.last_error is None
    assert srv.job_progress.total == 4
