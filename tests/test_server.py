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
    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "x", "steps": "1"},
        files={"image": ("big.png", _png_bytes(size=(2048, 2048)), "image/png")},
    )
    assert r.status_code == 200, r.text
    out = Image.open(BytesIO(r.content))
    assert out.size == (2048, 2048)
    # And the editor saw the downscaled size, not the upload size.
    assert stub_editor.last_request.image.size == (1024, 1024)


def test_edit_rejects_empty_prompt(client, stub_editor):
    r = client.post(
        "/api/edit",
        data={"mode": "kontext", "prompt": "   "},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400
