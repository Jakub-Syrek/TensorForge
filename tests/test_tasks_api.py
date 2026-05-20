"""Integration tests for the /api/tasks endpoints via TestClient.

The worker thread isn't started here (lifespan only fires on real
server startup); we exercise the API and DB layer directly. To verify
the 'happy path' completion + approval, we manually mutate the DB
between requests to simulate what the worker would have done.
"""

from __future__ import annotations

import importlib
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPIC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AIPIC_DB_PATH", str(tmp_path / "test.db"))
    import backend.storage as s

    importlib.reload(s)
    import backend.db as d

    importlib.reload(d)
    d.init_db()
    s.init()
    import backend.queue as q

    importlib.reload(q)
    import backend.server as srv

    importlib.reload(srv)
    return srv, d, s


@pytest.fixture
def client(app_env):
    srv, _d, _s = app_env
    return TestClient(srv.app)


def _png_bytes(size=(32, 32), color=(50, 50, 50)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_create_task_returns_queued(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "remove glasses"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["mode"] == "kontext"
    assert body["prompt"] == "remove glasses"
    assert body["variants_requested"] == 1
    assert len(body["variants"]) == 1
    assert body["variants"][0]["status"] == "queued"


def test_create_task_multiple_variants(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "x", "variants": "4", "seed": "123"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["variants_requested"] == 4
    assert len(body["variants"]) == 4
    # First variant honors the supplied seed; others are randomized.
    assert body["variants"][0]["seed"] == 123
    other_seeds = {v["seed"] for v in body["variants"][1:]}
    assert 123 not in other_seeds


def test_create_task_clamps_variants(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "x", "variants": "99"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.json()["variants_requested"] == 8  # clamped to max


def test_get_task_after_creation(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "x"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    tid = r.json()["id"]
    r2 = client.get(f"/api/tasks/{tid}")
    assert r2.status_code == 200
    assert r2.json()["id"] == tid


def test_get_unknown_task_404(client):
    r = client.get("/api/tasks/nonexistent")
    assert r.status_code == 404


def test_list_tasks_recent_first(client):
    ids = []
    for i in range(3):
        r = client.post(
            "/api/tasks",
            data={"mode": "kontext", "prompt": f"prompt {i}"},
            files={"image": ("in.png", _png_bytes(), "image/png")},
        )
        ids.append(r.json()["id"])

    r = client.get("/api/tasks?limit=5")
    body = r.json()
    returned_ids = [t["id"] for t in body["tasks"]]
    assert returned_ids == list(reversed(ids))


def test_inpaint_requires_mask(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "inpaint", "prompt": "x"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400


def test_inpaint_accepts_mask(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "inpaint", "prompt": "x"},
        files={
            "image": ("in.png", _png_bytes(), "image/png"),
            "mask": ("m.png", _png_bytes(color=(255, 255, 255)), "image/png"),
        },
    )
    assert r.status_code == 202


def test_unknown_mode_rejected(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "doodle", "prompt": "x"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400


def test_approve_done_variant(client, app_env):
    _srv, d, s = app_env
    # Create a task, then simulate the worker finishing one of its variants.
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "x", "variants": "2"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    tid = r.json()["id"]

    with d.get_session() as session:
        t = session.get(d.Task, tid)
        variants = list(t.variants)
        # Pretend the worker finished both.
        for v in variants:
            v.status = "done"
            v.output_path = str(s.save_variant_output(tid, v.id, b"fake-png-bytes"))
        t.status = "done"
        session.commit()
        chosen = variants[1].id

    r2 = client.post(f"/api/tasks/{tid}/approve", json={"variant_id": chosen})
    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "approved"
    approved_count = sum(1 for v in body["variants"] if v["approved"])
    assert approved_count == 1
    assert next(v["id"] for v in body["variants"] if v["approved"]) == chosen


def test_approve_not_done_409(client, app_env):
    _srv, d, _s = app_env
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "x"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    tid = r.json()["id"]
    with d.get_session() as session:
        v_id = session.get(d.Task, tid).variants[0].id

    r2 = client.post(f"/api/tasks/{tid}/approve", json={"variant_id": v_id})
    assert r2.status_code == 409


def test_variant_output_404_when_no_file_yet(client, app_env):
    _srv, d, _s = app_env
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "x"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    tid = r.json()["id"]
    with d.get_session() as session:
        v_id = session.get(d.Task, tid).variants[0].id

    r2 = client.get(f"/api/variants/{v_id}/output")
    assert r2.status_code == 404


def test_variant_output_returns_png_when_done(client, app_env):
    _srv, d, s = app_env
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "x"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    tid = r.json()["id"]
    with d.get_session() as session:
        v = session.get(d.Task, tid).variants[0]
        v.status = "done"
        png = _png_bytes()
        v.output_path = str(s.save_variant_output(tid, v.id, png))
        session.commit()
        vid = v.id

    r2 = client.get(f"/api/variants/{vid}/output")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/png"
    assert r2.content == png


def test_delete_task_removes_db_and_storage(client, app_env, tmp_path):
    _srv, d, _s = app_env
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "x"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    tid = r.json()["id"]
    assert (tmp_path / "tasks" / tid).exists()

    r2 = client.delete(f"/api/tasks/{tid}")
    assert r2.status_code == 200

    with d.get_session() as session:
        assert session.get(d.Task, tid) is None
    assert not (tmp_path / "tasks" / tid).exists()
