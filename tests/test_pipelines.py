"""Tests for the /api/pipelines surface + worker pipeline advancement."""

from __future__ import annotations

import importlib
import json
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
    import backend.worker as w

    importlib.reload(w)
    import backend.server as srv

    importlib.reload(srv)
    return srv, d, s, w


@pytest.fixture
def client(app_env):
    srv, _d, _s, _w = app_env
    return TestClient(srv.app)


def _png_bytes(size=(32, 32), color=(50, 50, 50)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_create_pipeline_with_steps_creates_tasks_in_order(client, app_env):
    _srv, d, _s, _w = app_env
    steps = [
        {"mode": "auto", "prompt": "remove the hat"},
        {"mode": "kontext", "prompt": "change shirt to blue"},
        {"mode": "kontext", "prompt": "make the background twilight"},
    ]
    r = client.post(
        "/api/pipelines",
        data={"steps_json": json.dumps(steps)},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    pid = body["pipeline_id"]
    assert len(body["steps"]) == 3

    with d.get_session() as s:
        rows = (
            s.query(d.Task).filter(d.Task.pipeline_id == pid).order_by(d.Task.pipeline_step).all()
        )
        assert [t.pipeline_step for t in rows] == [0, 1, 2]
        # First step is queued (has input), others blocked until previous done.
        assert rows[0].status == "queued"
        assert rows[1].status == "blocked"
        assert rows[2].status == "blocked"
        # Step 0 has the upload as input; later steps don't (parent_variant_id wired by worker).
        assert rows[0].input_path is not None
        assert rows[1].input_path is None
        assert rows[2].input_path is None
        # Modes: 'remove the hat' -> kontext via auto; explicit kontext rest.
        assert rows[0].mode == "kontext"
        assert rows[1].mode == "kontext"
        assert rows[2].mode == "kontext"


def test_create_pipeline_without_image_works_when_step0_is_generate(client, app_env):
    _srv, d, _s, _w = app_env
    steps = [
        {"mode": "generate", "prompt": "a sunset over mountains"},
        {"mode": "kontext", "prompt": "add a small cabin"},
    ]
    r = client.post(
        "/api/pipelines",
        data={"steps_json": json.dumps(steps)},
    )
    assert r.status_code == 202

    with d.get_session() as s:
        rows = (
            s.query(d.Task)
            .filter(d.Task.pipeline_id == r.json()["pipeline_id"])
            .order_by(d.Task.pipeline_step)
            .all()
        )
        assert rows[0].mode == "generate"
        assert rows[0].input_path is None
        assert rows[0].status == "queued"


def test_create_pipeline_rejects_empty_steps(client):
    r = client.post("/api/pipelines", data={"steps_json": json.dumps([])})
    assert r.status_code == 400


def test_create_pipeline_rejects_too_many_steps(client):
    r = client.post(
        "/api/pipelines",
        data={"steps_json": json.dumps([{"mode": "auto", "prompt": "x"}] * 11)},
    )
    assert r.status_code == 400


def test_create_pipeline_rejects_invalid_json(client):
    r = client.post("/api/pipelines", data={"steps_json": "not-json"})
    assert r.status_code == 400


def test_create_pipeline_rejects_step_with_empty_prompt(client):
    r = client.post(
        "/api/pipelines",
        data={"steps_json": json.dumps([{"mode": "auto", "prompt": "   "}])},
    )
    assert r.status_code == 400


def test_create_pipeline_rejects_step_with_unknown_mode(client):
    r = client.post(
        "/api/pipelines",
        data={"steps_json": json.dumps([{"mode": "doodle", "prompt": "x"}])},
    )
    assert r.status_code == 400


def test_get_pipeline_returns_aggregated_view(client, app_env):
    _srv, _d, _s, _w = app_env
    r = client.post(
        "/api/pipelines",
        data={
            "steps_json": json.dumps(
                [{"mode": "kontext", "prompt": "x"}, {"mode": "kontext", "prompt": "y"}]
            )
        },
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    pid = r.json()["pipeline_id"]
    r2 = client.get(f"/api/pipelines/{pid}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["pipeline_id"] == pid
    assert body["status"] == "running"  # step 0 queued, step 1 blocked
    assert len(body["steps"]) == 2


def test_get_unknown_pipeline_404(client):
    r = client.get("/api/pipelines/no-such-id")
    assert r.status_code == 404


def test_worker_advance_pipeline_unblocks_next_step(app_env):
    _srv, d, s, w = app_env

    # Create a 2-step pipeline manually (no live worker, no GPU).
    with d.get_session() as session:
        t0 = d.Task(
            mode="kontext",
            prompt="step 0",
            input_path="/tmp/in.png",
            original_width=100,
            original_height=200,
            pipeline_id="pid",
            pipeline_step=0,
            status="queued",
        )
        t1 = d.Task(
            mode="kontext",
            prompt="step 1",
            pipeline_id="pid",
            pipeline_step=1,
            status="blocked",
        )
        session.add_all([t0, t1])
        session.flush()
        v0 = d.Variant(task_id=t0.id, seed=1, status="queued")
        v1 = d.Variant(task_id=t1.id, seed=2, status="blocked")
        session.add_all([v0, v1])
        session.commit()
        t0_id, t1_id, v0_id, v1_id = t0.id, t1.id, v0.id, v1.id

    # Simulate the worker finishing step 0.
    with d.get_session() as session:
        v = session.get(d.Variant, v0_id)
        v.status = "done"
        v.output_path = str(s.save_variant_output(t0_id, v0_id, b"fake-png"))
        session.commit()

    worker = w.TaskWorker(editor=None)
    with d.get_session() as session:
        t = session.get(d.Task, t0_id)
        worker._maybe_finalize_task(session, t)
        session.commit()

    with d.get_session() as session:
        t1 = session.get(d.Task, t1_id)
        v1 = session.get(d.Variant, v1_id)
        assert t1.status == "queued"
        assert t1.parent_variant_id == v0_id
        assert v1.status == "queued"


def test_worker_propagates_failure_downstream(app_env):
    _srv, d, _s, w = app_env

    with d.get_session() as session:
        t0 = d.Task(
            mode="kontext",
            prompt="step 0",
            input_path="/tmp/in.png",
            original_width=100,
            original_height=200,
            pipeline_id="pid",
            pipeline_step=0,
            status="running",
        )
        t1 = d.Task(
            mode="kontext",
            prompt="step 1",
            pipeline_id="pid",
            pipeline_step=1,
            status="blocked",
        )
        t2 = d.Task(
            mode="kontext",
            prompt="step 2",
            pipeline_id="pid",
            pipeline_step=2,
            status="blocked",
        )
        session.add_all([t0, t1, t2])
        session.flush()
        session.add_all(
            [
                d.Variant(task_id=t0.id, seed=1, status="failed"),
                d.Variant(task_id=t1.id, seed=2, status="blocked"),
                d.Variant(task_id=t2.id, seed=3, status="blocked"),
            ]
        )
        session.commit()
        t0_id, t1_id, t2_id = t0.id, t1.id, t2.id

    worker = w.TaskWorker(editor=None)
    with d.get_session() as session:
        t = session.get(d.Task, t0_id)
        worker._maybe_finalize_task(session, t)
        session.commit()

    with d.get_session() as session:
        assert session.get(d.Task, t0_id).status == "failed"
        assert session.get(d.Task, t1_id).status == "aborted"
        assert session.get(d.Task, t2_id).status == "aborted"
