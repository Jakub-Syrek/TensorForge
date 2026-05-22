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
    import backend.worker as q

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


def test_generate_mode_works_without_image(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "generate", "prompt": "a cat on a chair", "steps": "4"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["mode"] == "generate"
    # original_size is null for generate (no input image dimensions to preserve).
    assert body["original_size"] == [None, None]


def test_generate_mode_can_still_attach_image_but_ignores_it(client):
    """Generate accepts image upload (we don't error) but doesn't condition
    on it — schnell is pure text-to-image."""
    r = client.post(
        "/api/tasks",
        data={"mode": "generate", "prompt": "a cat", "steps": "4"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 202
    assert r.json()["mode"] == "generate"


def test_auto_mode_resolves_to_kontext_for_edit_prompt(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "auto", "prompt": "remove the hat"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["mode"] == "kontext"
    assert body["params"]["routing"]["intent"] == "edit"


def test_auto_mode_resolves_to_generate_for_creation_prompt(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "auto", "prompt": "a cat sitting on a chair"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["mode"] == "generate"
    assert body["params"]["routing"]["intent"] == "generate"


def test_auto_no_image_goes_generate(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "auto", "prompt": "anything"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["mode"] == "generate"


def test_non_generate_mode_without_image_400(client):
    r = client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "remove the hat"},
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


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------


def test_video_t2v_accepted_without_image(client):
    """mode=video, subtype=t2v: pure text-to-video, no input image needed."""
    r = client.post(
        "/api/tasks",
        data={
            "mode": "video",
            "prompt": "a cat riding a skateboard",
            "video_backend": "ltx",
            "video_subtype": "t2v",
            "num_frames": "49",
            "fps": "24",
        },
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["mode"] == "video"
    assert body["params"]["video_backend"] == "ltx"
    assert body["params"]["video_subtype"] == "t2v"
    assert body["params"]["num_frames"] == 49
    assert body["params"]["fps"] == 24


def test_video_i2v_rejected_without_image(client):
    """mode=video, subtype=i2v: must reject when no reference image attached."""
    r = client.post(
        "/api/tasks",
        data={
            "mode": "video",
            "prompt": "the figure turns its head",
            "video_backend": "wan",
            "video_subtype": "i2v",
        },
    )
    assert r.status_code == 400


def test_video_i2v_accepted_with_image(client):
    r = client.post(
        "/api/tasks",
        data={
            "mode": "video",
            "prompt": "the figure turns its head",
            "video_backend": "wan",
            "video_subtype": "i2v",
        },
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["mode"] == "video"
    assert body["params"]["video_backend"] == "wan"
    assert body["params"]["video_subtype"] == "i2v"


def test_video_unknown_backend_rejected(client):
    r = client.post(
        "/api/tasks",
        data={
            "mode": "video",
            "prompt": "x",
            "video_backend": "sora",
            "video_subtype": "t2v",
        },
    )
    assert r.status_code == 400


def test_video_unknown_subtype_rejected(client):
    r = client.post(
        "/api/tasks",
        data={
            "mode": "video",
            "prompt": "x",
            "video_backend": "ltx",
            "video_subtype": "v2v",
        },
    )
    assert r.status_code == 400


def test_recent_prompts_empty_initially(client):
    r = client.get("/api/prompts/recent")
    assert r.status_code == 200
    assert r.json() == {"prompts": []}


def test_recent_prompts_returns_submitted_prompts(client):
    client.post(
        "/api/tasks",
        data={"mode": "generate", "prompt": "a cat", "steps": "4"},
    )
    client.post(
        "/api/tasks",
        data={"mode": "kontext", "prompt": "remove the hat"},
        files={"image": ("in.png", _png_bytes(), "image/png")},
    )
    r = client.get("/api/prompts/recent")
    assert r.status_code == 200
    body = r.json()
    prompts = [p["prompt"] for p in body["prompts"]]
    # Newest first.
    assert prompts == ["remove the hat", "a cat"]
    # Mode + timestamp surfaced.
    assert body["prompts"][0]["mode"] == "kontext"
    assert body["prompts"][1]["mode"] == "generate"
    assert body["prompts"][0]["created_at"] is not None


def test_recent_prompts_dedupes(client):
    for _ in range(3):
        client.post(
            "/api/tasks",
            data={"mode": "generate", "prompt": "same prompt"},
        )
    client.post(
        "/api/tasks",
        data={"mode": "generate", "prompt": "different prompt"},
    )
    r = client.get("/api/prompts/recent")
    body = r.json()
    prompts = [p["prompt"] for p in body["prompts"]]
    # "same prompt" appears once despite three submissions.
    assert prompts == ["different prompt", "same prompt"]


def test_recent_prompts_respects_limit(client):
    for i in range(15):
        client.post(
            "/api/tasks",
            data={"mode": "generate", "prompt": f"prompt #{i}"},
        )
    r = client.get("/api/prompts/recent?limit=5")
    body = r.json()
    assert len(body["prompts"]) == 5
    # Bulk-inserted prompts can share a created_at second (SQLite stores
    # to second resolution and these inserts run in <1 ms), so order
    # within the batch isn't deterministic. We only assert that the
    # returned 5 are from the *latest* portion of the batch — i.e. all
    # indices in [10..14].
    returned = {p["prompt"] for p in body["prompts"]}
    latest = {f"prompt #{i}" for i in range(10, 15)}
    assert returned == latest


def test_recent_prompts_limit_clamped(client):
    """Out-of-range limit values clamp to [1, 100] instead of erroring."""
    client.post("/api/tasks", data={"mode": "generate", "prompt": "x"})
    r1 = client.get("/api/prompts/recent?limit=-5")
    r2 = client.get("/api/prompts/recent?limit=9999")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert len(r1.json()["prompts"]) <= 1
    assert len(r2.json()["prompts"]) <= 100


def test_variant_output_returns_video_mp4_for_mp4_path(client, app_env):
    """Confirm the variant output endpoint serves .mp4 with video/mp4
    media type, not image/png."""
    _srv, d, s = app_env
    r = client.post(
        "/api/tasks",
        data={
            "mode": "video",
            "prompt": "x",
            "video_backend": "ltx",
            "video_subtype": "t2v",
        },
    )
    tid = r.json()["id"]
    with d.get_session() as session:
        v = session.get(d.Task, tid).variants[0]
        v.status = "done"
        # Pretend the worker produced an MP4 — payload is irrelevant for the
        # routing test; only the suffix matters for media-type resolution.
        fake_mp4 = b"\x00\x00\x00\x18ftypmp42fake-mp4-payload"
        v.output_path = str(s.save_variant_output(tid, v.id, fake_mp4, ext=".mp4"))
        session.commit()
        vid = v.id

    r2 = client.get(f"/api/variants/{vid}/output")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "video/mp4"
    assert r2.content == fake_mp4
