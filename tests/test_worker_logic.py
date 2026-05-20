"""Tests for the non-GPU parts of backend.queue.TaskWorker.

The GPU path (_render and the surrounding thread loop) is marked
'pragma: no cover'. What remains testable: the DB-transaction logic
(claim/finalize), which is critical for correctness when multiple
variants share a task.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPIC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AIPIC_DB_PATH", str(tmp_path / "test.db"))
    import backend.storage as s

    importlib.reload(s)
    import backend.db as d

    importlib.reload(d)
    d.init_db()
    import backend.worker as q

    importlib.reload(q)
    return q, d


def _make_task(d, *, variants_requested=1, seeds=(1,)):
    with d.get_session() as s:
        t = d.Task(
            mode="kontext",
            prompt="x",
            input_path="/tmp/in.png",
            original_width=100,
            original_height=200,
            params={},
            variants_requested=variants_requested,
        )
        s.add(t)
        s.flush()
        for seed in seeds:
            s.add(d.Variant(task_id=t.id, seed=seed))
        s.commit()
        return t.id


def test_claim_returns_none_when_no_queued_variants(env):
    q, _d = env
    worker = q.TaskWorker(editor=None)
    assert worker._claim_next_variant() is None


def test_claim_picks_oldest_first(env):
    q, d = env
    t1 = _make_task(d, seeds=(1,))
    t2 = _make_task(d, seeds=(2,))
    worker = q.TaskWorker(editor=None)
    v_id = worker._claim_next_variant()
    assert v_id is not None
    with d.get_session() as s:
        v = s.get(d.Variant, v_id)
        # Older task's variant comes first.
        assert v.task_id == t1
        # Status flipped to running.
        assert v.status == "running"
        # Parent task also flipped.
        assert s.get(d.Task, t1).status == "running"
        assert s.get(d.Task, t1).started_at is not None
        # The other task untouched.
        assert s.get(d.Task, t2).status == "queued"


def test_finalize_when_all_done_sets_task_done(env):
    q, d = env
    tid = _make_task(d, variants_requested=2, seeds=(1, 2))
    with d.get_session() as s:
        for v in s.query(d.Variant).filter(d.Variant.task_id == tid).all():
            v.status = "done"
        s.commit()

    worker = q.TaskWorker(editor=None)
    with d.get_session() as s:
        t = s.get(d.Task, tid)
        worker._maybe_finalize_task(s, t)
        s.commit()

    with d.get_session() as s:
        t = s.get(d.Task, tid)
        assert t.status == "done"
        assert t.finished_at is not None


def test_finalize_leaves_task_alone_while_variants_still_queued(env):
    q, d = env
    tid = _make_task(d, variants_requested=2, seeds=(1, 2))
    with d.get_session() as s:
        variants = s.query(d.Variant).filter(d.Variant.task_id == tid).all()
        variants[0].status = "done"
        s.commit()

    worker = q.TaskWorker(editor=None)
    with d.get_session() as s:
        t = s.get(d.Task, tid)
        worker._maybe_finalize_task(s, t)
        s.commit()

    with d.get_session() as s:
        t = s.get(d.Task, tid)
        # The other variant is still 'queued', so task stays as-is.
        assert t.status == "queued"
        assert t.finished_at is None


def test_finalize_partial_success_still_counts_as_done(env):
    """If some variants succeed and others fail, the user still got
    something useful — mark the task 'done' so the UI shows the
    successful variant(s) for approval."""
    q, d = env
    tid = _make_task(d, variants_requested=2, seeds=(1, 2))
    with d.get_session() as s:
        variants = s.query(d.Variant).filter(d.Variant.task_id == tid).all()
        variants[0].status = "done"
        variants[1].status = "failed"
        s.commit()

    worker = q.TaskWorker(editor=None)
    with d.get_session() as s:
        t = s.get(d.Task, tid)
        worker._maybe_finalize_task(s, t)
        s.commit()

    with d.get_session() as s:
        assert s.get(d.Task, tid).status == "done"


def test_finalize_all_failed_sets_task_failed(env):
    q, d = env
    tid = _make_task(d, variants_requested=2, seeds=(1, 2))
    with d.get_session() as s:
        for v in s.query(d.Variant).filter(d.Variant.task_id == tid).all():
            v.status = "failed"
        s.commit()

    worker = q.TaskWorker(editor=None)
    with d.get_session() as s:
        t = s.get(d.Task, tid)
        worker._maybe_finalize_task(s, t)
        s.commit()

    with d.get_session() as s:
        assert s.get(d.Task, tid).status == "failed"


def test_finalize_aborted_when_no_done_only_aborts(env):
    q, d = env
    tid = _make_task(d, variants_requested=2, seeds=(1, 2))
    with d.get_session() as s:
        for v in s.query(d.Variant).filter(d.Variant.task_id == tid).all():
            v.status = "aborted"
        s.commit()

    worker = q.TaskWorker(editor=None)
    with d.get_session() as s:
        t = s.get(d.Task, tid)
        worker._maybe_finalize_task(s, t)
        s.commit()

    with d.get_session() as s:
        assert s.get(d.Task, tid).status == "aborted"
