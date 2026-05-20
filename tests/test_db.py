"""Tests for backend.db — SQLAlchemy models + session factory.

Uses a temp SQLite file per test so we never touch the real data/aipic.db.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPIC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AIPIC_DB_PATH", str(tmp_path / "test.db"))
    import backend.storage as s

    importlib.reload(s)
    import backend.db as d

    importlib.reload(d)
    d.init_db()
    yield d


def test_init_db_creates_tables(db, tmp_path):
    assert (tmp_path / "test.db").exists()


def test_task_with_default_status_queued(db):
    with db.get_session() as s:
        t = db.Task(
            mode="kontext",
            prompt="x",
            input_path="/tmp/in.png",
            original_width=100,
            original_height=200,
            params={},
            variants_requested=1,
        )
        s.add(t)
        s.commit()
        s.refresh(t)
        assert t.status == "queued"
        assert t.id and len(t.id) == 32  # hex uuid
        assert t.created_at is not None


def test_task_variant_relationship(db):
    with db.get_session() as s:
        t = db.Task(
            mode="kontext",
            prompt="x",
            input_path="/tmp/in.png",
            original_width=100,
            original_height=200,
            variants_requested=2,
        )
        s.add(t)
        s.flush()
        v1 = db.Variant(task_id=t.id, seed=111)
        v2 = db.Variant(task_id=t.id, seed=222)
        s.add_all([v1, v2])
        s.commit()

        s.refresh(t)
        assert len(t.variants) == 2
        seeds = sorted(v.seed for v in t.variants)
        assert seeds == [111, 222]


def test_variant_cascade_delete(db):
    with db.get_session() as s:
        t = db.Task(
            mode="kontext",
            prompt="x",
            input_path="/tmp/in.png",
            original_width=100,
            original_height=200,
            variants_requested=1,
        )
        s.add(t)
        s.flush()
        v = db.Variant(task_id=t.id, seed=1)
        s.add(v)
        s.commit()
        tid, vid = t.id, v.id

    with db.get_session() as s:
        # Deleting the task should cascade to the variant.
        s.delete(s.get(db.Task, tid))
        s.commit()

    with db.get_session() as s:
        assert s.get(db.Task, tid) is None
        assert s.get(db.Variant, vid) is None


def test_variant_default_not_approved(db):
    with db.get_session() as s:
        t = db.Task(
            mode="kontext",
            prompt="x",
            input_path="/tmp/in.png",
            original_width=100,
            original_height=200,
        )
        s.add(t)
        s.flush()
        v = db.Variant(task_id=t.id, seed=42)
        s.add(v)
        s.commit()
        s.refresh(v)
        assert v.approved is False
        assert v.status == "queued"
