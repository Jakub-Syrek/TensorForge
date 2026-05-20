"""Tests for backend.storage — local filesystem layout for tasks/variants."""

from __future__ import annotations

import os

import pytest

# Force a temp DATA_DIR before importing backend.storage so the module-level
# constant resolves there, not into the real ./data/ tree.


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPIC_DATA_DIR", str(tmp_path))
    # Reimport the module so DATA_DIR picks up the env override.
    import importlib

    import backend.storage as s

    importlib.reload(s)
    s.init()
    yield s


def test_init_creates_data_and_tasks_dirs(storage, tmp_path):
    assert (tmp_path / "tasks").is_dir()


def test_task_dir_created_on_demand(storage, tmp_path):
    p = storage.task_dir("abc123")
    assert p == tmp_path / "tasks" / "abc123"
    assert p.is_dir()


def test_save_input_writes_bytes(storage, tmp_path):
    path = storage.save_input("t1", b"hello", ext=".png")
    assert path.exists()
    assert path.read_bytes() == b"hello"
    assert path.name == "input.png"


def test_save_mask_overwrites_on_repeat(storage):
    storage.save_mask("t1", b"first")
    p = storage.save_mask("t1", b"second")
    assert p.read_bytes() == b"second"


def test_save_variant_output_keyed_by_variant_id(storage):
    p1 = storage.save_variant_output("t1", "v1", b"AAA")
    p2 = storage.save_variant_output("t1", "v2", b"BBB")
    assert p1 != p2
    assert p1.read_bytes() == b"AAA"
    assert p2.read_bytes() == b"BBB"


def test_delete_task_dir_idempotent(storage):
    storage.save_input("t1", b"x")
    storage.delete_task_dir("t1")
    # Second delete is a no-op, not an error.
    storage.delete_task_dir("t1")
    storage.delete_task_dir("does-not-exist")


def test_delete_task_dir_removes_variants_too(storage, tmp_path):
    storage.save_input("t1", b"x")
    storage.save_variant_output("t1", "v1", b"y")
    storage.delete_task_dir("t1")
    assert not (tmp_path / "tasks" / "t1").exists()


def test_default_data_dir_is_relative_to_cwd(monkeypatch, tmp_path):
    """When the env var is unset, DATA_DIR points to ./data — verified by
    creating into a known cwd and observing the resolved path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIPIC_DATA_DIR", raising=False)
    import importlib

    import backend.storage as s

    importlib.reload(s)
    assert (tmp_path / "data").resolve() == s.DATA_DIR
    # Cleanup so the next test's fixture starts clean.
    os.environ.pop("AIPIC_DATA_DIR", None)
