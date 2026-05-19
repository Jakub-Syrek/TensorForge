"""Tests for backend.progress — JobProgress lifecycle + nvidia-smi parser."""

from __future__ import annotations

from backend.progress import JobProgress, parse_nvsmi_line


def test_job_progress_start_resets_state():
    j = JobProgress()
    j.last_error = "old"
    j.start(total=28, mode="kontext")
    assert j.active is True
    assert j.step == 0
    assert j.total == 28
    assert j.mode == "kontext"
    assert j.last_error is None
    assert j.started_at is not None
    assert j.finished_at is None


def test_job_progress_advance_is_one_indexed_for_display():
    j = JobProgress()
    j.start(total=10, mode="kontext")
    j.advance(0)  # diffusers' first step-end fires with step=0
    assert j.step == 1
    j.advance(9)
    assert j.step == 10


def test_job_progress_finish_clean_snaps_to_total():
    j = JobProgress()
    j.start(total=10, mode="kontext")
    j.advance(4)
    j.finish()
    assert j.active is False
    assert j.step == 10  # snapped to 100%
    assert j.last_error is None


def test_job_progress_finish_with_error_keeps_step():
    j = JobProgress()
    j.start(total=10, mode="kontext")
    j.advance(2)
    j.finish(error="boom")
    assert j.active is False
    assert j.step == 3  # partial, NOT snapped
    assert j.last_error == "boom"


def test_job_progress_snapshot_shape():
    j = JobProgress()
    j.start(total=4, mode="inpaint")
    j.advance(0)
    snap = j.snapshot()
    assert snap["active"] is True
    assert snap["step"] == 1
    assert snap["total"] == 4
    assert snap["mode"] == "inpaint"
    assert snap["percent"] == 25.0
    assert snap["error"] is None
    assert snap["elapsed_s"] is not None


def test_job_progress_idle_snapshot_has_zero_percent():
    j = JobProgress()
    snap = j.snapshot()
    assert snap["active"] is False
    assert snap["percent"] == 0.0


def test_parse_nvsmi_line_happy_path():
    line = "NVIDIA GeForce RTX 5080, 87, 12450, 16303, 64, 245.30"
    s = parse_nvsmi_line(line)
    assert s is not None
    assert s.name == "NVIDIA GeForce RTX 5080"
    assert s.util_percent == 87
    assert s.vram_used_mb == 12450
    assert s.vram_total_mb == 16303
    assert s.temperature_c == 64
    assert s.power_w == 245.30


def test_parse_nvsmi_line_handles_na_columns():
    line = "Tesla T4, 5, 100, 16000, [N/A], [N/A]"
    s = parse_nvsmi_line(line)
    assert s is not None
    assert s.temperature_c is None
    assert s.power_w is None


def test_parse_nvsmi_line_returns_none_on_garbage():
    assert parse_nvsmi_line("") is None
    assert parse_nvsmi_line("not, enough") is None
    assert parse_nvsmi_line("name, not-an-int, 1, 2") is None


def test_request_abort_on_active_job_returns_true_and_sets_flag():
    j = JobProgress()
    j.start(total=28, mode="kontext")
    assert j.aborted is False
    assert j.request_abort() is True
    assert j.aborted is True


def test_request_abort_on_idle_job_returns_false_and_no_op():
    j = JobProgress()
    assert j.request_abort() is False
    assert j.aborted is False


def test_start_resets_aborted_flag():
    j = JobProgress()
    j.start(total=10, mode="kontext")
    j.request_abort()
    assert j.aborted is True
    j.start(total=10, mode="inpaint")  # new job
    assert j.aborted is False


def test_snapshot_includes_aborted_field():
    j = JobProgress()
    j.start(total=10, mode="kontext")
    snap = j.snapshot()
    assert "aborted" in snap
    assert snap["aborted"] is False
    j.request_abort()
    assert j.snapshot()["aborted"] is True


def test_gpu_stats_to_dict_computes_vram_percent():
    from backend.progress import GpuStats

    s = GpuStats(util_percent=50, vram_used_mb=8000, vram_total_mb=16000)
    d = s.to_dict()
    assert d["vram_percent"] == 50.0
    assert d["util_percent"] == 50
    assert d["temperature_c"] is None
