"""Tests for backend.pipeline.AccelConfig — the real class, no duplicate.

torch is stubbed in tests/conftest.py before any backend import.
"""

from __future__ import annotations

import pytest

from backend.pipeline import AccelConfig


def _clear_accel_env(monkeypatch):
    for var in ("FLUX_ACCEL_REPO", "FLUX_ACCEL_WEIGHT", "FLUX_ACCEL_SCALE"):
        monkeypatch.delenv(var, raising=False)


def test_from_env_returns_none_when_unset(monkeypatch):
    _clear_accel_env(monkeypatch)
    assert AccelConfig.from_env() is None


def test_from_env_returns_none_when_only_repo_set(monkeypatch):
    _clear_accel_env(monkeypatch)
    monkeypatch.setenv("FLUX_ACCEL_REPO", "ByteDance/Hyper-SD")
    assert AccelConfig.from_env() is None


def test_from_env_defaults_scale_to_one(monkeypatch):
    _clear_accel_env(monkeypatch)
    monkeypatch.setenv("FLUX_ACCEL_REPO", "ByteDance/Hyper-SD")
    monkeypatch.setenv("FLUX_ACCEL_WEIGHT", "Hyper-FLUX.1-dev-8steps-lora.safetensors")
    cfg = AccelConfig.from_env()
    assert cfg is not None
    assert cfg.repo == "ByteDance/Hyper-SD"
    assert cfg.weight_name == "Hyper-FLUX.1-dev-8steps-lora.safetensors"
    assert cfg.scale == 1.0


def test_from_env_honours_explicit_scale(monkeypatch):
    _clear_accel_env(monkeypatch)
    monkeypatch.setenv("FLUX_ACCEL_REPO", "ByteDance/Hyper-SD")
    monkeypatch.setenv("FLUX_ACCEL_WEIGHT", "Hyper-FLUX.1-dev-8steps-lora.safetensors")
    monkeypatch.setenv("FLUX_ACCEL_SCALE", "0.125")
    cfg = AccelConfig.from_env()
    assert cfg is not None
    assert cfg.scale == pytest.approx(0.125)
