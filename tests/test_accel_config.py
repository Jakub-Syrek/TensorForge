"""Tests for the real backend.pipeline.AccelConfig — no duplicated logic.

backend.pipeline imports torch at module load. In CI we don't install torch
(would add ~2 minutes for nothing useful), so we stub the module with a
minimal fake before importing. The stub only needs the attributes that
pipeline.py touches at import time and inside AccelConfig methods.
"""
from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(scope="module")
def AccelConfig():
    if "torch" not in sys.modules:
        fake = types.ModuleType("torch")
        fake.bfloat16 = object()  # sentinel — pipeline only reads this at __init__
        fake.float32 = object()

        class _Cuda:
            @staticmethod
            def is_available():
                return False
        fake.cuda = _Cuda
        sys.modules["torch"] = fake

    from backend.pipeline import AccelConfig as _AccelConfig
    return _AccelConfig


def _clear_accel_env(monkeypatch):
    for var in ("FLUX_ACCEL_REPO", "FLUX_ACCEL_WEIGHT", "FLUX_ACCEL_SCALE"):
        monkeypatch.delenv(var, raising=False)


def test_from_env_returns_none_when_unset(AccelConfig, monkeypatch):
    _clear_accel_env(monkeypatch)
    assert AccelConfig.from_env() is None


def test_from_env_returns_none_when_only_repo_set(AccelConfig, monkeypatch):
    _clear_accel_env(monkeypatch)
    monkeypatch.setenv("FLUX_ACCEL_REPO", "ByteDance/Hyper-SD")
    assert AccelConfig.from_env() is None


def test_from_env_defaults_scale_to_one(AccelConfig, monkeypatch):
    _clear_accel_env(monkeypatch)
    monkeypatch.setenv("FLUX_ACCEL_REPO", "ByteDance/Hyper-SD")
    monkeypatch.setenv("FLUX_ACCEL_WEIGHT", "Hyper-FLUX.1-dev-8steps-lora.safetensors")
    cfg = AccelConfig.from_env()
    assert cfg is not None
    assert cfg.repo == "ByteDance/Hyper-SD"
    assert cfg.weight_name == "Hyper-FLUX.1-dev-8steps-lora.safetensors"
    assert cfg.scale == 1.0


def test_from_env_honours_explicit_scale(AccelConfig, monkeypatch):
    _clear_accel_env(monkeypatch)
    monkeypatch.setenv("FLUX_ACCEL_REPO", "ByteDance/Hyper-SD")
    monkeypatch.setenv("FLUX_ACCEL_WEIGHT", "Hyper-FLUX.1-dev-8steps-lora.safetensors")
    monkeypatch.setenv("FLUX_ACCEL_SCALE", "0.125")
    cfg = AccelConfig.from_env()
    assert cfg is not None
    assert cfg.scale == pytest.approx(0.125)
