"""Tests for FLUX_QUANT env var parsing in backend.pipeline."""

from __future__ import annotations

import pytest

from backend.pipeline import _read_quant_mode


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("FLUX_QUANT", raising=False)


def test_unset_returns_none():
    assert _read_quant_mode() is None


def test_empty_string_returns_none(monkeypatch):
    monkeypatch.setenv("FLUX_QUANT", "")
    assert _read_quant_mode() is None


def test_whitespace_returns_none(monkeypatch):
    monkeypatch.setenv("FLUX_QUANT", "   ")
    assert _read_quant_mode() is None


def test_4bit_lowercase(monkeypatch):
    monkeypatch.setenv("FLUX_QUANT", "4bit")
    assert _read_quant_mode() == "4bit"


def test_4bit_uppercase_normalized(monkeypatch):
    monkeypatch.setenv("FLUX_QUANT", "4BIT")
    assert _read_quant_mode() == "4bit"


def test_4bit_with_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("FLUX_QUANT", "  4bit  ")
    assert _read_quant_mode() == "4bit"


def test_unknown_mode_returns_none(monkeypatch):
    """Unrecognized values fall through to None — fail-safe, not fail-loud,
    because the alternative (raise on import) would brick the server on a typo."""
    monkeypatch.setenv("FLUX_QUANT", "8bit")
    assert _read_quant_mode() is None

    monkeypatch.setenv("FLUX_QUANT", "yes")
    assert _read_quant_mode() is None

    monkeypatch.setenv("FLUX_QUANT", "true")
    assert _read_quant_mode() is None
