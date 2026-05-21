"""Test-wide setup.

backend.pipeline / backend.server import torch at module load. CI doesn't
install torch (would add ~2 minutes for nothing useful in routing tests),
so we install a minimal stub into sys.modules before any test module
imports backend.*. The stub only carries the attributes our code touches.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    fake = types.ModuleType("torch")
    # transformers imports trigger ``importlib.util.find_spec('torch')`` to
    # decide whether to wire up its torch-backed paths. A ModuleType with
    # __spec__ unset (default None) makes find_spec raise ValueError, which
    # aborts ``from transformers import X`` at module import time. Give the
    # stub a real ModuleSpec so the probe succeeds; transformers then
    # discovers there's no actual functionality and stays in the no-torch
    # code path, which is what we want for these routing tests.
    fake.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    fake.__version__ = "stub"
    fake.bfloat16 = object()
    fake.float32 = object()
    fake.dtype = type
    fake.Tensor = type("Tensor", (), {})

    def _inference_mode(*args, **kwargs):
        """Stub matching ``torch.inference_mode()`` used as a decorator."""

        def _decorate(fn):
            return fn

        return _decorate

    fake.inference_mode = _inference_mode
    fake.no_grad = _inference_mode
    fake.load = lambda *a, **k: {}  # pragma: no cover — stub never exercised

    nn = types.ModuleType("torch.nn")

    class _StubModule:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):  # pragma: no cover
            raise RuntimeError("torch.nn stub: forward called in tests")

        def to(self, *args, **kwargs):
            return self

    nn.Module = _StubModule
    nn.Linear = _StubModule
    nn.GELU = _StubModule
    nn.LayerNorm = _StubModule
    nn.Sequential = _StubModule
    nn.ModuleList = _StubModule
    fake.nn = nn
    sys.modules["torch.nn"] = nn

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def get_device_name(idx):  # pragma: no cover — only hit when is_available True
            raise RuntimeError("CUDA not available in tests")

        @staticmethod
        def get_device_capability(idx):  # pragma: no cover
            raise RuntimeError("CUDA not available in tests")

    fake.cuda = _Cuda

    class _Generator:
        def __init__(self, device="cpu"):
            self.device = device

        def manual_seed(self, seed):
            return self

    fake.Generator = _Generator

    sys.modules["torch"] = fake


_install_torch_stub()
