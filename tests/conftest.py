"""Test-wide setup.

backend.pipeline / backend.server import torch at module load. CI doesn't
install torch (would add ~2 minutes for nothing useful in routing tests),
so we install a minimal stub into sys.modules before any test module
imports backend.*. The stub only carries the attributes our code touches.
"""

from __future__ import annotations

import sys
import types


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    fake = types.ModuleType("torch")
    fake.__version__ = "stub"
    fake.bfloat16 = object()
    fake.float32 = object()

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
