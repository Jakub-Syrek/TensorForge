"""One-shot setup for AiPictureModifier.

Detects Blackwell (sm_120 — RTX 50xx) and installs the PyTorch nightly cu128
build that actually ships kernels for that architecture. Stable PyTorch only
ships up to sm_90; using it on a 5080 silently falls back to CPU or errors with
"no kernel image is available".

Run once inside your venv:
    python scripts/setup.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu128"


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd)


def detect_blackwell() -> bool:
    """Return True when an RTX 50xx (sm_120) GPU is visible to nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("nvidia-smi not found — proceeding without GPU detection.")
        return False

    names = [line.strip() for line in out.splitlines() if line.strip()]
    print(f"Detected GPUs: {names}")
    return any("RTX 50" in n for n in names)


def install_torch_nightly() -> None:
    run([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--pre", "torch", "torchvision",
        "--index-url", TORCH_NIGHTLY_INDEX,
    ])


def install_torch_stable() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision"])


def install_requirements() -> None:
    req = REPO_ROOT / "requirements.txt"
    run([sys.executable, "-m", "pip", "install", "-r", str(req)])


def verify_cuda() -> None:
    code = (
        "import torch;"
        "print('torch', torch.__version__);"
        "print('cuda available', torch.cuda.is_available());"
        "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU');"
        "print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None);"
        "x = torch.randn(4, 4, device='cuda' if torch.cuda.is_available() else 'cpu');"
        "print('tensor ok', (x @ x).sum().item())"
    )
    run([sys.executable, "-c", code])


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if detect_blackwell():
        print("Blackwell GPU detected — installing PyTorch nightly cu128.")
        install_torch_nightly()
    else:
        print("No Blackwell GPU detected — installing stable PyTorch.")
        install_torch_stable()

    install_requirements()
    verify_cuda()
    print("\nSetup complete.")


if __name__ == "__main__":
    main()
