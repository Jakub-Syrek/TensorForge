# AiPictureModifier

[![CI](https://github.com/Jakub-Syrek/AiPictureModifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Jakub-Syrek/AiPictureModifier/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch nightly cu128](https://img.shields.io/badge/pytorch-nightly%20cu128-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

Web app for prompt-driven image editing with FLUX, sized for an RTX 5080
(Blackwell · sm_120 · 16 GB).

- **Global edit** — FLUX.1 Kontext: keeps composition, follows instructions.
- **Inpaint** — FLUX.1 Fill: regenerates a painted mask region only.

## Setup

The 5080 needs PyTorch nightly **cu128** — stable PyTorch ships kernels only up
to sm_90 and will fall back to CPU. `scripts/setup.py` detects the card and
picks the right wheel.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python scripts\setup.py
```

You'll also need a Hugging Face token with access granted to
[`FLUX.1-Kontext-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
and [`FLUX.1-Fill-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev):

```powershell
hf auth login
```

(Older docs say `huggingface-cli login` — that command is deprecated in
`huggingface_hub` ≥ 1.0; use `hf` instead.)

## Run

```powershell
python backend\server.py
```

Open <http://127.0.0.1:8000>. First request downloads model weights (~24 GB
each) into the HF cache.

## Development

Pre-commit hook mirrors the CI lint step (ruff check + ruff format + a few
whitespace fixers) so the same gate that gates merges also gates commits.

```powershell
pip install pre-commit
pre-commit install            # one-time, wires .git/hooks/pre-commit
pre-commit run --all-files    # run on demand
```

Tests run torch-free (a stub is installed in `tests/conftest.py`), so
`pytest -q` works without the multi-gigabyte GPU stack — useful when iterating
on routing or pure helpers.

## Notes

- `enable_model_cpu_offload()` plus VAE slicing/tiling are required, not
  optional — Flux is ~24 GB in fp16, the card is 16 GB.
- Default is 28 steps. To drop to ~8 steps, set the acceleration LoRA via
  env vars before launching the server (read once at startup, fused into
  both pipelines):

  ```powershell
  $env:FLUX_ACCEL_REPO   = "ByteDance/Hyper-SD"
  $env:FLUX_ACCEL_WEIGHT = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
  $env:FLUX_ACCEL_SCALE  = "0.125"   # Hyper-SD recommended scale for 8 steps
  python backend\server.py
  ```

  Then set `steps=8` in the UI. Caveat: Hyper-SD and Flux-Turbo were trained
  on base FLUX.1-dev. Kontext and Fill share the transformer architecture so
  the weights load cleanly, but quality at 8 steps on these variants is not
  officially validated — verify visually before relying on it. Check
  `GET /api/health` to confirm the LoRA is active.
