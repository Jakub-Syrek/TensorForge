# AiPictureModifier

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
huggingface-cli login
```

## Run

```powershell
python backend\server.py
```

Open <http://127.0.0.1:8000>. First request downloads model weights (~24 GB
each) into the HF cache.

## Notes

- `enable_model_cpu_offload()` plus VAE slicing/tiling are required, not
  optional — Flux is ~24 GB in fp16, the card is 16 GB.
- Default 28 steps. Dropping to ~8 needs an acceleration LoRA (Hyper-SD or
  Flux-Turbo) loaded via `pipe.load_lora_weights(...)` in `backend/pipeline.py`.
