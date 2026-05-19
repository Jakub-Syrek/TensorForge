"""FastAPI server for AiPictureModifier.

Endpoints:
  GET  /              -> serves the single-page UI
  GET  /api/health    -> reports GPU + sm version
  POST /api/edit      -> multipart: image, mask?, mode, prompt, steps, guidance, seed
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Allow `python backend/server.py` execution.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.imgutils import fit_long_edge, image_to_png_bytes
from backend.pipeline import QUANT_MODE, EditAborted, EditRequest, FluxEditor
from backend.progress import job_progress, query_gpu_stats

# Cap on the longest edge of the input image. 512 is the conservative
# default for a 16 GB 5080 with cpu_offload — empirically 1024 saturates
# VRAM (~95%) and degrades to ~470s/step due to PCIe thrashing; 768 is
# borderline; 512 leaves real headroom. Detail is bounded by Flux's
# internal latent resolution anyway. Override with FLUX_MAX_EDGE.
MAX_EDGE = int(os.environ.get("FLUX_MAX_EDGE", "512"))

app = FastAPI(title="AiPictureModifier")
editor = FluxEditor()

FRONTEND_DIR = REPO_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    cuda = torch.cuda.is_available()
    accel = editor.accel
    payload = {
        "torch": torch.__version__,
        "cuda_available": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "accel": (
            {"repo": accel.repo, "weight": accel.weight_name, "scale": accel.scale}
            if accel is not None
            else None
        ),
        "quant": QUANT_MODE,
        "max_edge": MAX_EDGE,
    }
    return JSONResponse(payload)


@app.get("/api/progress")
def progress() -> JSONResponse:
    gpu = query_gpu_stats()
    return JSONResponse(
        {
            "job": job_progress.snapshot(),
            "gpu": gpu.to_dict() if gpu is not None else None,
        }
    )


@app.post("/api/abort")
def abort() -> JSONResponse:
    """Signal the running edit (if any) to abort at the next step boundary.

    diffusers can only be interrupted between denoising steps, so abort is
    near-instant during inference (≤ 1 step latency) and a no-op during
    pre-loop phases (text encoding, model loading)."""
    requested = job_progress.request_abort()
    status = 200 if requested else 409
    return JSONResponse(
        {"aborted": requested, "active": job_progress.active},
        status_code=status,
    )


@app.post("/api/edit")
async def edit(
    image: UploadFile = File(...),
    mode: str = Form(...),
    prompt: str = Form(...),
    mask: UploadFile | None = File(None),
    steps: int = Form(28),
    guidance: float = Form(3.5),
    seed: int | None = Form(None),
) -> Response:
    if mode not in {"kontext", "inpaint"}:
        raise HTTPException(400, f"mode must be 'kontext' or 'inpaint', got {mode!r}")
    if not prompt.strip():
        raise HTTPException(400, "prompt is empty")

    img = Image.open(io.BytesIO(await image.read()))
    original_size = img.size
    img = fit_long_edge(img, MAX_EDGE)
    if img.size != original_size:
        print(f"resized input {original_size} -> {img.size} (FLUX_MAX_EDGE={MAX_EDGE})")

    mask_img: Image.Image | None = None
    if mode == "inpaint":
        if mask is None:
            raise HTTPException(400, "inpaint requires a mask")
        # Mask is resized to match the image inside the pipeline; no resize here.
        mask_img = Image.open(io.BytesIO(await mask.read()))

    req = EditRequest(
        mode=mode,  # type: ignore[arg-type]
        prompt=prompt,
        image=img,
        mask=mask_img,
        steps=int(steps),
        guidance=float(guidance),
        seed=int(seed) if seed not in (None, "") else None,
    )

    job_progress.start(total=int(steps), mode=mode)
    try:
        # Offload the blocking inference to a worker thread so the FastAPI
        # event loop stays free to serve /api/progress + /api/abort concurrently.
        out = await asyncio.to_thread(editor.edit, req)
    except EditAborted as exc:
        # User-initiated abort — 499 (NGINX convention: Client Closed Request).
        job_progress.finish(error=str(exc))
        raise HTTPException(499, f"edit aborted: {exc}") from exc
    except Exception as exc:  # surface failures to the UI rather than 500-spinner
        job_progress.finish(error=str(exc))
        raise HTTPException(500, f"edit failed: {exc}") from exc
    job_progress.finish()

    # Round-trip: restore the original canvas size so callers get back what
    # they sent in. Upscale is LANCZOS — doesn't invent detail beyond Flux's
    # 1024-px output, but matches user expectations.
    if out.size != original_size:
        out = out.resize(original_size, Image.Resampling.LANCZOS)

    return Response(content=image_to_png_bytes(out), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
