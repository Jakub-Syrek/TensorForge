"""FastAPI server for AiPictureModifier.

Endpoints:
  GET  /              -> serves the single-page UI
  GET  /api/health    -> reports GPU + sm version
  POST /api/edit      -> multipart: image, mask?, mode, prompt, steps, guidance, seed
"""

from __future__ import annotations

import io
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

from backend.imgutils import image_to_png_bytes
from backend.pipeline import EditRequest, FluxEditor

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
    }
    return JSONResponse(payload)


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
    mask_img: Image.Image | None = None
    if mode == "inpaint":
        if mask is None:
            raise HTTPException(400, "inpaint requires a mask")
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

    try:
        out = editor.edit(req)
    except Exception as exc:  # surface failures to the UI rather than 500-spinner
        raise HTTPException(500, f"edit failed: {exc}") from exc

    return Response(content=image_to_png_bytes(out), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
