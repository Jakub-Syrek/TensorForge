"""FastAPI server for TensorForge.

Endpoints:
  GET  /              -> serves the single-page UI
  GET  /api/health    -> reports GPU + sm version
  GET  /api/progress  -> live job state + GPU stats
  POST /api/abort     -> request cancellation of running edit
  POST /api/edit      -> SYNCHRONOUS edit (legacy/compat)

  POST /api/tasks                    -> enqueue async edit, returns {task_id}
  GET  /api/tasks/{task_id}          -> task + variant statuses
  GET  /api/tasks                    -> list recent tasks
  POST /api/tasks/{task_id}/approve  -> body {variant_id}
  DELETE /api/tasks/{task_id}        -> wipe DB row + storage
  GET  /api/variants/{variant_id}/output -> raw PNG bytes
"""

from __future__ import annotations

import asyncio
import io
import os
import random
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

# PIL guards against decompression-bomb attacks with a 178 MP cap on the
# decoded pixel count. Full-res phone photos and any 4:3 image at ~16 kpx
# trip it — which makes /api/tasks 500 on perfectly normal uploads. This
# is a local single-user tool, not an exposed service, so we raise the
# ceiling well above any realistic camera output but keep *some* sanity
# limit instead of disabling the guard entirely (None would silently load
# truly hostile files into VRAM).
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500 MP
from pydantic import BaseModel

# Allow `python backend/server.py` execution.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import loras, storage
from backend.bg_remove import BackgroundRemover
from backend.controlnet import UNION_MODES, ControlNetGenerator
from backend.db import Task, Variant, get_session, init_db
from backend.imgutils import fit_long_edge, fit_to_flux_bucket, image_to_png_bytes, sharpen
from backend.intent import explain as classify_intent
from backend.ipa import IPAdapterGenerator
from backend.pipeline import QUANT_MODE, EditAborted, EditRequest, FluxEditor
from backend.progress import job_progress, query_gpu_stats
from backend.prompt_rewriter import PromptRewriter
from backend.pulid import PuLIDGenerator
from backend.upscale import Upscaler
from backend.video import VideoGenerator
from backend.vision import VisionAnalyzer
from backend.worker import TaskWorker

# Cap on the longest edge of the input image. The right value depends on
# whether NF4 is active:
#   bf16 + cpu_offload (default):  512 — bigger thrashes PCIe (~470s/step)
#   NF4 resident (FLUX_QUANT=4bit): 1024 — model fits in VRAM with headroom,
#                                  detail is bounded by Flux's training
#                                  resolution either way.
# Override either default with FLUX_MAX_EDGE env var.
MAX_EDGE = int(os.environ.get("FLUX_MAX_EDGE", "1024" if QUANT_MODE else "512"))

editor = FluxEditor()
# Florence-2 vision backend — lazy-loaded on first /api/vision/* hit so
# adding the import doesn't move 1.5 GB into VRAM at server startup.
vision = VisionAnalyzer()
# Real-ESRGAN upscaler — lazy-loaded on first request that picks
# ``upscale="real_esrgan_4x"``. ~67 MB weights, ~600 MB resident.
upscaler = Upscaler()
# Background removal — lazy-loaded on first /api/vision/bg_remove hit.
# BiRefNet-general weights are ~150 MB; runs on CPU via ONNX runtime
# so it doesn't compete with FLUX for VRAM.
bg_remover = BackgroundRemover()
# Qwen2.5-1.5B-Instruct prompt rewriter — lazy-loaded on first
# /api/prompt/expand hit. ~3 GB bf16; coexists with NF4 FLUX on 16 GB.
prompt_rewriter = PromptRewriter()
# FLUX ControlNet (Shakker-Labs Union-Pro) — lazy-loaded on first task
# with mode="control". ~30 GB on disk (FLUX.1-dev base + Union-Pro);
# ~8 GB VRAM under NF4. Cannot coexist with warm Kontext on 16 GB — the
# worker releases editor state before loading ControlNet.
controlnet = ControlNetGenerator()
# PuLID face-identity preservation — lazy-loaded on mode="pulid" tasks.
# Requires insightface + the PuLID-FLUX weights; ~7 GB resident under NF4
# (FLUX.1-dev base + PuLID adapter + ID encoder). Cannot coexist with
# warm Kontext on 16 GB — worker releases editor state before loading.
pulid = PuLIDGenerator()
# IP-Adapter (InstantX FLUX) — image-as-prompt style transfer. Uses
# their vendored custom pipeline + attention processors. ~25 GB disk
# (FLUX.1-dev shared with ControlNet / PuLID, + ~1 GB adapter, + ~1.5
# GB SigLIP encoder). ~8 GB VRAM under NF4. Cannot coexist with warm
# Kontext / ControlNet / PuLID on 16 GB.
ipa = IPAdapterGenerator()
# Video generator (LTX-Video + Wan 2.2) — lazy-loaded on the first
# mode="video" task. Heaviest pipeline in the project: ~14 GB VRAM for
# Wan 2.2 14B, ~6-9 GB for LTX. Cannot coexist with warm Kontext /
# ControlNet / PuLID / IPA on 16 GB; worker releases the FluxEditor
# state before loading.
video = VideoGenerator()
task_worker = TaskWorker(
    editor,
    vision=vision,
    upscaler=upscaler,
    controlnet=controlnet,
    pulid=pulid,
    ipa=ipa,
    video=video,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: ensure data dirs + DB schema exist, then bring up the worker
    # thread that drains the task queue.
    storage.init()
    init_db()
    task_worker.start()
    yield
    # Shutdown: signal the worker, give it a few seconds to flush. We don't
    # try to interrupt mid-inference here — abort flow is via /api/abort.
    task_worker.stop(timeout=5.0)


app = FastAPI(title="TensorForge", lifespan=lifespan)

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


@app.post("/api/prompt/expand")
async def prompt_expand(
    prompt: str = Form(...),
    intent: str = Form("generate"),
) -> JSONResponse:
    """Expand a short user prompt into a verbose FLUX-style description.

    intent='generate' (default): text-to-image rewrite — packs in
        lighting, composition, lens cues, atmosphere.
    intent='edit': Kontext-style rewrite — preserves the action verb,
        adds visual specifics about how the edit should look.

    First call downloads Qwen2.5-1.5B-Instruct (~3 GB) into the HF cache.
    Subsequent calls run in ~0.5-2 s on a warm model.
    """
    if intent not in {"generate", "edit"}:
        raise HTTPException(400, f"intent must be 'generate' or 'edit', got {intent!r}")
    if not prompt.strip():
        raise HTTPException(400, "prompt is empty")
    try:
        expanded = await asyncio.to_thread(prompt_rewriter.expand, prompt, intent)
    except Exception as ex:
        raise HTTPException(500, f"prompt expansion failed: {ex}") from ex
    return JSONResponse({"original": prompt, "expanded": expanded, "intent": intent})


@app.get("/api/loras")
def list_loras() -> JSONResponse:
    """List available style LoRAs for the UI dropdown. Static list — comes
    from ``backend.loras.STYLE_LORAS`` (built-ins + optional user JSON)."""
    return JSONResponse({"loras": loras.as_public_list()})


@app.post("/api/vision/bg_remove")
async def vision_bg_remove(
    image: UploadFile = File(...),
) -> Response:
    """Remove the background from ``image``, returning a transparent-PNG.

    Uses BiRefNet (via rembg) for high-quality alpha edges on hair, fur,
    and fabric. Output is RGBA at the input's native resolution. Subject
    isolation is per-pixel continuous, not binary — feathered edges.
    """
    img_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        img.load()
    except (UnidentifiedImageError, OSError) as ex:
        raise HTTPException(400, f"Could not decode image: {ex}") from ex
    except Image.DecompressionBombError as ex:
        raise HTTPException(413, f"Image too large: {ex}") from ex

    try:
        cutout = await asyncio.to_thread(bg_remover.remove, img)
    except Exception as ex:
        raise HTTPException(500, f"bg_remove failed: {ex}") from ex
    return Response(content=image_to_png_bytes(cutout), media_type="image/png")


@app.post("/api/vision/segment")
async def vision_segment(
    image: UploadFile = File(...),
    text: str = Form(...),
) -> Response:
    """Referring expression segmentation via Florence-2.

    Send an image + a phrase ("the dragon in the back", "the sky", "her
    coat") — Florence-2 returns a single PNG mask at the image's native
    resolution: white where the phrase grounds, black elsewhere.

    This is the auto-mask path that feeds inpaint without manual brushing.
    First call downloads ~1.5 GB of Florence-2 weights into HF cache; later
    calls take ~100-300 ms depending on image size.
    """
    if not text.strip():
        raise HTTPException(400, "text is empty")
    img_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        img.load()
    except (UnidentifiedImageError, OSError) as ex:
        raise HTTPException(400, f"Could not decode image: {ex}") from ex
    except Image.DecompressionBombError as ex:
        raise HTTPException(413, f"Image too large: {ex}") from ex

    try:
        mask = await asyncio.to_thread(vision.segment, img, text)
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex
    except Exception as ex:
        log_msg = f"vision.segment failed: {ex}"
        raise HTTPException(500, log_msg) from ex

    return Response(content=image_to_png_bytes(mask), media_type="image/png")


@app.post("/api/vision/caption")
async def vision_caption(
    image: UploadFile = File(...),
    depth: str = Form("fast"),
    level: str = Form("detailed"),
) -> JSONResponse:
    """Generate a text description of the scene.

    depth: 'fast' (BLIP-large, single sentence, ~0.9 GB VRAM) or
           'deep' (BLIP-2 OPT-2.7B, multi-sentence, ~5.4 GB VRAM).
           Default 'fast' keeps the UX snappy and the VRAM headroom big.

    level: legacy parameter retained for back-compat with the old
           Florence-2 backend; ignored by the current BLIP pipeline.
    """
    if depth not in {"fast", "deep"}:
        raise HTTPException(400, f"depth must be 'fast' or 'deep', got {depth!r}")
    img_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        img.load()
    except (UnidentifiedImageError, OSError) as ex:
        raise HTTPException(400, f"Could not decode image: {ex}") from ex
    except Image.DecompressionBombError as ex:
        raise HTTPException(413, f"Image too large: {ex}") from ex

    try:
        caption = await asyncio.to_thread(vision.caption, img, depth, level)
    except Exception as ex:
        raise HTTPException(500, f"caption failed: {ex}") from ex
    return JSONResponse({"caption": caption, "depth": depth, "level": level})


@app.post("/api/vision/detect")
async def vision_detect(
    image: UploadFile = File(...),
    text: str | None = Form(None),
) -> JSONResponse:
    """Open-vocabulary object detection.

    Without ``text``: returns boxes for every detected object (COCO-style
    classes from Florence-2's generic OD head).
    With ``text``: caption-to-phrase grounding — Florence-2 grounds the
    named phrase(s) in the image and returns matching boxes. Use this to
    locate "the dragon" or "two figures in the background" without
    segmenting them.

    Output is a list of {label, box: [x1,y1,x2,y2]} dicts in image-pixel
    coordinates. UI surfaces labels as clickable chips that insert into
    the prompt textarea.
    """
    img_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        img.load()
    except (UnidentifiedImageError, OSError) as ex:
        raise HTTPException(400, f"Could not decode image: {ex}") from ex
    except Image.DecompressionBombError as ex:
        raise HTTPException(413, f"Image too large: {ex}") from ex

    try:
        objects = await asyncio.to_thread(vision.detect, img, (text or "").strip() or None)
    except Exception as ex:
        raise HTTPException(500, f"detect failed: {ex}") from ex
    return JSONResponse(
        {
            "objects": [{"label": o.label, "box": list(o.box), "score": o.score} for o in objects],
        }
    )


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


class ApproveBody(BaseModel):
    variant_id: str


class PipelineStep(BaseModel):
    mode: str = "auto"
    prompt: str
    steps: int | None = None
    guidance: float | None = None
    use_accel: bool | None = None
    sharpen_level: str | None = None
    max_edge: int | None = None
    gen_width: int | None = None
    gen_height: int | None = None
    style_lora: str | None = None
    style_lora_scale: float | None = None
    upscale: str | None = None


def _serialize_variant(v: Variant) -> dict:
    return {
        "id": v.id,
        "seed": v.seed,
        "status": v.status,
        "approved": v.approved,
        "runtime_ms": v.runtime_ms,
        "error": v.error,
        "output_url": f"/api/variants/{v.id}/output" if v.output_path else None,
        "created_at": v.created_at.isoformat() if v.created_at else None,
        "finished_at": v.finished_at.isoformat() if v.finished_at else None,
    }


def _serialize_task(t: Task) -> dict:
    return {
        "id": t.id,
        "status": t.status,
        "mode": t.mode,
        "prompt": t.prompt,
        "variants_requested": t.variants_requested,
        "params": t.params,
        "original_size": [t.original_width, t.original_height],
        "error": t.error,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "finished_at": t.finished_at.isoformat() if t.finished_at else None,
        "variants": [_serialize_variant(v) for v in t.variants],
    }


@app.post("/api/tasks")
async def create_task(
    mode: str = Form(...),
    prompt: str = Form(...),
    image: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),
    steps: int = Form(28),
    guidance: float = Form(3.5),
    seed: int | None = Form(None),
    use_accel: bool = Form(True),
    sharpen_level: str = Form("off"),
    max_edge: int | None = Form(None),
    variants: int = Form(1),
    gen_width: int = Form(1024),
    gen_height: int = Form(1024),
    style_lora: str | None = Form(None),
    style_lora_scale: float = Form(1.0),
    upscale: str = Form("lanczos"),
    ip_scale: float = Form(0.7),
    control_type: str = Form("canny"),
    control_scale: float = Form(0.7),
    id_scale: float = Form(1.0),
    video_backend: str = Form("ltx"),
    video_subtype: str = Form("t2v"),
    num_frames: int = Form(81),
    fps: int = Form(24),
) -> JSONResponse:
    """Enqueue an edit/generate task. Returns task id and initial state.

    mode values:
      - 'kontext'/'inpaint'/'qwen': image-conditioned edit
      - 'generate': text-to-image (no input image needed)
      - 'auto': server picks 'kontext' or 'generate' from prompt keywords
      - 'control': FLUX ControlNet generation. ``image`` is interpreted as
        the conditioning map (canny edges, depth map, pose skeleton —
        controlled by ``control_type``).
      - 'pulid': face-preserving generation. ``image`` is interpreted as
        the reference face photo; ``prompt`` drives the scene, PuLID
        keeps the face recognizable across stylization.
    """
    valid_modes = {
        "kontext",
        "inpaint",
        "qwen",
        "generate",
        "auto",
        "control",
        "pulid",
        "ipa",
        "video",
    }
    if mode not in valid_modes:
        raise HTTPException(400, f"mode must be one of {sorted(valid_modes)}, got {mode!r}")
    if not prompt.strip():
        raise HTTPException(400, "prompt is empty")
    if mode == "inpaint" and mask is None:
        raise HTTPException(400, "inpaint requires a mask")
    if mode == "control":
        if image is None:
            raise HTTPException(400, "control requires a control image (uploaded as 'image')")
        if control_type not in UNION_MODES:
            raise HTTPException(
                400,
                f"control_type must be one of {sorted(UNION_MODES)}, got {control_type!r}",
            )
    if mode == "pulid" and image is None:
        raise HTTPException(400, "pulid requires a face reference (uploaded as 'image')")
    if mode == "ipa" and image is None:
        raise HTTPException(400, "ipa requires a reference image (uploaded as 'image')")
    if mode == "video":
        if video_backend not in {"ltx", "wan"}:
            raise HTTPException(
                400,
                f"video_backend must be 'ltx' or 'wan', got {video_backend!r}",
            )
        if video_subtype not in {"t2v", "i2v"}:
            raise HTTPException(
                400,
                f"video_subtype must be 't2v' or 'i2v', got {video_subtype!r}",
            )
        if video_subtype == "i2v" and image is None:
            raise HTTPException(
                400,
                "video i2v requires a reference image (uploaded as 'image')",
            )
    clamped_control_scale = max(0.0, min(2.0, float(control_scale)))
    clamped_id_scale = max(0.0, min(2.0, float(id_scale)))

    # Resolve 'auto' -> concrete mode based on prompt keywords + image presence.
    routing = None
    if mode == "auto":
        routing = classify_intent(prompt, has_image=image is not None)
        mode = "kontext" if routing["intent"] == "edit" else "generate"

    # Modes that genuinely need no input image: text-to-image and text-to-video.
    image_optional = mode == "generate" or (mode == "video" and video_subtype == "t2v")
    if not image_optional and image is None:
        raise HTTPException(400, f"mode {mode!r} requires an input image")

    variants_n = max(1, min(8, int(variants)))
    effective_max_edge = MAX_EDGE if max_edge is None else max(64, min(2048, int(max_edge)))

    # Validate upscale mode against the supported set; silently fall back
    # to the LANCZOS default for unknown strings so an old client doesn't
    # 400 out on a typo.
    if upscale not in {"lanczos", "real_esrgan_4x"}:
        upscale = "lanczos"

    # Validate style LoRA id against the registry; ignore if not compatible
    # with the resolved mode so we don't poison Fill / Qwen runs.
    resolved_style_id: str | None = None
    if style_lora:
        lora_obj = loras.get(style_lora)
        if lora_obj is None:
            raise HTTPException(400, f"unknown style_lora: {style_lora!r}")
        if loras.is_compatible(lora_obj, mode):
            resolved_style_id = lora_obj.id
    clamped_style_scale = max(0.0, min(2.0, float(style_lora_scale)))

    # Image is optional for generate; load + persist if present.
    img_bytes = None
    original_w = original_h = None
    if image is not None:
        img_bytes = await image.read()
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.load()  # force decode now so we surface format errors here
        except Image.DecompressionBombError as ex:
            raise HTTPException(
                413,
                f"Image too large: {ex}. Resize before uploading "
                f"(current cap: {Image.MAX_IMAGE_PIXELS // 1_000_000} MP).",
            ) from ex
        except (UnidentifiedImageError, OSError) as ex:
            raise HTTPException(400, f"Could not decode image: {ex}") from ex
        img = ImageOps.exif_transpose(img)
        original_w, original_h = img.size

    # Generate width/height clamped to /16 and a sane range.
    gen_w = max(256, min(2048, (int(gen_width) // 16) * 16))
    gen_h = max(256, min(2048, (int(gen_height) // 16) * 16))

    base_seed = random.randint(0, 2**31 - 1) if seed in (None, "") else int(seed)
    seeds = [base_seed if i == 0 else random.randint(0, 2**31 - 1) for i in range(variants_n)]

    with get_session() as s:
        t = Task(
            mode=mode,
            prompt=prompt,
            input_path=None,
            original_width=original_w,
            original_height=original_h,
            variants_requested=variants_n,
            params={
                "steps": int(steps),
                "guidance": float(guidance),
                "use_accel": bool(use_accel),
                "sharpen_level": sharpen_level,
                "max_edge": effective_max_edge,
                "base_seed": base_seed,
                "gen_width": gen_w,
                "gen_height": gen_h,
                "routing": routing,  # null unless mode was 'auto'
                "style_lora_id": resolved_style_id,
                "style_lora_scale": clamped_style_scale,
                "upscale": upscale,
                "ip_scale": max(0.0, min(2.0, float(ip_scale))),
                "control_type": control_type,
                "control_scale": clamped_control_scale,
                "id_scale": clamped_id_scale,
                "video_backend": video_backend,
                "video_subtype": video_subtype,
                "num_frames": max(8, min(257, int(num_frames))),
                "fps": max(8, min(60, int(fps))),
            },
        )
        s.add(t)
        s.flush()  # need t.id for storage paths

        if img_bytes is not None:
            input_path = storage.save_input(t.id, img_bytes)
            t.input_path = str(input_path)
        if mode == "inpaint" and mask is not None:
            mask_bytes = await mask.read()
            t.mask_path = str(storage.save_mask(t.id, mask_bytes))

        for sd in seeds:
            s.add(Variant(task_id=t.id, seed=sd))
        s.commit()
        s.refresh(t)
        return JSONResponse(_serialize_task(t), status_code=202)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> JSONResponse:
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(404, "task not found")
        return JSONResponse(_serialize_task(t))


@app.get("/api/tasks")
def list_tasks(limit: int = 20) -> JSONResponse:
    """Most recent tasks first. Default 20, intended for the history strip."""
    limit = max(1, min(100, int(limit)))
    with get_session() as s:
        rows = s.query(Task).order_by(Task.created_at.desc()).limit(limit).all()
        return JSONResponse({"tasks": [_serialize_task(t) for t in rows]})


@app.get("/api/prompts/recent")
def list_recent_prompts(limit: int = 20) -> JSONResponse:
    """Return the N most recent unique non-empty prompts.

    Sourced from the Task table — every submitted generation/edit
    leaves its prompt here, so this doubles as a "prompt history"
    feature without needing a separate table. Duplicates are dropped
    (the same prompt run multiple times shows up once, at its newest
    submission). Each entry carries the resolved mode + creation
    timestamp so the UI can render a mode chip and relative time.

    Implementation detail: SQLite lacks ``DISTINCT ON``. We over-fetch
    ``limit * 3`` rows and dedupe in Python, which is fast on a
    single-user desktop DB (typically <1k tasks total).
    """
    limit = max(1, min(100, int(limit)))
    with get_session() as s:
        rows = (
            s.query(Task.prompt, Task.mode, Task.created_at)
            .filter(Task.prompt.isnot(None))
            .filter(Task.prompt != "")
            .order_by(Task.created_at.desc())
            .limit(limit * 3)
            .all()
        )
        seen: set[str] = set()
        unique: list[dict] = []
        for prompt, mode, created_at in rows:
            key = prompt.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(
                {
                    "prompt": prompt,
                    "mode": mode,
                    "created_at": created_at.isoformat() if created_at else None,
                }
            )
            if len(unique) >= limit:
                break
        return JSONResponse({"prompts": unique})


@app.post("/api/tasks/{task_id}/approve")
def approve_variant(task_id: str, body: ApproveBody) -> JSONResponse:
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(404, "task not found")
        v = s.get(Variant, body.variant_id)
        if v is None or v.task_id != task_id:
            raise HTTPException(404, "variant not found under this task")
        if v.status != "done":
            raise HTTPException(409, f"variant is not done (status: {v.status})")
        # Clear previous approval on the same task — only one approved variant per task.
        for sibling in t.variants:
            if sibling.id != v.id:
                sibling.approved = False
        v.approved = True
        t.status = "approved"
        t.finished_at = t.finished_at or datetime.utcnow()
        s.commit()
        s.refresh(t)
        return JSONResponse(_serialize_task(t))


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str) -> JSONResponse:
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(404, "task not found")
        s.delete(t)
        s.commit()
    storage.delete_task_dir(task_id)
    return JSONResponse({"deleted": task_id})


@app.post("/api/pipelines")
async def create_pipeline(
    steps_json: str = Form(...),
    image: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),
) -> JSONResponse:
    """Create a multi-step pipeline. Steps run sequentially: each step's
    output (its first 'done' variant) becomes the next step's input.

    Form field `steps_json` is a JSON-encoded list of PipelineStep objects
    (1 to 10). Per-step params override defaults from the first step.
    The optional `image` is the initial input to step 0 (omit for a
    'generate-first' chain); `mask` applies only if step 0 is 'inpaint'.
    """
    import json

    try:
        raw_steps = json.loads(steps_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(400, f"steps_json must be valid JSON: {exc}") from exc
    if not isinstance(raw_steps, list) or not (1 <= len(raw_steps) <= 10):
        raise HTTPException(400, "steps must be a list of 1-10 step objects")

    parsed_steps = []
    for i, raw in enumerate(raw_steps):
        try:
            parsed_steps.append(PipelineStep(**raw))
        except Exception as exc:
            raise HTTPException(400, f"step {i}: {exc}") from exc
        if not parsed_steps[-1].prompt.strip():
            raise HTTPException(400, f"step {i}: prompt is empty")
        if parsed_steps[-1].mode not in {
            "kontext",
            "inpaint",
            "qwen",
            "generate",
            "auto",
            "auto_mask",
        }:
            raise HTTPException(400, f"step {i}: invalid mode {parsed_steps[-1].mode!r}")

    pipeline_id = uuid.uuid4().hex

    # Persist initial image (if any) — step 0 picks it up.
    img_bytes = await image.read() if image is not None else None
    mask_bytes = await mask.read() if mask is not None else None
    original_w = original_h = None
    if img_bytes:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        original_w, original_h = img.size

    created_tasks: list[Task] = []
    with get_session() as s:
        for i, step in enumerate(parsed_steps):
            # Resolve auto -> concrete mode, considering whether THIS step
            # has an effective image input. Step 0 uses the uploaded image;
            # later steps will have a parent variant set by the worker, but
            # at submission time we don't have that yet — treat them as
            # having an image (the pipeline implicitly chains).
            mode = step.mode
            routing = None
            if mode == "auto":
                has_img = i > 0 or img_bytes is not None
                routing = classify_intent(step.prompt, has_image=has_img)
                mode = "kontext" if routing["intent"] == "edit" else "generate"

            # Resolve + clamp the per-step style LoRA. Unknown ids are
            # rejected; incompatible-with-mode is silently dropped (so a
            # 'fantasy' LoRA on an inpaint step doesn't kill the pipeline).
            step_style_id: str | None = None
            if step.style_lora:
                lora_obj = loras.get(step.style_lora)
                if lora_obj is None:
                    raise HTTPException(400, f"step {i}: unknown style_lora {step.style_lora!r}")
                if loras.is_compatible(lora_obj, mode):
                    step_style_id = lora_obj.id
            step_style_scale = (
                max(0.0, min(2.0, float(step.style_lora_scale)))
                if step.style_lora_scale is not None
                else 1.0
            )

            step_upscale = (
                step.upscale if step.upscale in {"lanczos", "real_esrgan_4x"} else "lanczos"
            )
            params = {
                "steps": step.steps if step.steps is not None else 28,
                "guidance": step.guidance if step.guidance is not None else 3.5,
                "use_accel": step.use_accel if step.use_accel is not None else True,
                "sharpen_level": step.sharpen_level or "off",
                "max_edge": step.max_edge if step.max_edge is not None else MAX_EDGE,
                "gen_width": step.gen_width if step.gen_width is not None else 1024,
                "gen_height": step.gen_height if step.gen_height is not None else 1024,
                "routing": routing,
                "style_lora_id": step_style_id,
                "style_lora_scale": step_style_scale,
                "upscale": step_upscale,
            }
            t = Task(
                mode=mode,
                prompt=step.prompt,
                input_path=None,
                original_width=original_w if i == 0 else None,
                original_height=original_h if i == 0 else None,
                variants_requested=1,  # pipeline: single-track
                params=params,
                status="queued" if i == 0 else "blocked",
                pipeline_id=pipeline_id,
                pipeline_step=i,
            )
            s.add(t)
            s.flush()  # need t.id for storage paths
            created_tasks.append(t)

            if i == 0 and img_bytes is not None:
                t.input_path = str(storage.save_input(t.id, img_bytes))
            if i == 0 and mode == "inpaint" and mask_bytes is not None:
                t.mask_path = str(storage.save_mask(t.id, mask_bytes))

            # Single variant per step (pipeline is sequential, no branching).
            s.add(
                Variant(
                    task_id=t.id,
                    seed=random.randint(0, 2**31 - 1),
                    status="queued" if i == 0 else "blocked",
                )
            )

        s.commit()
        for t in created_tasks:
            s.refresh(t)
        return JSONResponse(
            {
                "pipeline_id": pipeline_id,
                "steps": [_serialize_task(t) for t in created_tasks],
            },
            status_code=202,
        )


@app.get("/api/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: str) -> JSONResponse:
    with get_session() as s:
        rows = (
            s.query(Task)
            .filter(Task.pipeline_id == pipeline_id)
            .order_by(Task.pipeline_step.asc())
            .all()
        )
        if not rows:
            raise HTTPException(404, "pipeline not found")
        statuses = {t.status for t in rows}
        if statuses == {"done"} or statuses == {"approved"} or statuses == {"done", "approved"}:
            agg = "done"
        elif "failed" in statuses:
            agg = "failed"
        elif "running" in statuses or "queued" in statuses:
            agg = "running"
        else:
            agg = "blocked"
        return JSONResponse(
            {
                "pipeline_id": pipeline_id,
                "status": agg,
                "steps": [_serialize_task(t) for t in rows],
            }
        )


_OUTPUT_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


@app.get("/api/variants/{variant_id}/output")
def variant_output(variant_id: str) -> FileResponse:
    with get_session() as s:
        v = s.get(Variant, variant_id)
        if v is None:
            raise HTTPException(404, "variant not found")
        if not v.output_path:
            raise HTTPException(404, "variant has no output yet")
        p = Path(v.output_path)
        if not p.exists():
            raise HTTPException(410, "output file gone (was the task deleted?)")
        media_type = _OUTPUT_MEDIA_TYPES.get(p.suffix.lower(), "application/octet-stream")
        return FileResponse(p, media_type=media_type)


@app.post("/api/edit")
async def edit(
    image: UploadFile = File(...),
    mode: str = Form(...),
    prompt: str = Form(...),
    mask: UploadFile | None = File(None),
    steps: int = Form(28),
    guidance: float = Form(3.5),
    seed: int | None = Form(None),
    use_accel: bool = Form(True),
    sharpen_level: str = Form("off"),
    max_edge: int | None = Form(None),
) -> Response:
    if mode not in {"kontext", "inpaint", "qwen"}:
        raise HTTPException(400, f"mode must be 'kontext', 'inpaint', or 'qwen', got {mode!r}")
    if not prompt.strip():
        raise HTTPException(400, "prompt is empty")

    # Per-request max_edge override (clamped to a sane range). Falls back to
    # the env-configured server default when the client omits it.
    effective_max_edge = MAX_EDGE if max_edge is None else max(64, min(2048, int(max_edge)))

    img = Image.open(io.BytesIO(await image.read()))
    # Apply EXIF orientation to the actual pixels. Phone cameras store the
    # sensor's raw landscape pixels and tag a 'rotate 90°' EXIF flag for
    # portrait shots; PIL doesn't honor that flag on open, so without this
    # the model sees the image sideways while the browser preview shows it
    # upright. exif_transpose rewrites pixels + strips the flag — input and
    # output orientations stay aligned with how the user sees them.
    img = ImageOps.exif_transpose(img)
    original_size = img.size
    # Flux Kontext / Fill bucket aspect ratios internally during inference,
    # which side-crops content when the input doesn't land on a known bucket.
    # Pre-snap to the closest Flux bucket on the server so the model has
    # nothing left to crop; the LANCZOS round-trip then restores the user's
    # original aspect (with a few-percent stretch absorbed). Qwen has its
    # own architecture and bucket profile — leave it on the simpler fit.
    if mode in ("kontext", "inpaint"):
        img = fit_to_flux_bucket(img, effective_max_edge)
    else:
        img = fit_long_edge(img, effective_max_edge)
    if img.size != original_size:
        print(
            f"resized input {original_size} -> {img.size} (mode={mode}, max_edge={effective_max_edge})"
        )

    mask_img: Image.Image | None = None
    if mode == "inpaint":
        if mask is None:
            raise HTTPException(400, "inpaint requires a mask")
        # Mask is resized to match the image inside the pipeline; no resize here.
        mask_img = Image.open(io.BytesIO(await mask.read()))

    # Resolve seed up-front so we can surface the actually-used value via
    # response header. Empty/missing = generate a fresh int32; user-supplied
    # = honored verbatim. This is the key affordance for the A/B iteration
    # workflow (tweak prompt, keep seed) — without it the user can't pin
    # the seed of a random run.
    used_seed = random.randint(0, 2**31 - 1) if seed in (None, "") else int(seed)

    req = EditRequest(
        mode=mode,  # type: ignore[arg-type]
        prompt=prompt,
        image=img,
        mask=mask_img,
        steps=int(steps),
        guidance=float(guidance),
        seed=used_seed,
        use_accel=bool(use_accel),
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

    # Optional post-processing sharpen — runs AFTER the LANCZOS upscale,
    # since that's the step that introduced the softness we're undoing.
    out = sharpen(out, sharpen_level)  # type: ignore[arg-type]

    return Response(
        content=image_to_png_bytes(out),
        media_type="image/png",
        headers={"X-Used-Seed": str(used_seed)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
