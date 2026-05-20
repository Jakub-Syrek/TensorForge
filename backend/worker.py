"""Single-machine background worker that pulls queued Variants and runs
inference. One thread, one GPU, strictly serial — matches the hardware
constraint (16 GB VRAM holds one Flux pipeline at a time under NF4).

Lifecycle:
  - start() on app startup, spawns one daemon thread that loops
  - stop() flips an event the loop checks between iterations
  - loop poll interval is 200 ms when idle, immediate on hit
  - each variant: status queued -> running -> done | failed
  - aborted via /api/abort is honored at the diffusers callback boundary
    (see backend/pipeline.py); the worker just transitions the row to
    'aborted' afterwards.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

from PIL import Image, ImageOps

from backend.db import SessionLocal, Task, Variant
from backend.imgutils import fit_long_edge, fit_to_flux_bucket, image_to_png_bytes, sharpen
from backend.pipeline import EditAborted, EditRequest, FluxEditor
from backend.progress import job_progress
from backend.storage import save_variant_output

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.2


class TaskWorker:
    def __init__(self, editor: FluxEditor) -> None:
        self.editor = editor
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:  # pragma: no cover — spawns a thread, not deterministically testable
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="task-worker", daemon=True)
        self._thread.start()
        log.info("task worker started")

    def stop(self, timeout: float = 5.0) -> None:  # pragma: no cover — thread lifecycle
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:  # pragma: no cover — long-running, exercised in real server
        while not self._stop.is_set():
            variant_id = self._claim_next_variant()
            if variant_id is None:
                time.sleep(POLL_INTERVAL_S)
                continue
            self._process_variant(variant_id)

    def _claim_next_variant(self) -> str | None:
        """Atomically pick the oldest 'queued' variant and flip it to 'running'.

        SQLite's per-DB lock combined with the single-worker design makes the
        SELECT-then-UPDATE race-free in practice — but we still wrap the
        transition in a transaction for safety against future multi-worker.
        """
        with SessionLocal() as s:
            v = (
                s.query(Variant)
                .filter(Variant.status == "queued")
                .order_by(Variant.created_at.asc())
                .first()
            )
            if v is None:
                return None
            v.status = "running"
            t = s.get(Task, v.task_id)
            if t is not None and t.status == "queued":
                t.status = "running"
                t.started_at = datetime.utcnow()
            s.commit()
            return v.id

    def _process_variant(self, variant_id: str) -> None:  # pragma: no cover — calls GPU _render
        start = time.monotonic()
        with SessionLocal() as s:
            v = s.get(Variant, variant_id)
            if v is None:
                return
            t = s.get(Task, v.task_id)
            if t is None:
                v.status = "failed"
                v.error = "parent task missing"
                v.finished_at = datetime.utcnow()
                s.commit()
                return

            try:
                output_bytes = self._render(t, v)
            except EditAborted as exc:
                v.status = "aborted"
                v.error = str(exc)
                v.finished_at = datetime.utcnow()
                self._maybe_finalize_task(s, t)
                s.commit()
                return
            except Exception as exc:
                log.exception("variant %s failed", variant_id)
                v.status = "failed"
                v.error = str(exc)
                v.finished_at = datetime.utcnow()
                self._maybe_finalize_task(s, t)
                s.commit()
                return

            path = save_variant_output(t.id, v.id, output_bytes)
            v.output_path = str(path)
            v.status = "done"
            v.finished_at = datetime.utcnow()
            v.runtime_ms = int((time.monotonic() - start) * 1000)
            self._maybe_finalize_task(s, t)
            s.commit()

    def _maybe_finalize_task(self, s, t: Task) -> None:
        """Once every variant of a task has reached a terminal state,
        roll up the task status."""
        remaining = (
            s.query(Variant)
            .filter(Variant.task_id == t.id, Variant.status.in_(["queued", "running"]))
            .count()
        )
        if remaining > 0:
            return
        statuses = {v.status for v in t.variants}
        if statuses == {"done"}:
            t.status = "done"
        elif "done" in statuses:
            t.status = "done"  # partial success still counts as 'done' for the task
        elif "aborted" in statuses:
            t.status = "aborted"
        else:
            t.status = "failed"
        t.finished_at = datetime.utcnow()

    def _render(self, t: Task, v: Variant) -> bytes:  # pragma: no cover — GPU inference path
        params = t.params or {}
        max_edge = int(params.get("max_edge", 1024))
        steps = int(params.get("steps", 28))
        guidance = float(params.get("guidance", 3.5))
        use_accel = bool(params.get("use_accel", True))
        sharpen_level = params.get("sharpen_level", "off")

        img = Image.open(t.input_path)
        img = ImageOps.exif_transpose(img)
        original_size = (t.original_width, t.original_height)

        if t.mode in ("kontext", "inpaint"):
            img = fit_to_flux_bucket(img, max_edge)
        else:
            img = fit_long_edge(img, max_edge)

        mask = None
        if t.mode == "inpaint" and t.mask_path:
            mask = Image.open(t.mask_path)

        job_progress.start(total=steps, mode=t.mode)
        try:
            req = EditRequest(
                mode=t.mode,
                prompt=t.prompt,
                image=img,
                mask=mask,
                steps=steps,
                guidance=guidance,
                seed=v.seed,
                use_accel=use_accel,
            )
            out = self.editor.edit(req)
            job_progress.finish()
        except EditAborted:
            job_progress.finish(error="aborted")
            raise
        except Exception as exc:
            job_progress.finish(error=str(exc))
            raise

        if out.size != original_size:
            out = out.resize(original_size, Image.Resampling.LANCZOS)
        out = sharpen(out, sharpen_level)
        return image_to_png_bytes(out)


# Path helpers re-exported for convenience.
__all__ = ["TaskWorker"]
