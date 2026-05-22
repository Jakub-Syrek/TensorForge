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

from backend import storage
from backend.controlnet import ControlNetGenerator, ControlRequest
from backend.db import SessionLocal, Task, Variant
from backend.imgutils import fit_long_edge, fit_to_flux_bucket, image_to_png_bytes, sharpen
from backend.ipa import IPAdapterGenerator, IPARequest
from backend.pipeline import EditAborted, EditRequest, FluxEditor
from backend.progress import job_progress
from backend.pulid import PuLIDGenerator, PuLIDRequest
from backend.storage import save_variant_output
from backend.upscale import Upscaler
from backend.video import VideoGenerator, VideoRequest
from backend.vision import VisionAnalyzer

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.2


class TaskWorker:
    def __init__(
        self,
        editor: FluxEditor,
        vision: VisionAnalyzer | None = None,
        upscaler: Upscaler | None = None,
        controlnet: ControlNetGenerator | None = None,
        pulid: PuLIDGenerator | None = None,
        ipa: IPAdapterGenerator | None = None,
        video: VideoGenerator | None = None,
    ) -> None:
        self.editor = editor
        # Vision is optional so existing tests can construct the worker
        # without booting Florence-2. Production wiring passes the server's
        # shared instance so the model loads once across requests.
        self.vision = vision
        # Real-ESRGAN upscaler — also optional. When omitted the worker
        # falls back to LANCZOS, which is the legacy behavior.
        self.upscaler = upscaler
        # ControlNet generator — optional. Required for mode="control"
        # tasks; tests don't need it.
        self.controlnet = controlnet
        # PuLID generator — optional. Required for mode="pulid" tasks
        # (face-preserving generation).
        self.pulid = pulid
        # IP-Adapter generator (InstantX custom stack) — optional.
        # Required for mode="ipa" tasks (reference-image style transfer).
        self.ipa = ipa
        # Video generator (LTX-Video + Wan 2.2) — optional. Required for
        # mode="video" tasks (text-to-video, image-to-video).
        self.video = video
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
                if t.mode == "auto_mask":
                    output_bytes = self._render_auto_mask(t, v)
                elif t.mode == "control":
                    output_bytes = self._render_control(t, v)
                elif t.mode == "pulid":
                    output_bytes = self._render_pulid(t, v)
                elif t.mode == "ipa":
                    output_bytes = self._render_ipa(t, v)
                elif t.mode == "video":
                    output_bytes = self._render_video(t, v)
                else:
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

            # Video variants are .mp4; everything else is .png. The DB
            # column stores the path verbatim, so /api/variants/{id}/output
            # infers the media type from the suffix.
            output_ext = ".mp4" if t.mode == "video" else ".png"
            path = save_variant_output(t.id, v.id, output_bytes, ext=output_ext)
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

        # If this task is part of a pipeline AND it succeeded, unblock the
        # next step by wiring its parent_variant_id to one of our done
        # variants and flipping its status to 'queued'.
        if t.status == "done" and t.pipeline_id is not None:
            self._maybe_advance_pipeline(s, t)
        # Failure / abort in a pipeline propagates: mark all downstream
        # blocked steps as 'aborted' so the UI can show the chain stopped.
        elif t.status in ("failed", "aborted") and t.pipeline_id is not None:
            downstream = (
                s.query(Task)
                .filter(
                    Task.pipeline_id == t.pipeline_id,
                    Task.pipeline_step > (t.pipeline_step or 0),
                    Task.status == "blocked",
                )
                .all()
            )
            for d in downstream:
                d.status = "aborted"
                d.error = f"upstream step {t.pipeline_step} {t.status}"
                d.finished_at = datetime.utcnow()
                for dv in d.variants:
                    if dv.status == "blocked":
                        dv.status = "aborted"

    def _resolve_input_path(self, t: Task) -> str | None:
        """Where does this task's input image live? Pipeline mid-steps point
        at a previous variant's output via parent_variant_id; everything
        else uses input_path from the upload."""
        if t.parent_variant_id is None:
            return t.input_path
        # Lazy lookup so we don't widen the worker's session boundary —
        # the variant might not be eagerly loaded.
        from backend.db import SessionLocal

        with SessionLocal() as s:
            parent = s.get(Variant, t.parent_variant_id)
            if parent is None:
                return None
            return parent.output_path

    def _maybe_advance_pipeline(self, s, t: Task) -> None:
        """Find the next step in this pipeline and unblock it.

        Picks the first 'done' variant of the just-finished task as the
        source for the next step's input. If multiple variants succeeded
        we ignore the rest — pipeline mode is single-track by design;
        users who want variant-pick branching submit one task at a time.
        """
        next_step = t.pipeline_step + 1 if t.pipeline_step is not None else None
        if next_step is None:
            return
        nxt = (
            s.query(Task)
            .filter(
                Task.pipeline_id == t.pipeline_id,
                Task.pipeline_step == next_step,
                Task.status == "blocked",
            )
            .first()
        )
        if nxt is None:
            return
        # Source variant — first one that finished cleanly.
        source = next((v for v in t.variants if v.status == "done"), None)
        if source is None or source.output_path is None:
            # No usable output; leave the next step blocked. The pipeline
            # effectively stalls — a future user action (delete/retry)
            # handles it.
            return
        nxt.parent_variant_id = source.id
        nxt.status = "queued"
        # If this step was auto_mask, hand the produced mask off to the
        # next step's mask_path. Typical next step is inpaint, which then
        # has both the parent image AND a ready-made mask without any
        # brushing. If the next step doesn't use masks (e.g. kontext),
        # the mask_path is set but harmlessly ignored.
        produced_mask = (
            (t.params or {}).get("produced_mask_path") if t.mode == "auto_mask" else None
        )
        if produced_mask:
            nxt.mask_path = produced_mask
        # The single variant of the next step is also 'blocked' at submission
        # time — flip it to 'queued' so _claim_next_variant picks it up.
        for nv in nxt.variants:
            if nv.status == "blocked":
                nv.status = "queued"

    def _render_auto_mask(self, t: Task, v: Variant) -> bytes:  # pragma: no cover — GPU + Florence
        """Run Florence-2 segmentation as a pipeline-only step.

        The step's prompt holds the text to segment ("the dragon"); the
        input image comes from the previous step's variant (or upload for
        step 0). We don't edit the image — we just produce a mask. To keep
        the pipeline-chaining contract simple, this method returns the
        input image bytes UNCHANGED as the variant's output. The mask itself
        is saved to the task's storage dir and its path stashed in
        ``task.params['produced_mask_path']``; ``_maybe_advance_pipeline``
        then wires it onto the next step's ``mask_path``.

        Net effect: ``[auto_mask:X]`` followed by ``[inpaint]`` runs as
        "segment X, then inpaint that region" with no manual brushing.
        """
        if self.vision is None:
            raise RuntimeError("auto_mask step requires a VisionAnalyzer instance on TaskWorker")

        text = (t.prompt or "").strip()
        if not text:
            raise ValueError("auto_mask: empty prompt (need a segmentation phrase)")

        input_source_path = self._resolve_input_path(t)
        if not input_source_path:
            raise ValueError("auto_mask: no input image available for this step")

        img = Image.open(input_source_path)
        img = ImageOps.exif_transpose(img)
        # No resize / bucketing — Florence-2 handles arbitrary sizes and
        # downstream inpaint will do its own bucketing.

        # Mark progress so the live counter doesn't freeze on this step.
        # Florence-2 is a single forward pass, so 1 / 1 is the honest report.
        job_progress.start(total=1, mode="auto_mask")
        try:
            mask = self.vision.segment(img, text)
            job_progress.advance(0)
            job_progress.finish()
        except Exception as exc:
            job_progress.finish(error=str(exc))
            raise

        # Save the mask under this task's storage dir. We reuse save_mask
        # which writes data/tasks/{task_id}/mask.png — fine because
        # auto_mask never has its own brushed mask to collide with.
        mask_bytes = image_to_png_bytes(mask)
        mask_path = storage.save_mask(t.id, mask_bytes)

        # Stash the mask path in params so _maybe_advance_pipeline can wire
        # it onto the next step. We reassign params to a fresh dict so
        # SQLAlchemy's JSON column sees the mutation (no MutableDict here).
        new_params = dict(t.params or {})
        new_params["produced_mask_path"] = str(mask_path)
        t.params = new_params

        # The variant's output is the unchanged input image — keeps the
        # parent_variant_id chain transparent for the next step.
        return image_to_png_bytes(img)

    def _render_control(self, t: Task, v: Variant) -> bytes:  # pragma: no cover — GPU + ControlNet
        """Run FLUX ControlNet generation. The control image lives at
        ``t.input_path`` (uploaded as the 'image' field — repurposed for
        the conditioning map). All other params are sourced from
        ``t.params`` per the standard flow.
        """
        if self.controlnet is None:
            raise RuntimeError("control mode requires a ControlNetGenerator on TaskWorker")

        if not t.input_path:
            raise ValueError("control mode requires a control_image (uploaded as 'image')")

        params = t.params or {}
        control_type = params.get("control_type", "canny")
        control_scale = float(params.get("control_scale", 0.7))
        steps = int(params.get("steps", 28))
        guidance = float(params.get("guidance", 3.5))
        gen_width = int(params.get("gen_width", 1024))
        gen_height = int(params.get("gen_height", 1024))

        control_image = Image.open(t.input_path)
        control_image = ImageOps.exif_transpose(control_image)

        # ControlNet pipeline is heavy. Release any warm Flux editor
        # components to free VRAM before loading. We don't auto-reload
        # them afterwards — the next non-control task will pay the
        # reload cost. Acceptable for a single-user desktop tool.
        self.editor._release_intermediate_memory()

        job_progress.start(total=steps, mode="control")
        try:
            req = ControlRequest(
                prompt=t.prompt,
                control_image=control_image,
                control_type=control_type,
                control_scale=control_scale,
                steps=steps,
                guidance=guidance,
                seed=v.seed,
                width=gen_width,
                height=gen_height,
            )
            out = self.controlnet.generate(req)
            job_progress.finish()
        except Exception as exc:
            job_progress.finish(error=str(exc))
            raise

        return image_to_png_bytes(out)

    def _render_pulid(self, t: Task, v: Variant) -> bytes:  # pragma: no cover — GPU + PuLID
        """Run face-preserving generation. The face reference photo lives
        at ``t.input_path`` (uploaded as the 'image' field — repurposed
        as the face ID source). Prompt drives the scene; PuLID drives
        whose face appears in it.
        """
        if self.pulid is None:
            raise RuntimeError("pulid mode requires a PuLIDGenerator on TaskWorker")
        if not t.input_path:
            raise ValueError("pulid mode requires a face reference (uploaded as 'image')")

        params = t.params or {}
        id_scale = float(params.get("id_scale", 1.0))
        steps = int(params.get("steps", 28))
        guidance = float(params.get("guidance", 3.5))
        gen_width = int(params.get("gen_width", 1024))
        gen_height = int(params.get("gen_height", 1024))

        face_image = Image.open(t.input_path)
        face_image = ImageOps.exif_transpose(face_image)

        # Same VRAM consideration as ControlNet — PuLID loads FLUX.1-dev
        # + a 700 MB adapter + InsightFace. Release the main editor's
        # warm state before kicking off; next non-pulid task pays the
        # reload cost. Acceptable for a single-user desktop tool.
        self.editor._release_intermediate_memory()

        job_progress.start(total=steps, mode="pulid")
        try:
            req = PuLIDRequest(
                prompt=t.prompt,
                face_image=face_image,
                id_scale=id_scale,
                steps=steps,
                guidance=guidance,
                seed=v.seed,
                width=gen_width,
                height=gen_height,
            )
            out = self.pulid.generate(req)
            job_progress.finish()
        except Exception as exc:
            job_progress.finish(error=str(exc))
            raise
        return image_to_png_bytes(out)

    def _render_ipa(self, t: Task, v: Variant) -> bytes:  # pragma: no cover — GPU + IPA
        """Run InstantX FLUX IP-Adapter generation. ``t.input_path`` is the
        reference image (uploaded as 'image'); ``t.prompt`` drives the new
        scene; the reference biases style/composition via IP-Adapter."""
        if self.ipa is None:
            raise RuntimeError("ipa mode requires an IPAdapterGenerator on TaskWorker")
        if not t.input_path:
            raise ValueError("ipa mode requires a reference image (uploaded as 'image')")

        params = t.params or {}
        ip_scale = float(params.get("ip_scale", 0.7))
        steps = int(params.get("steps", 24))
        guidance = float(params.get("guidance", 3.5))
        gen_width = int(params.get("gen_width", 1024))
        gen_height = int(params.get("gen_height", 1024))

        ref_image = Image.open(t.input_path)
        ref_image = ImageOps.exif_transpose(ref_image)

        # Same VRAM dance as ControlNet / PuLID — release the main
        # editor's warm state before loading the heavy IPA pipeline.
        self.editor._release_intermediate_memory()

        job_progress.start(total=steps, mode="ipa")
        try:
            req = IPARequest(
                prompt=t.prompt,
                ref_image=ref_image,
                ip_scale=ip_scale,
                steps=steps,
                guidance=guidance,
                seed=v.seed,
                width=gen_width,
                height=gen_height,
            )
            out = self.ipa.generate(req)
            job_progress.finish()
        except Exception as exc:
            job_progress.finish(error=str(exc))
            raise
        return image_to_png_bytes(out)

    def _render_video(self, t: Task, v: Variant) -> bytes:  # pragma: no cover — GPU + video
        """Run a text-to-video or image-to-video generation.

        Params (from ``t.params``):
          - ``video_backend``: 'ltx' (fast) or 'wan' (HQ).
          - ``video_subtype``: 't2v' or 'i2v'. i2v requires ``t.input_path``.
          - ``num_frames``, ``fps``, ``gen_width``, ``gen_height``,
            ``steps``, ``guidance``.

        Returns the encoded MP4 bytes. The worker saves them with
        ``.mp4`` extension and the variant endpoint serves them with
        ``video/mp4`` media type.
        """
        if self.video is None:
            raise RuntimeError("video mode requires a VideoGenerator on TaskWorker")

        params = t.params or {}
        backend = params.get("video_backend", "ltx")
        subtype = params.get("video_subtype", "t2v")
        num_frames = int(params.get("num_frames", 81))
        fps = int(params.get("fps", 24))
        gen_width = int(params.get("gen_width", 768))
        gen_height = int(params.get("gen_height", 512))
        steps = int(params.get("steps", 28))
        guidance = float(params.get("guidance", 5.0))

        # VRAM safety for video: FLUX defaults gen_width/gen_height to 1024,
        # which is fine for image diffusion but blows up the 4D tensor for
        # video. Wan 2.2 A14B (MoE, ~28 GB total weights) at 1024^2 x 81
        # frames does not fit on a 16 GB card; even LTX strains. If the user
        # left the FLUX-sized default (1024+), auto-clamp to a backend-
        # appropriate target: LTX -> 768x512 (training resolution), Wan ->
        # 832x480 (16:9 480p, Wan-native). Anything below 1024 is taken
        # verbatim — the user knows what they want.
        if gen_width >= 1024 or gen_height >= 1024:
            if backend == "wan":
                gen_width, gen_height = 832, 480
            else:  # ltx
                gen_width, gen_height = 768, 512
            log.info(
                "video: clamped FLUX-sized gen_width/gen_height to %dx%d for backend=%s",
                gen_width,
                gen_height,
                backend,
            )

        ref_image = None
        if subtype == "i2v":
            if not t.input_path:
                raise ValueError("video i2v requires a reference image (uploaded as 'image')")
            ref_image = Image.open(t.input_path)
            ref_image = ImageOps.exif_transpose(ref_image)

        # Video pipelines are the heaviest thing we run; release the
        # FluxEditor's intermediate buffers first to free VRAM. Same
        # pattern as ControlNet / PuLID / IPA dispatch.
        self.editor._release_intermediate_memory()

        # Per-step progress so the UI shows a moving bar instead of
        # freezing for minutes. We don't currently hook diffusers'
        # callback for video (subtle differences across pipelines), so
        # this is start-then-finish only — the GPU stats card still
        # updates from /api/progress's gpu side.
        job_progress.start(total=steps, mode="video")
        try:
            req = VideoRequest(
                backend=backend,
                subtype=subtype,
                prompt=t.prompt,
                ref_image=ref_image,
                num_frames=num_frames,
                fps=fps,
                width=gen_width,
                height=gen_height,
                steps=steps,
                guidance=guidance,
                seed=v.seed,
            )
            out_bytes = self.video.generate(req)
            job_progress.finish()
        except Exception as exc:
            job_progress.finish(error=str(exc))
            raise
        return out_bytes

    def _render(self, t: Task, v: Variant) -> bytes:  # pragma: no cover — GPU inference path
        params = t.params or {}
        max_edge = int(params.get("max_edge", 1024))
        steps = int(params.get("steps", 28))
        guidance = float(params.get("guidance", 3.5))
        use_accel = bool(params.get("use_accel", True))
        sharpen_level = params.get("sharpen_level", "off")
        gen_width = int(params.get("gen_width", 1024))
        gen_height = int(params.get("gen_height", 1024))
        style_lora_id = params.get("style_lora_id") or None
        style_lora_scale = float(params.get("style_lora_scale", 1.0))
        upscale_mode = params.get("upscale", "lanczos")
        # IP-Adapter is its own pipeline path (mode="ipa", see
        # _render_ipa) — no overlay on Kontext / schnell anymore.

        # Resolve input: prefer parent_variant_id (pipeline mid-step) over
        # input_path (initial upload). Mid-pipeline tasks have input_path=NULL
        # because their input is the previous step's output.
        input_source_path = self._resolve_input_path(t)

        img = None
        original_size = None
        if t.mode != "generate" and input_source_path:
            img = Image.open(input_source_path)
            img = ImageOps.exif_transpose(img)
            # If we got the input from a parent variant, use the variant's
            # dimensions (which already match the previous step's input).
            if t.original_width and t.original_height:
                original_size = (t.original_width, t.original_height)
            else:
                original_size = img.size
            if t.mode in ("kontext", "inpaint"):
                img = fit_to_flux_bucket(img, max_edge)
            else:
                img = fit_long_edge(img, max_edge)

        mask = None
        if t.mode == "inpaint" and t.mask_path:
            mask = Image.open(t.mask_path)

        # Schnell defaults to 4 steps; anything else clamps to a sane floor.
        run_steps = max(1, steps)
        job_progress.start(total=run_steps, mode=t.mode)
        try:
            req = EditRequest(
                mode=t.mode,
                prompt=t.prompt,
                image=img,
                mask=mask,
                steps=run_steps,
                guidance=guidance,
                seed=v.seed,
                width=gen_width,
                height=gen_height,
                use_accel=use_accel,
                style_lora_id=style_lora_id,
                style_lora_scale=style_lora_scale,
            )
            out = self.editor.edit(req)
            job_progress.finish()
        except EditAborted:
            job_progress.finish(error="aborted")
            raise
        except Exception as exc:
            job_progress.finish(error=str(exc))
            raise

        # Generate has no original_size — the model output IS the final canvas.
        # For edit modes, restore the user's upload dimensions; choice of
        # resampler depends on the upscale setting:
        #   - "lanczos" (default, legacy): plain LANCZOS interpolation
        #   - "real_esrgan_4x": learned 4x upscale via Real-ESRGAN, then
        #     LANCZOS-downscale to exact target if the original isn't a
        #     clean 4x of the FLUX output. Synthesizes texture detail
        #     instead of blurring.
        if original_size is not None and out.size != original_size:
            if upscale_mode == "real_esrgan_4x" and self.upscaler is not None:
                out = self.upscaler.upscale_to(out, original_size)
            else:
                out = out.resize(original_size, Image.Resampling.LANCZOS)
        # Sharpen still runs on top; Real-ESRGAN already produces sharp
        # output but the user may want to push contrast further.
        out = sharpen(out, sharpen_level)
        return image_to_png_bytes(out)


# Path helpers re-exported for convenience.
__all__ = ["TaskWorker"]
