"""Vision analysis backend — Florence-2 multi-task model.

Florence-2 (microsoft/Florence-2-large, ~770 M params) is a single
encoder-decoder VLM that handles caption / object detection / referring
expression segmentation / OCR via task tokens in the prompt. We use it for
three flows in this app:

  - ``segment(image, text)`` — referring expression segmentation.
    "the dragon in the back" → binary PIL mask snapped to the dragon's
    pixels. Feeds the inpaint pipeline so users don't have to brush masks.
  - ``detect(image, text=None)`` — open-set object detection. Returns a
    list of (label, box, score). With no text → generic OD; with text →
    caption-to-phrase grounding (find specifically what was named).
  - ``caption(image, level)`` — text description of the scene. Used as a
    prompt-writing helper in the UI.

VRAM budget: bf16 ≈ 1.5 GB resident. With NF4 FLUX (~10 GB) we still have
headroom, so we keep Florence-2 GPU-resident. If the budget tightens later,
flip to ``enable_model_cpu_offload()`` — adds ~200 ms first-call latency
per request but recovers ~1 GB.

Loading: ``trust_remote_code=True`` is required — Florence-2 ships custom
modeling code (FlorenceForConditionalGeneration) not yet in the
transformers stable surface. Microsoft is the publisher and the repo is
pinned, so the risk model is the same as any other HF checkpoint we load.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw

from backend.imgutils import ensure_rgb

log = logging.getLogger(__name__)

FLORENCE_MODEL = "microsoft/Florence-2-large"

# Task tokens — these are NOT free-form prompts; Florence-2 was trained on a
# fixed task vocabulary and routes its decoder behavior based on the token.
TASK_CAPTION = "<CAPTION>"
TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
TASK_MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
TASK_OD = "<OD>"  # generic object detection across COCO-style classes
TASK_CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
TASK_REFERRING_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"


@dataclass
class DetectedObject:
    label: str
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) in image pixels
    score: float | None = None  # Florence-2 doesn't expose scores; kept for API parity


class VisionAnalyzer:
    """Lazy holder for Florence-2. The model loads on first use, then stays
    resident — subsequent calls are ~100 ms inference + post-processing."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:  # pragma: no cover — needs GPU + 1.5 GB download
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        log.info("loading Florence-2 (this can take ~30s on first download)")
        # nosec B615 — microsoft is trusted upstream; pinning a revision would
        # block legitimate upstream fixes for a single-user desktop app.
        self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            FLORENCE_MODEL,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(  # nosec B615
            FLORENCE_MODEL,
            trust_remote_code=True,
        )
        self._model.eval()
        log.info("Florence-2 ready on %s (%s)", self._device, self._dtype)

    def _run(self, image: Image.Image, task: str, text_input: str | None = None) -> dict:
        """Single-shot inference. Returns the post-processed Florence-2 dict
        keyed by the task token (e.g. {"<OD>": {"bboxes": [...], "labels": [...]}})."""
        self._load()
        rgb = ensure_rgb(image)
        prompt = task if text_input is None else f"{task}{text_input}"

        inputs = self._processor(text=prompt, images=rgb, return_tensors="pt").to(
            self._device, self._dtype
        )
        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(
            generated_text, task=task, image_size=rgb.size
        )

    def caption(
        self,
        image: Image.Image,
        level: str = "detailed",
    ) -> str:
        """level: 'short' | 'detailed' | 'more_detailed'."""
        task = {
            "short": TASK_CAPTION,
            "detailed": TASK_DETAILED_CAPTION,
            "more_detailed": TASK_MORE_DETAILED_CAPTION,
        }.get(level, TASK_DETAILED_CAPTION)
        out = self._run(image, task)
        return str(out.get(task, "")).strip()

    def detect(
        self,
        image: Image.Image,
        text: str | None = None,
    ) -> list[DetectedObject]:
        """When `text` is None: generic open-vocabulary OD. When provided:
        caption-to-phrase grounding — Florence-2 grounds the named phrase(s)
        in the image and returns boxes for them.
        """
        if text:
            out = self._run(image, TASK_CAPTION_TO_PHRASE_GROUNDING, text_input=text)
            payload = out.get(TASK_CAPTION_TO_PHRASE_GROUNDING, {})
        else:
            out = self._run(image, TASK_OD)
            payload = out.get(TASK_OD, {})
        bboxes = payload.get("bboxes", []) or []
        labels = payload.get("labels", []) or []
        results: list[DetectedObject] = []
        for box, label in zip(bboxes, labels, strict=False):
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            results.append(DetectedObject(label=str(label), box=(x1, y1, x2, y2)))
        return results

    def segment(self, image: Image.Image, text: str) -> Image.Image:
        """Referring expression segmentation. Returns a single L-mode PIL
        image at the input's resolution: white (255) inside the matched
        region, black (0) outside.

        Florence-2 returns polygons (list of [x1, y1, x2, y2, ...]); we
        rasterize them onto a blank canvas. If multiple polygons come back
        (multi-instance match) they're all unioned into one mask — the
        downstream inpaint pipeline expects a single mask layer.
        """
        if not text or not text.strip():
            raise ValueError("segment(): text prompt is required")
        out = self._run(image, TASK_REFERRING_SEGMENTATION, text_input=text)
        payload = out.get(TASK_REFERRING_SEGMENTATION, {})
        polygons_per_instance = payload.get("polygons", []) or []

        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for instance in polygons_per_instance:
            # `instance` is itself a list of polygons (Florence-2's API allows
            # a single label to come back as multiple disconnected regions).
            for poly in instance:
                if len(poly) < 6:  # need ≥ 3 points = 6 coords for a polygon
                    continue
                pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
                draw.polygon(pts, fill=255)
        return mask

    def release(self) -> None:  # pragma: no cover — GPU
        """Drop the model from VRAM. Useful before loading another heavy
        component if the budget gets tight at runtime."""
        if self._model is not None:
            del self._model
            self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["FLORENCE_MODEL", "DetectedObject", "VisionAnalyzer"]
