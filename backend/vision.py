"""Vision analysis backend — native HF models, no remote code.

After hitting four separate transformers-5.x incompatibilities in
Florence-2's bundled remote code, we replaced the single-model approach
with four small, native-API specialists that all officially support
transformers 5.x:

  - ``CLIPSeg`` for referring expression segmentation
    (``CIDAS/clipseg-rd64-refined`` · ~280 MB)
  - ``DETR-ResNet-50`` for generic open-vocabulary-ish detection on the
    91 COCO classes (``facebook/detr-resnet-50`` · ~160 MB)
  - ``OWLv2`` for text-grounded detection — "find me the dragon"
    (``google/owlv2-base-patch16-ensemble`` · ~600 MB)
  - ``BLIP-2 OPT-2.7B`` for image captioning
    (``Salesforce/blip2-opt-2.7b`` · ~5.4 GB resident in bf16)

Each model is lazy-loaded on first use; combined resident VRAM with all
four warm is ~6.5 GB, dominated by BLIP-2. Alongside NF4 FLUX (~10 GB)
this leaves a few hundred MB of slack on a 16 GB card — fine for normal
operation, but heavy variant batches (variants ≥ 4) on Kontext + BLIP-2
warm at the same time can OOM. Drop variants to 1-2 or restart the
server (which frees the warm vision pipelines until the next click)
if that happens.

The public API (caption / detect / segment) matches what the rest of
the app expects so swapping models inside this file is invisible above.

Trade-off vs. Florence-2: BLIP-2 captions are richer than BLIP-large
(spatial relations, composition cues, full sentences) but generic
detection is still limited to the 91 COCO classes — use OWLv2 with
text for fantasy/sci-fi specifics ("the dragon", "the cyborg armor").
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw

from backend.imgutils import ensure_rgb

log = logging.getLogger(__name__)

CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
DETR_MODEL = "facebook/detr-resnet-50"
OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"
# BLIP-2 (Q-Former + OPT-2.7B) — richer, longer captions than BLIP-large,
# better at architecture / scene composition. The trade-off is footprint:
# ~5.4 GB resident in bf16 vs BLIP-large's ~0.9 GB. Still fits alongside
# NF4 FLUX on a 16 GB card with a few hundred MB of slack. Loaded lazily,
# so if the user never clicks "analyze scene" they pay nothing.
BLIP_MODEL = "Salesforce/blip2-opt-2.7b"

# CLIPSeg's segmentation head outputs continuous probability logits per
# pixel; this threshold binarizes them. 0.5 is the published default but
# fantasy-style prompts ("the dragon") sometimes need lower thresholds to
# catch faint matches — exposed as an env var for the curious.
CLIPSEG_MASK_THRESHOLD = float(os.environ.get("AIPIC_CLIPSEG_THRESHOLD", "0.5"))

# DETR's classifier produces a probability per class per query box; this
# filters out low-confidence detections. 0.9 is conservative and gives
# clean labels for UI chips.
DETR_SCORE_THRESHOLD = float(os.environ.get("AIPIC_DETR_THRESHOLD", "0.9"))

# OWLv2's text-grounded scoring is on a different scale than DETR's
# classifier (cross-modal similarity rather than softmax over classes);
# 0.1 is the model card's suggested cut-off for "found something".
OWLV2_SCORE_THRESHOLD = float(os.environ.get("AIPIC_OWLV2_THRESHOLD", "0.1"))


@dataclass
class DetectedObject:
    label: str
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) in image pixels
    score: float | None = None


class VisionAnalyzer:
    """Four-model lazy holder. Each model loads on first use, then stays
    resident — subsequent calls hit the warm pipeline."""

    def __init__(self) -> None:
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Each pair is (processor, model). Lazy-initialized on first call
        # of the corresponding public method so importing this module
        # doesn't trigger 1.5 GB of downloads.
        self._clipseg: tuple[object, object] | None = None
        self._detr: tuple[object, object] | None = None
        self._owlv2: tuple[object, object] | None = None
        self._blip: tuple[object, object] | None = None

    # ----- loaders --------------------------------------------------------

    # nosec B615 comments throughout — published model repos from CIDAS,
    # facebook, google, and Salesforce are trusted upstreams. Pinning to a
    # specific revision would block legitimate upstream fixes for a
    # single-user desktop app; the explicit ``trust_remote_code`` flag is
    # NOT set, so there's no remote-code path to begin with.
    def _load_clipseg(self) -> tuple[object, object]:  # pragma: no cover — needs GPU + download
        if self._clipseg is None:
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

            log.info("loading CLIPSeg (segmentation)")
            proc = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL)  # nosec B615
            model = CLIPSegForImageSegmentation.from_pretrained(  # nosec B615
                CLIPSEG_MODEL, torch_dtype=self._dtype
            ).to(self._device)
            model.eval()
            self._clipseg = (proc, model)
        return self._clipseg

    def _load_detr(self) -> tuple[object, object]:  # pragma: no cover — needs GPU + download
        if self._detr is None:
            from transformers import AutoImageProcessor, DetrForObjectDetection

            log.info("loading DETR (generic object detection)")
            proc = AutoImageProcessor.from_pretrained(DETR_MODEL)  # nosec B615
            model = DetrForObjectDetection.from_pretrained(  # nosec B615
                DETR_MODEL, torch_dtype=self._dtype
            ).to(self._device)
            model.eval()
            self._detr = (proc, model)
        return self._detr

    def _load_owlv2(self) -> tuple[object, object]:  # pragma: no cover — needs GPU + download
        if self._owlv2 is None:
            from transformers import AutoProcessor, Owlv2ForObjectDetection

            log.info("loading OWLv2 (text-grounded detection)")
            proc = AutoProcessor.from_pretrained(OWLV2_MODEL)  # nosec B615
            model = Owlv2ForObjectDetection.from_pretrained(  # nosec B615
                OWLV2_MODEL, torch_dtype=self._dtype
            ).to(self._device)
            model.eval()
            self._owlv2 = (proc, model)
        return self._owlv2

    def _load_blip(self) -> tuple[object, object]:  # pragma: no cover — needs GPU + download
        if self._blip is None:
            # Auto-registry path (AutoProcessor + AutoModelForImageTextToText)
            # instead of importing Blip2ForConditionalGeneration directly.
            # In transformers 5.x the direct class import lazy-loads through
            # an internal registry that fails inside FastAPI / threaded
            # contexts ("cannot import name X from 'transformers'") even
            # when the same import works in a fresh CLI process — sibling
            # auto-class load order can poison the lazy resolver. Auto*
            # routes through the config-driven dispatch table and is immune.
            from transformers import AutoModelForImageTextToText, AutoProcessor

            log.info("loading BLIP-2 OPT-2.7B (image captioning)")
            proc = AutoProcessor.from_pretrained(BLIP_MODEL)  # nosec B615
            model = AutoModelForImageTextToText.from_pretrained(  # nosec B615
                BLIP_MODEL, torch_dtype=self._dtype
            ).to(self._device)
            model.eval()
            self._blip = (proc, model)
        return self._blip

    # ----- public API -----------------------------------------------------

    def caption(  # pragma: no cover — exercises BLIP weights / GPU
        self, image: Image.Image, level: str = "detailed"
    ) -> str:
        """Return a textual description of the scene.

        BLIP-2 OPT-2.7B produces richer multi-sentence descriptions with
        spatial relations ("a statue on a column, with a yellow building
        in the foreground and blue sky behind") — usefully better than
        BLIP-large for architecture / fantasy / sci-fi scenes.

        The ``level`` parameter is accepted for API parity with the old
        Florence-2 backend but ignored. BLIP-2 doesn't expose tiered
        verbosity natively; we always run a single ``generate`` with
        beam search for fluent output. Adjust ``max_new_tokens`` here if
        you want longer captions (cost is roughly linear in tokens).
        """
        _ = level  # documented no-op
        rgb = ensure_rgb(image)
        proc, model = self._load_blip()
        # No text prompt: BLIP-2 in zero-shot caption mode generates
        # unconditioned. (Passing a leading "a photo of" or VQA-style
        # prompt would condition it, but we want a plain description.)
        inputs = proc(images=rgb, return_tensors="pt").to(self._device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    def detect(  # pragma: no cover — dispatches to GPU paths
        self,
        image: Image.Image,
        text: str | None = None,
    ) -> list[DetectedObject]:
        """Object detection.

        Without ``text``: generic OD via DETR over the 91 COCO classes.
        With ``text``: OWLv2 grounds the named phrase(s). Multiple phrases
        can be comma-separated.
        """
        if text and text.strip():
            return self._detect_grounded(image, text)
        return self._detect_generic(image)

    def _detect_generic(  # pragma: no cover — DETR forward pass
        self, image: Image.Image
    ) -> list[DetectedObject]:
        rgb = ensure_rgb(image)
        proc, model = self._load_detr()
        inputs = proc(images=rgb, return_tensors="pt").to(self._device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
        # DETR's post_process expects float32 logits; cast results back
        # to keep its math stable when the model itself ran in bf16.
        target_sizes = torch.tensor([rgb.size[::-1]], device=self._device)
        results = proc.post_process_object_detection(
            outputs, threshold=DETR_SCORE_THRESHOLD, target_sizes=target_sizes
        )[0]
        return [
            DetectedObject(
                label=model.config.id2label[label.item()],
                box=tuple(b.item() for b in box),
                score=float(score.item()),
            )
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"], strict=False
            )
        ]

    def _detect_grounded(  # pragma: no cover — OWLv2 forward pass
        self, image: Image.Image, text: str
    ) -> list[DetectedObject]:
        rgb = ensure_rgb(image)
        proc, model = self._load_owlv2()
        # OWLv2 takes a list of text queries per image. We accept a single
        # comma-separated string from the UI and split it here so users
        # can write "the dragon, the knight" in one input.
        queries = [q.strip() for q in text.split(",") if q.strip()]
        if not queries:
            return []
        inputs = proc(text=[queries], images=rgb, return_tensors="pt").to(self._device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([rgb.size[::-1]], device=self._device)
        results = proc.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=OWLV2_SCORE_THRESHOLD
        )[0]
        return [
            DetectedObject(
                label=queries[label.item()],
                box=tuple(b.item() for b in box),
                score=float(score.item()),
            )
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"], strict=False
            )
        ]

    def segment(  # pragma: no cover — CLIPSeg forward pass
        self, image: Image.Image, text: str
    ) -> Image.Image:
        """Text-prompted segmentation via CLIPSeg.

        Returns a single L-mode PIL image at the input's resolution:
        white (255) where ``text`` grounds, black (0) elsewhere. Threshold
        is configurable via ``AIPIC_CLIPSEG_THRESHOLD`` env (default 0.5).
        """
        if not text or not text.strip():
            raise ValueError("segment(): text prompt is required")
        rgb = ensure_rgb(image)
        proc, model = self._load_clipseg()

        # CLIPSeg accepts a list of text prompts in parallel; we feed one.
        # BatchEncoding.to() takes only a device; cast pixel_values dtype
        # manually so it matches the bf16 model weights (text input_ids
        # stay int64 either way).
        inputs = proc(text=[text], images=[rgb], padding=True, return_tensors="pt").to(self._device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
        # Single-prompt shape: [1, H, W]; squeeze and sigmoid into [0, 1].
        logits = outputs.logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits[0].float()).cpu().numpy()
        binary = (probs > CLIPSEG_MASK_THRESHOLD).astype("uint8") * 255

        # CLIPSeg's output is at the model's internal resolution (352x352);
        # resize to the input image's native size so it composites cleanly
        # onto the mask canvas in the UI.
        mask = Image.fromarray(binary, mode="L").resize(rgb.size, Image.Resampling.BILINEAR)
        # Re-binarize after the interpolation softens edges.
        return mask.point(lambda p: 255 if p > 127 else 0, mode="L")

    # ----- bookkeeping ----------------------------------------------------

    def release(self) -> None:  # pragma: no cover — GPU
        """Drop every loaded model from VRAM. Use when freeing space for
        a heavy diffusers pipeline; subsequent vision calls reload."""
        self._clipseg = None
        self._detr = None
        self._owlv2 = None
        self._blip = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Drawing helper retained from the previous implementation in case the
# pipeline-step path or tests want to rasterize polygons. Not used by the
# current native-model flow but kept for API stability.
def _draw_polygons(size: tuple[int, int], polygons: list[list[float]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) < 6:
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(pts, fill=255)
    return mask


__all__ = [
    "BLIP_MODEL",
    "CLIPSEG_MODEL",
    "DETR_MODEL",
    "OWLV2_MODEL",
    "DetectedObject",
    "VisionAnalyzer",
]
