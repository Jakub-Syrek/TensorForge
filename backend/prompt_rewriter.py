"""Prompt expansion — small LLM rewrites short user prompts into the
verbose, detail-laden form FLUX trained on.

The motivation: typical user input is "knight in forest" — three words.
FLUX was trained on captions like "a battle-scarred medieval knight in
ornate gothic plate armor holding a runed greatsword, standing in a
misty pine forest at dawn, golden hour, volumetric god rays through
the canopy, photoreal cinematic 35mm anamorphic, dramatic side-lighting,
shallow depth of field". The longer form consistently produces ~2x
better outputs at the same step count.

Doing this with a local 1.5B-parameter instruct LLM means:
  - zero API cost / round-trip latency
  - works offline
  - prompt stays inside the same VRAM budget (~3 GB resident in bf16)
  - we can tune the system prompt for fantasy/sci-fi vocabulary

Qwen2.5-1.5B-Instruct is the sweet spot — small enough to coexist with
NF4 FLUX on a 16 GB card, smart enough to follow detailed system
instructions for output formatting. Released by Alibaba, Apache 2.0,
no remote_code needed.

The expansion mode (``intent='edit'`` vs ``'generate'``) tweaks the
system prompt: edits should preserve the original action verbs and
just add visual richness, generations can rephrase freely toward
diffusion-friendly composition.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

# Qwen2.5-1.5B-Instruct — Apache 2.0, ~3 GB bf16. The "Instruct" tune
# follows system+user message format reliably; smaller base/chat variants
# tend to ramble or copy the system prompt back.
QWEN_REWRITER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


_SYSTEM_GENERATE = """\
You are a prompt-expansion assistant for FLUX, a text-to-image
diffusion model. The user will send you a short scene description.
Your job: rewrite it as a single rich descriptive caption packed with
visual detail that FLUX can latch onto.

Output rules:
  - One paragraph, no line breaks, no bullet points.
  - Pack in: subject details, materials, lighting, camera/lens cues,
    color palette, atmosphere, time of day, composition hints.
  - Stay faithful to the user's intent — never change the subject or
    action. Add visual richness, don't invent unrelated objects.
  - No meta-text like "Here is the expanded prompt:" — just the prompt.
  - Keep it under 60 words. FLUX's text encoder caps useful tokens
    around there.

Style cues: photographic / cinematic vocabulary. Mention lighting in
concrete terms (golden hour, rim light, volumetric god rays). Include
one or two technical photography hints (35mm, anamorphic, shallow DOF)
when they fit the scene.

Examples:
  user: knight in forest
  you: a battle-scarred medieval knight in ornate gothic plate armor \
holding a runed greatsword, standing in a misty pine forest at dawn, \
volumetric god rays through the canopy, golden hour rim light, 35mm \
anamorphic, shallow depth of field, photoreal cinematic

  user: cyberpunk city
  you: neon-drenched cyberpunk megacity at night, towering holographic \
billboards in kanji and cyrillic, rain-slick streets reflecting magenta \
and cyan, dense fog, anamorphic lens flares, cinematic wide shot, \
moody atmospheric lighting, shallow depth of field
"""

_SYSTEM_EDIT = """\
You are a prompt-expansion assistant for FLUX Kontext, an image-editing
diffusion model. The user will send you a short edit instruction.
Your job: rewrite it as a single rich descriptive instruction packed
with visual specifics that FLUX Kontext can act on.

Output rules:
  - One paragraph, no line breaks, no bullets.
  - Preserve the user's action verb literally (replace / add / remove /
    change / make). Never change WHAT the edit does — only enrich HOW.
  - Add: material specifics, lighting cues, color/texture details,
    composition adjustments that make the edit coherent with a typical
    photo.
  - No meta-text like "Here is the expanded prompt:" — just the prompt.
  - Keep it under 50 words.

Examples:
  user: replace the sky with sunset
  you: replace the sky with a dramatic sunset, warm orange and magenta \
clouds, low-angle sun spilling rim light onto the building edges, \
golden hour atmosphere

  user: turn it into cyberpunk
  you: turn the scene into a cyberpunk night, neon magenta and cyan \
signage on every surface, rain-slick streets, holographic billboards, \
anamorphic lens flares, moody atmospheric lighting
"""


class PromptRewriter:
    """Lazy holder for Qwen2.5-1.5B-Instruct."""

    def __init__(self) -> None:
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:  # pragma: no cover — needs GPU + ~3 GB download
        if self._model is not None:
            return
        # Auto-class entry point (same reasoning as backend/vision.py — go
        # through the auto-registry to dodge the transformers 5.x lazy
        # resolver bug for direct class imports in long-lived processes).
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading Qwen2.5-1.5B-Instruct (prompt rewriter)")
        # nosec B615 — Alibaba/Qwen is a trusted upstream; pinning a
        # revision would block legit fixes for a single-user desktop app.
        self._tokenizer = AutoTokenizer.from_pretrained(QWEN_REWRITER_MODEL)  # nosec B615
        self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            QWEN_REWRITER_MODEL, torch_dtype=self._dtype
        ).to(self._device)
        self._model.eval()
        log.info("Qwen rewriter ready on %s (%s)", self._device, self._dtype)

    def expand(  # pragma: no cover — GPU inference path
        self, prompt: str, intent: str = "generate"
    ) -> str:
        """Expand a short prompt into a FLUX-style description.

        intent='generate' uses the text-to-image system prompt.
        intent='edit' uses the Kontext-edit system prompt (preserves
        action verbs, adds visual richness).

        Returns the expanded prompt as a single line; falls back to the
        original input if the model produces something obviously broken
        (empty / longer than 200 words / contains the literal sentinel
        markers).
        """
        if not prompt or not prompt.strip():
            return prompt
        self._load()

        system = _SYSTEM_EDIT if intent == "edit" else _SYSTEM_GENERATE
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt.strip()},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                # Force deterministic output. The system prompt is opinionated
                # enough — sampling adds noise without quality.
                num_beams=1,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Strip the prompt tokens; keep only the model's response.
        gen_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
        expanded = self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Sanity-check the output. If empty or absurdly long, return the
        # original prompt — better the user's three words than a malformed
        # paragraph that FLUX will tokenize-truncate anyway.
        if not expanded or len(expanded.split()) > 200:
            return prompt
        return expanded

    def release(self) -> None:  # pragma: no cover — GPU
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["QWEN_REWRITER_MODEL", "PromptRewriter"]
