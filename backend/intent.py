"""Prompt-intent classifier: edit vs generate.

Heuristic based on the first few tokens. Edit verbs in imperative ("remove
the hat", "change her coat") map to instructional models like Flux Kontext.
Creation verbs ("a photo of...", "imagine a...", "draw a...") or noun-led
descriptions map to text-to-image generation models like Flux schnell.

No LLM, no ML — just a keyword table. Cheap, transparent, easy to adjust.
"""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["edit", "generate"]

# Verbs that strongly imply "modify the input image". Compared against the
# prompt's first word case-insensitively.
EDIT_VERBS = frozenset(
    {
        "remove",
        "delete",
        "erase",
        "change",
        "swap",
        "replace",
        "modify",
        "alter",
        "edit",
        "make",
        "turn",
        "convert",
        "recolor",
        "color",
        "colour",
        "add",
        "put",
        "place",
        "insert",
        "blur",
        "sharpen",
        "brighten",
        "darken",
        "enhance",
        "fix",
        "repair",
        "restore",
        "crop",
        "extend",
    }
)

# Verbs/openers that strongly imply "create a new image from scratch".
GENERATE_VERBS = frozenset(
    {
        "create",
        "generate",
        "draw",
        "paint",
        "illustrate",
        "render",
        "design",
        "imagine",
        "depict",
        "show",
        "produce",
        "compose",
        "picture",
    }
)

# Phrases that signal "a [thing]…" style descriptive prompts — typical of
# text-to-image. Anchored at the start of the prompt.
_GENERATE_OPENERS_RE = re.compile(
    r"^\s*(a|an|the)\s+\w+",
    re.IGNORECASE,
)


def classify(prompt: str, has_image: bool) -> Intent:
    """Pick the intent for a prompt.

    Rules, in order:
      1. No image attached: it must be generation.
      2. First word in EDIT_VERBS: edit.
      3. First word in GENERATE_VERBS: generate.
      4. Prompt starts with "a/an/the <noun>": descriptive, generation.
      5. Default with image present: edit (preserve historical behavior).
    """
    if not has_image:
        return "generate"

    stripped = (prompt or "").strip()
    if not stripped:
        return "edit"  # empty prompt with image — handled by validators elsewhere

    first = stripped.split()[0].lower().strip(".,;:!?\"'`")
    if first in EDIT_VERBS:
        return "edit"
    if first in GENERATE_VERBS:
        return "generate"
    if _GENERATE_OPENERS_RE.match(stripped):
        return "generate"
    return "edit"


def explain(prompt: str, has_image: bool) -> dict:
    """Return both the classification and the reason for it — useful for
    surfacing to the UI ('routed to <X> because <Y>')."""
    if not has_image:
        return {"intent": "generate", "reason": "no input image"}

    stripped = (prompt or "").strip()
    if not stripped:
        return {"intent": "edit", "reason": "empty prompt, defaulting to edit"}

    first = stripped.split()[0].lower().strip(".,;:!?\"'`")
    if first in EDIT_VERBS:
        return {"intent": "edit", "reason": f"starts with edit verb '{first}'"}
    if first in GENERATE_VERBS:
        return {"intent": "generate", "reason": f"starts with generation verb '{first}'"}
    if _GENERATE_OPENERS_RE.match(stripped):
        return {"intent": "generate", "reason": "descriptive opener ('a/an/the …')"}
    return {"intent": "edit", "reason": "image present, no clear signal"}
