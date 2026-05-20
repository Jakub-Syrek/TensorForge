"""Tests for backend.intent — prompt-intent classifier."""

from __future__ import annotations

import pytest

from backend.intent import classify, explain


@pytest.mark.parametrize(
    "prompt",
    [
        "remove the hat",
        "Remove the glasses",
        "REPLACE the sky with sunset",
        "change her coat to red",
        "make the background dark",
        "add a clock on the wall",
        "blur the background",
    ],
)
def test_edit_verbs_with_image(prompt):
    assert classify(prompt, has_image=True) == "edit"


@pytest.mark.parametrize(
    "prompt",
    [
        "create a landscape with mountains",
        "Generate a cat sitting on a chair",
        "draw a Renaissance portrait",
        "paint a sunset over the ocean",
        "imagine a futuristic city at night",
        "render a 3D model of a robot",
    ],
)
def test_generate_verbs_with_image(prompt):
    assert classify(prompt, has_image=True) == "generate"


@pytest.mark.parametrize(
    "prompt",
    [
        "a cat on a chair",
        "an abandoned warehouse at sunset",
        "The Eiffel Tower under storm clouds",
    ],
)
def test_descriptive_openers_with_image_classify_as_generate(prompt):
    assert classify(prompt, has_image=True) == "generate"


def test_no_image_always_generate():
    assert classify("remove the hat", has_image=False) == "generate"
    assert classify("anything at all", has_image=False) == "generate"
    assert classify("", has_image=False) == "generate"


def test_empty_prompt_with_image_defaults_to_edit():
    assert classify("", has_image=True) == "edit"
    assert classify("   ", has_image=True) == "edit"


def test_ambiguous_with_image_defaults_to_edit():
    # No verb, not a descriptive opener — fall through to 'edit' as safer
    # default when we have something to edit.
    assert classify("happier mood", has_image=True) == "edit"
    assert classify("vintage style", has_image=True) == "edit"


def test_punctuation_after_first_word_is_stripped():
    assert classify("remove, the hat", has_image=True) == "edit"
    assert classify("Generate: a cat", has_image=True) == "generate"
    assert classify('"draw" a tree', has_image=True) == "generate"


def test_explain_returns_intent_and_reason():
    out = explain("remove the hat", has_image=True)
    assert out["intent"] == "edit"
    assert "edit verb" in out["reason"]

    out = explain("a cat", has_image=True)
    assert out["intent"] == "generate"
    assert "opener" in out["reason"]

    out = explain("draw a tree", has_image=False)
    assert out["intent"] == "generate"
    assert "no input image" in out["reason"]

    out = explain("colorful mood", has_image=True)
    assert out["intent"] == "edit"
    assert "no clear signal" in out["reason"]


def test_explain_empty_prompt():
    out = explain("", has_image=True)
    assert out["intent"] == "edit"
    assert "empty" in out["reason"]
