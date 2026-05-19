# AiPictureModifier · Usage guide

Hands-on reference for both edit modes. Quick rule of thumb: **start with
global edit; reach for inpaint only when you need region precision**.

---

## Global edit (Kontext) — whole-image instruction

UI: select `global edit (Kontext)` (active by default).

**Workflow**
1. Drop / click an image.
2. Write an **instruction**, not a description — Kontext understands
   imperative-mood commands.
3. Hit `edit image`.

**Prompts that work**

| prompt | what it does |
|---|---|
| `remove the hairdryer` | object removal |
| `change the woman's coat to red leather` | attribute swap on a specific object |
| `replace the sky with a dramatic sunset` | region swap |
| `make this a black and white film photograph` | style change |
| `add snow on the roofs and trees` | element addition |
| `in the style of Edward Hopper` | style transfer (known artist names work) |

**Prompts that work poorly**
- `A beautiful photo of …` — descriptive prompts make Kontext generate
  from scratch instead of editing.
- `make it cinematic / better` — too vague, model picks an arbitrary
  interpretation.
- `change X and remove Y and add Z` — chaining three edits in one prompt
  usually executes one cleanly and breaks the others.

**Stabilizers worth adding**
- `… while keeping the rest of the image unchanged`
- `… preserve the original lighting and composition`

Without these, Kontext sometimes changes more than you asked.

**Localize by reference, not coordinates**

Kontext understands object references:
- `the car on the left`
- `the man wearing glasses`
- `the bottle in the foreground`

You don't have to paint a mask — a textual reference works in ~80% of
cases.

---

## Inpaint (Fill) — regenerate only the masked region

UI: select `inpaint (mask)`. A brush row appears.

**Workflow**
1. Drop / click an image.
2. **Paint the region** to rebuild — orange overlay shows on the canvas.
3. Adjust brush size with the `brush` slider (4–120 px).
4. `clear mask` if you mis-painted.
5. Write a prompt describing **what should appear inside the mask** —
   not the whole scene.
6. Hit `edit image`.

**Prompt phrasing — critical difference**

| mode | example |
|---|---|
| Kontext | `replace the hairdryer with a clock on the wall` |
| Fill   | `a clock on the wall` |

Fill already sees the surrounding scene through what's NOT masked. Just
describe the object that goes inside the mask.

**Use cases it nails**
- Object removal: paint the object + prompt with `the wall behind` /
  `empty floor` / `continuation of the scene`
- Object replacement: paint + describe the new object
- Defect repair (out-of-focus face, missing corner): tight mask + describe
  the desired state (`a sharp portrait`, `more sky`)
- Local edits to clothing, hair, accessories — paint the region + describe
  the new state

**Common failure modes**
- **Mask too big** → Fill starts improvising, loses scene context. Keep
  the mask tight around the object with a 5–10 px margin.
- **Mask too small** → undergeneration, visible seams. Better slightly
  over than slightly under.
- **Prompt describes the whole scene** → Fill starts rebuilding
  surroundings it shouldn't touch.

**Brush sizing**
- 40–60 px for most regions
- 12–20 px for precision (face, fingers, small objects)
- **Include shadows + reflections** of objects you're removing. Otherwise
  the cast shadow stays and floats in empty space.

---

## Choosing the mode — fast lookup

| you want to… | use |
|---|---|
| change the whole image style | Kontext |
| remove ONE specific object | **Fill** (precision) or Kontext with textual reference |
| swap background, keep subject | Kontext: `replace the background with …` |
| swap one object for another | **Fill** (region precision) |
| add an element that isn't there | Kontext — Fill has nothing to mask over |
| repair a blurry face | **Fill** with a tight mask |
| recolor an object | Kontext with reference (`the red coat → blue`) |
| guarantee zero change outside the edit | **Fill** — anything outside the mask is preserved exactly |

---

## Parameters that matter

| field | sane range | effect |
|---|---|---|
| `steps` | 20–28 baseline (8 if Hyper-SD LoRA loaded) | more = better + slower; diminishing returns above 28 |
| `guidance` | 2.5–4.0 | lower = more creative, higher = more literal; above 5 burns detail |
| `seed` | any int | **the key knob for iteration** — same seed + new prompt = comparable A/B |

## Disk hygiene

The setup leaks disk in two places: Hugging Face cache (Flux Kontext + Fill
are ~24 GB each, persistent) and torch/triton kernel caches. VRAM also drifts
upward after many sequential edits if you don't release intermediates between
calls — the server already does this automatically via
`torch.cuda.empty_cache() + gc.collect()` in `pipeline._release_intermediate_memory()`,
called from the `finally` block of every edit.

The `scripts/clean.py` helper reports + cleans selectively:

```powershell
# Dry-run report — what's eating disk
python scripts\clean.py

# Drop __pycache__ across the repo
python scripts\clean.py --apply --pyc

# Drop one HF model (frees ~24 GB)
python scripts\clean.py --apply --hf black-forest-labs/FLUX.1-Kontext-dev

# Drop torch.compile + triton caches
python scripts\clean.py --apply --torch-compile

# Drop the local outputs/ folder
python scripts\clean.py --apply --outputs
```

Typical disk costs to be aware of:

| location | size | how often to clean |
|---|---|---|
| `~/.cache/huggingface` Flux Kontext | ~33 GB | only if you stop using global edit |
| `~/.cache/huggingface` Flux Fill | ~33 GB | only if you stop using inpaint |
| `~/.cache/torch_inductor` | ~hundreds of MB | safe to drop, rebuilds on demand |
| `~/.triton` | ~hundreds of MB | safe to drop, rebuilds on demand |
| `<repo>/__pycache__/` | <1 MB total | cosmetic, drop anytime |

If you're switching between bf16 and NF4 a lot, the torch compile cache can
grow because each config has separate compiled kernels. `--torch-compile`
drops everything and the next run rebuilds.

## Iteration workflow that saves time

1. **Exploration pass.** 20 steps, guidance 3.5, seed `random` →
   quick sanity check (~30 s with NF4).
2. **Result 70% of where you want it?** Copy the seed (UI shows it after
   the run), tweak the prompt, keep the same seed and steps. Direct A/B.
3. **Final render.** 28 steps, same seed.

This is ~3× faster than brute-forcing 28 steps on every attempt. With
NF4 active (6.9 s/step), a full A/B/C/D iteration round is ~10 minutes
instead of an hour.
