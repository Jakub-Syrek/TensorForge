# TensorForge · Usage guide

Hands-on reference for every mode + tool. Quick rule of thumb:

- **Start with `auto`** — server picks Kontext (you have an image) or
  schnell (you don't) from the prompt's first word.
- **Reach for `inpaint` / `outpaint` / `control` / `pulid`** only when
  you need region precision, canvas extension, structural conditioning,
  or face preservation that auto-routing won't pick for you.

---

## Mode picker — fast lookup

| You want to… | Use | Why |
|---|---|---|
| Change the whole image style | `Kontext` | follows imperative instructions |
| Generate from scratch | `generate` | text-to-image, 4 steps |
| Remove ONE specific object | `inpaint` (region precision) or `Kontext` with textual reference | mask = exact control |
| Swap background, keep subject | `Kontext: "replace the background with …"` | global edit handles whole-frame swap |
| Swap one object for another | `inpaint` | region precision |
| Add an element that isn't there | `Kontext` — Fill has nothing to mask over | additive edit |
| Repair a blurry face | `inpaint` with a tight mask | local rebuild |
| Recolor a specific object | `Kontext` with reference (`the red coat → blue`) | targeted attribute swap |
| Guarantee zero change outside the edit | `inpaint` | mask defines boundary |
| Extend the canvas | `outpaint` | reuses Fill with auto-generated edge mask |
| Generation following a pose / depth / sketch | `control` | ControlNet Union-Pro |
| "Make me look like X" (style transfer from an image) | normal mode + **ref image** (IP-Adapter) | image-as-prompt |
| "Me in a cyberpunk setting" / face must stay recognizable | `PuLID` | InsightFace + ID encoder |

---

## Kontext (global edit) — whole-image instruction

UI: select `Kontext` (or leave `auto` with an image uploaded).

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

**Localize by reference, not coordinates** — Kontext understands object
references: `the car on the left`, `the man wearing glasses`, `the
bottle in the foreground`. You don't have to paint a mask in ~80% of
cases.

---

## Inpaint (Fill) — regenerate only the masked region

UI: select `inpaint`. Brush row + auto-mask row appear.

**Workflow**
1. Drop an image.
2. **Paint the region** (orange overlay) — adjust brush 4-120 px, `clear mask` to redo.
3. *Or* type a phrase in **describe what to mask** + click **auto-mask** —
   CLIPSeg segments the phrase server-side and paints the mask for you;
   refine with brush if needed.
4. Write a prompt describing **what should appear inside the mask** — not
   the whole scene.
5. Hit `edit image`.

**Prompt phrasing — critical difference**

| mode | example |
|---|---|
| Kontext | `replace the hairdryer with a clock on the wall` |
| Fill   | `a clock on the wall` |

Fill already sees the surrounding scene through what's NOT masked. Just
describe the object inside the mask.

**Use cases it nails**
- Object removal: paint the object + prompt `the wall behind` / `empty
  floor` / `continuation of the scene`.
- Object replacement: paint + describe the new object.
- Defect repair (out-of-focus face, missing corner): tight mask + describe
  desired state.
- Local edits to clothing, hair, accessories.

**Common failure modes**
- **Mask too big** → Fill improvises, loses scene context. Keep tight,
  5-10 px margin around the object.
- **Mask too small** → undergeneration, visible seams. Slightly over is
  safer than slightly under.
- **Prompt describes the whole scene** → Fill rebuilds surroundings it
  shouldn't touch.
- **Cast shadows of removed objects** stay floating — include shadows /
  reflections in the mask.

---

## Outpaint — extend the canvas

UI: select `outpaint`. Outpaint row appears with 4 padding inputs
(top / right / bottom / left in pixels, step 32, default 256).

**Workflow**
1. Upload image.
2. Set how many pixels to add on each side (any combination, ≥1 must
   be > 0).
3. Prompt describes what should fill the new area.
4. `edit image`.

**Under the hood:** frontend builds an extended canvas with the original
at the chosen offset (black-filled pad), auto-generates a binary mask
white over the pad, sends both as `image` + `mask` with `mode=inpaint`.
Server has no special outpaint code — pure UI desugaring.

**Tips**
- "extend the sky upward, dramatic clouds, golden hour" — works great
  for landscapes.
- For symmetrical extension (all four sides), use 256 px each.
- Above ~512 px per side, Fill starts to invent more than continue;
  break into two passes (`top=256` then `top=256` again) for clean
  long extensions.

---

## ControlNet — structural conditioning

UI: select `control`. Control row appears with `control_type` selector
(canny / depth / pose) and `scale` (default 0.7).

**Critical:** the uploaded image must already be **the pre-processed map**,
not a regular photo. Use one of:

- [controlnet_aux](https://github.com/huggingface/controlnet_aux) Python
  package: `from controlnet_aux import CannyDetector, MidasDetector,
  OpenposeDetector`
- ComfyUI preprocessor nodes (if you have it installed)
- OpenCV: `cv2.Canny(image, 100, 200)` for canny edges
- Online tools: search for "canny edge generator" / "depth map estimator"

**Workflow**
1. Generate the conditioning map externally.
2. Upload that map as the input image.
3. Pick `control_type` matching what you uploaded.
4. Prompt: describe the **scene** (style, subject, atmosphere) — the
   map handles composition.
5. `edit image`.

**Scale guide**
- `0.4` — loose suggestion, prompt dominates
- `0.7` — default, balanced
- `1.0` — tight adherence, geometry is sacred
- `1.3+` — over-conditioned, output flattens

**Cost:** first run downloads FLUX.1-dev (~24 GB) + Union-Pro (~6 GB).
Subsequent runs reuse cache. ~30 s VRAM swap before each control task
(releases warm Kontext / Fill).

---

## IP-Adapter — image-as-prompt

UI: in any mode that accepts an image-conditioned generation (Kontext,
generate), use the **ref image** dropzone above the style LoRA picker.

**Workflow**
1. Drop a reference image (painting, photo, concept art) into the **ref
   image** slot.
2. Adjust weight (0.5-0.8 typical, 1.0 = strong).
3. Generate / edit normally — the reference biases style, color, and
   composition without you describing it verbally.
4. Click `clear` to remove the reference.

**Stack with style LoRA + ControlNet** for three independent layers:

- IP-Adapter — visual reference (style, color, mood)
- Style LoRA — semantic style (Ghibsky, Tarot, Anime)
- ControlNet — geometry (pose, depth)

All three coexist on the same pipe; weights are independent.

**Cost:** first request that carries an `ip_image` downloads ~1 GB
adapter weights. Once warm, toggle on/off per-request via the scale slider
(scale=0 disables without unloading).

---

## PuLID — face identity preservation

UI: select `PuLID`. PuLID row appears with `id_strength` input (default 1.0).

**Workflow**
1. Upload a **clear, front-facing face photo** as the input image. Higher
   resolution = better embedding.
2. Prompt describes the **scene** (style, setting, costume) — not the
   face.
3. Adjust `id_strength`:
   - `0.7` — strong stylization with recognizable face
   - `1.0` — default balance
   - `1.2` — faithful face, lighter style
4. `edit image`.

**Cost:** first run downloads PuLID weights (~700 MB) + InsightFace
(~280 MB). FLUX.1-dev base shared with ControlNet (~24 GB on disk).
~7 GB VRAM under NF4. Worker releases Kontext warm state before
loading.

**Requirements:** `pip install insightface` (added to requirements.txt).
If insightface isn't installed, PuLID raises an `ImportError` with
install instructions.

---

## Tools (anywhere)

These don't change the mode — they're side actions on the loaded image
or the prompt textarea.

### Fast analyze / Deep analyze (scene description)

After uploading an image, the **fast analyze** / **deep analyze** buttons
appear. Click:

- **fast analyze** — BLIP-large caption + DETR object chips. ~1 s warm.
- **deep analyze** — BLIP-2 OPT-2.7B caption (multi-sentence, spatial
  relations) + same chips. Slower, ~10 GB download first time.

Each detected object becomes a clickable chip. Click → label inserts
into the prompt textarea at cursor. Useful for "replace the X with Y"
edits where you don't know the model's preferred vocabulary.

### Remove bg

Same row as analyze. Click → BiRefNet (via rembg, on CPU) isolates the
subject. Result loads as the new input image with a transparent
background — feed it to FLUX Fill with a backdrop prompt to swap scenes.

### ✨ expand (prompt rewriter)

Floats on the top-right of the prompt textarea. Click → Qwen2.5-1.5B
rewrites your short prompt into a verbose FLUX-style caption. Intent
(generate vs edit) is inferred from the current mode.

**Ctrl+Z** in the textarea brings the original short prompt back.

### Auto-mask (in inpaint mode)

Below the brush, an **describe what to mask** input + auto-mask button.
Type a phrase ("the sky", "the dragon"), click → CLIPSeg paints the mask
for you. Refine with brush if needed.

---

## Pipeline mode — sequential chains

Toggle **pipeline mode** under the prompt textarea. Each non-empty line
becomes a separate step. Steps run sequentially, each step's output
becomes the next step's input.

### Syntax

```
<prompt>                       — mode=auto, no style override
[<mode>] <prompt>              — force mode for this step
[<mode>|<lora>] <prompt>       — mode + style LoRA override
[|<lora>] <prompt>             — auto mode + style LoRA
[auto_mask] <phrase>           — CLIPSeg segments <phrase>, passes mask
                                 to the NEXT step (typically inpaint)
```

### Example chains

**Generate → restyle:**
```
[generate|tarot] heroic knight in stormy mountain pass
[kontext|realism] cinematic lighting, anamorphic lens flare
```

**Generate → segment → repaint:**
```
[generate|tarot] knight in stormy mountain pass
[auto_mask] the knight
[inpaint] in obsidian cyborg armor with neon glow
```

**Edit → upscale via control reference** (manual two-step since outpaint
isn't pipeline-addressable yet):
```
[kontext] make it cyberpunk at night
[|tarot] enhance contrast, neon spill
```

### Limits

- 10 steps max per pipeline (server-side cap).
- Pipeline is **linear** — no branching. For variants, run multiple
  pipelines.
- Mid-step variants are NOT generated; each step produces one image.
- `auto_mask` only feeds an `inpaint` step downstream. If next step is
  `kontext`, the mask is set but ignored.

---

## LoRA stacking (advanced)

Internally, FluxEditor manages **three concurrent adapter layers** per
pipeline:

1. **Acceleration LoRA** (Hyper-SD / Flux-Turbo) — toggle via `acceleration
   LoRA · 8 steps` checkbox. When on: 8 steps gives near-28-step quality
   at ~3× speed.
2. **Style LoRA** — picked from the dropdown, scaled by the weight input.
3. **IP-Adapter** — separately loaded; scale controls bias.

All three coexist via diffusers' multi-adapter API. Stacking advice:

- Accel + Style: works great. Use 8 steps with the accel LoRA + a
  style LoRA at 0.8-1.0.
- Accel + IP-Adapter: also works. IP-Adapter at 0.7 is a typical sweet
  spot.
- Style + IP-Adapter: both push the aesthetic — be careful, can
  conflict. Drop style LoRA to 0.6 if you're also using IP-Adapter.
- All three: possible but easy to over-condition. Start with all
  scales at 0.5-0.6 and creep up.

---

## Parameters that matter

| field | sane range | effect |
|---|---|---|
| `steps` | 20-28 baseline (8 if Hyper-SD LoRA loaded; 4 for schnell) | more = better + slower; diminishing returns above 28 |
| `guidance` | 2.5-4.0 | lower = more creative, higher = more literal; above 5 burns detail |
| `seed` | any int | **the key knob for iteration** — same seed + new prompt = comparable A/B |
| `max edge` | 512-1024 (NF4); 384-512 (bf16) | server-side downscale before inference |
| `variants` | 1-8 | how many seeds to run sequentially |
| `sharpen` | off / light / medium / strong | PIL UnsharpMask after LANCZOS upscale |
| `upscale` | `lanczos` / `real_esrgan_4x` | resampler for the round-trip back to upload dims |
| `ip_scale` (IP-Adapter) | 0.4-1.0 | strength of the visual reference |
| `control_scale` (ControlNet) | 0.4-1.0 | strength of the conditioning map |
| `id_strength` (PuLID) | 0.6-1.2 | strength of face identity vs prompt style |

---

## Iteration workflow that saves time

1. **Exploration pass.** 20 steps, guidance 3.5, seed `random` → quick
   sanity check (~30 s with NF4).
2. **Result 70% of where you want it?** Copy the seed (UI shows it after
   the run), tweak the prompt, keep the same seed and steps. Direct A/B.
3. **Final render.** 28 steps, same seed, optionally `real-esrgan 4x`
   upscale for the output.

This is ~3× faster than brute-forcing 28 steps on every attempt. With
NF4 active (6.9 s/step), a full A/B/C/D iteration round is ~10 minutes
instead of an hour.

For multi-LoRA tuning, fix the seed first, then sweep one knob at a
time (style LoRA weight, IP scale, control scale). Changing two at
once makes the responsible variable impossible to identify.

---

## Disk hygiene

The setup leaks disk in three places:

- **Hugging Face cache** — FLUX Kontext (~24 GB), Fill (~24 GB),
  schnell (~13 GB), Qwen-Edit (~20 GB), FLUX.1-dev (~24 GB if you use
  ControlNet or PuLID), Union-Pro (~6 GB), PuLID (~700 MB), BLIP-large
  (~470 MB), BLIP-2 (~10 GB if you tap deep analyze), DETR/OWLv2/CLIPSeg
  (~1.5 GB combined), Qwen2.5-1.5B (~3 GB), Real-ESRGAN (~67 MB).
- **InsightFace bundle** — `~/.insightface/models/buffalo_l/` ~280 MB.
- **rembg models** — `~/.u2net/` ~150 MB (BiRefNet).
- **Torch / Triton kernel caches** — hundreds of MB, rebuilt on demand
  if cleared.

**Active VRAM** also drifts upward after many sequential edits if you
don't release intermediates between calls — the server does this
automatically via `torch.cuda.empty_cache() + gc.collect()` in
`pipeline._release_intermediate_memory()`, called from the `finally`
block of every edit.

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

If you're swapping between bf16 and NF4 a lot, the torch compile cache
grows because each config has separate compiled kernels. `--torch-compile`
drops everything; next run rebuilds.
