const $ = (id) => document.getElementById(id);

const state = {
  mode: "kontext",
  image: null,        // HTMLImageElement
  imageFile: null,    // File
  drawing: false,
  brush: 40,
  maskDirty: false,
  lastResultBlob: null,  // last successful /api/edit response (for chaining)
  userAborted: false,
  accelAvailable: false,  // set true when /api/health reports accel config
};

// In-memory history of successful edits. Cleared on page refresh.
const history = [];
const HISTORY_MAX = 20;

const imgCanvas = $("imgCanvas");
const maskCanvas = $("maskCanvas");
const imgCtx = imgCanvas.getContext("2d");
const maskCtx = maskCanvas.getContext("2d");

// --- health check --------------------------------------------------------
fetch("/api/health").then(r => r.json()).then(h => {
  const dev = h.cuda_available ? `${h.device} · sm_${(h.capability || []).join("")}` : "CPU";
  $("health").textContent = `torch ${h.torch} · ${dev}`;
  // Reveal the accel toggle only if the server has a LoRA configured.
  if (h.accel && h.accel.repo) {
    state.accelAvailable = true;
    $("accelRow").classList.remove("hidden");
    $("accelInfo").textContent = `${h.accel.repo.split("/").pop()}`;
  }
}).catch(() => { $("health").textContent = "health: server unreachable"; });

// --- file input + drop ---------------------------------------------------
const dz = $("dropzone");
const file = $("file");
// NB: dropzone is a <label> wrapping the hidden file input, so the click
// reaches the input natively. Don't add a JS click handler that calls
// file.click() — it doubles the dialog (user has to pick the file twice).
file.addEventListener("change", (e) => loadFile(e.target.files[0]));
["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));

function loadFile(f) {
  if (!f) return;
  state.imageFile = f;
  const img = new Image();
  img.onload = () => {
    state.image = img;
    fitCanvases(img.naturalWidth, img.naturalHeight);
    imgCtx.drawImage(img, 0, 0, imgCanvas.width, imgCanvas.height);
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    state.maskDirty = false;
    $("dropHint").textContent = f.name;
  };
  img.src = URL.createObjectURL(f);
}

function fitCanvases(w, h) {
  imgCanvas.width = maskCanvas.width = w;
  imgCanvas.height = maskCanvas.height = h;
}

// --- accel toggle: when on, auto-pin steps to 8 (Hyper-SD default);
// when off, auto-pin to 28 (full quality). User can still override manually.
$("accelToggle").addEventListener("change", (e) => {
  $("steps").value = e.target.checked ? "8" : "28";
});

// --- mode toggle ---------------------------------------------------------
// Per-mode sensible defaults. User can override afterwards via the inputs.
const MODE_DEFAULTS = {
  kontext: { steps: 28, guidance: 3.5 },
  inpaint: { steps: 28, guidance: 3.5 },
  qwen:    { steps: 50, guidance: 4.0 }, // Qwen-Image-Edit native defaults
};

document.querySelectorAll(".seg-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const previousMode = state.mode;
    state.mode = btn.dataset.mode;
    $("brushRow").classList.toggle("hidden", state.mode !== "inpaint");
    maskCanvas.style.pointerEvents = state.mode === "inpaint" ? "auto" : "none";
    // Auto-adjust params only when switching to a mode with materially
    // different defaults (Qwen needs ~50 steps; Flux modes need ~28).
    if (state.mode !== previousMode && MODE_DEFAULTS[state.mode]) {
      $("steps").value = MODE_DEFAULTS[state.mode].steps;
      $("guidance").value = MODE_DEFAULTS[state.mode].guidance;
    }
  });
});
maskCanvas.style.pointerEvents = "none";

// --- mask drawing --------------------------------------------------------
$("brushSize").addEventListener("input", (e) => { state.brush = +e.target.value; });
$("clearMask").addEventListener("click", () => {
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  state.maskDirty = false;
});

function canvasPos(evt) {
  const r = maskCanvas.getBoundingClientRect();
  const sx = maskCanvas.width / r.width;
  const sy = maskCanvas.height / r.height;
  return { x: (evt.clientX - r.left) * sx, y: (evt.clientY - r.top) * sy };
}

maskCanvas.addEventListener("pointerdown", (e) => {
  if (state.mode !== "inpaint") return;
  state.drawing = true;
  maskCanvas.setPointerCapture(e.pointerId);
  paintAt(canvasPos(e));
});
maskCanvas.addEventListener("pointermove", (e) => {
  if (!state.drawing) return;
  paintAt(canvasPos(e));
});
maskCanvas.addEventListener("pointerup", () => { state.drawing = false; });

function paintAt(p) {
  const r = state.brush * (maskCanvas.width / maskCanvas.getBoundingClientRect().width) / 2;
  maskCtx.fillStyle = "rgba(255, 107, 26, 1)";
  maskCtx.beginPath();
  maskCtx.arc(p.x, p.y, r, 0, Math.PI * 2);
  maskCtx.fill();
  state.maskDirty = true;
}

// --- run -----------------------------------------------------------------
$("run").addEventListener("click", runEdit);

async function runEdit() {
  if (!state.imageFile) return setStatus("upload an image first", "err");
  const prompt = $("prompt").value.trim();
  if (!prompt) return setStatus("prompt is empty", "err");
  if (state.mode === "inpaint" && !state.maskDirty) return setStatus("paint a mask first", "err");

  const fd = new FormData();
  fd.append("image", state.imageFile);
  fd.append("mode", state.mode);
  fd.append("prompt", prompt);
  fd.append("steps", $("steps").value);
  fd.append("guidance", $("guidance").value);
  if ($("seed").value) fd.append("seed", $("seed").value);
  // use_accel: send the checkbox state IF the toggle is visible (accel
  // configured server-side). Otherwise omit — backend defaults to True
  // which is a no-op when no LoRA is loaded.
  if (state.accelAvailable) {
    fd.append("use_accel", $("accelToggle").checked ? "true" : "false");
  }
  if (state.mode === "inpaint") fd.append("mask", await maskBlob(), "mask.png");

  $("run").disabled = true;
  $("abort").classList.remove("hidden");
  state.userAborted = false;
  setStatus("running…", "");
  // Switch the idle poller to the active (faster) cadence for this job.
  startProgressPoll(500);
  try {
    const r = await fetch("/api/edit", { method: "POST", body: fd });
    if (!r.ok) {
      // 499 (NGINX): client closed request — we sent abort, expected case.
      if (r.status === 499 || state.userAborted) {
        setStatus("aborted", "err");
        return;
      }
      throw new Error((await r.text()) || r.statusText);
    }
    const usedSeed = r.headers.get("X-Used-Seed");
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    $("result").src = url;
    state.lastResultBlob = blob;
    $("download").href = url;
    $("resultActions").classList.remove("hidden");
    if (usedSeed) {
      // Pin the seed used by this run so the next click reuses it —
      // gives free A/B testing at fixed seed when iterating on prompt.
      // User can clear the field to randomize again.
      $("seed").value = usedSeed;
      setStatus(`done · seed ${usedSeed}`, "ok");
    } else {
      setStatus("done", "ok");
    }
    addHistoryEntry({
      resultBlob: blob,
      inputFile: state.imageFile,
      prompt,
      seed: usedSeed,
      mode: state.mode,
      steps: $("steps").value,
      guidance: $("guidance").value,
      useAccel: state.accelAvailable ? $("accelToggle").checked : null,
    });
  } catch (e) {
    if (state.userAborted) {
      setStatus("aborted", "err");
    } else {
      setStatus(String(e.message || e), "err");
    }
  } finally {
    $("run").disabled = false;
    $("abort").classList.add("hidden");
    // Drop back to the slow idle cadence; don't stop completely so GPU
    // stats stay live while the user inspects the result.
    startProgressPoll(2000);
  }
}

// --- abort --------------------------------------------------------------
$("abort").addEventListener("click", async () => {
  $("abort").disabled = true;
  setStatus("aborting…", "");
  state.userAborted = true;
  try {
    const r = await fetch("/api/abort", { method: "POST" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok && r.status !== 409) {
      setStatus(`abort failed: ${r.status}`, "err");
    }
    // 200: aborted at step boundary. 409: nothing to abort (race — job
    // already finished). Either way, runEdit's response handler will
    // reflect the final state.
    void body;
  } finally {
    $("abort").disabled = false;
  }
});

// --- refresh / reset ----------------------------------------------------
$("refresh").addEventListener("click", () => {
  // Clear image, mask, prompt, result, status — back to a clean slate
  // without reloading the page (so GPU panel stays live).
  state.image = null;
  state.imageFile = null;
  state.maskDirty = false;
  state.userAborted = false;
  $("file").value = "";
  $("dropHint").textContent = "drop image · or click to choose";
  $("prompt").value = "";
  $("seed").value = "";
  imgCtx.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  imgCanvas.width = maskCanvas.width = 0;
  imgCanvas.height = maskCanvas.height = 0;
  $("result").removeAttribute("src");
  $("resultActions").classList.add("hidden");
  state.lastResultBlob = null;
  setStatus("", "");
});

// --- history: ring buffer of last HISTORY_MAX successful edits ---------
function addHistoryEntry(entry) {
  const id = Date.now();
  const resultUrl = URL.createObjectURL(entry.resultBlob);
  // Store a thumbnail-ish version by reusing the full blob URL — small
  // enough for in-memory storage of ~20 entries, and lets us treat the
  // history thumb as the canonical reference if user wants to re-load it.
  const item = { id, resultUrl, ...entry };
  history.unshift(item);
  while (history.length > HISTORY_MAX) {
    const dropped = history.pop();
    URL.revokeObjectURL(dropped.resultUrl);
  }
  renderHistory();
}

function renderHistory() {
  const strip = $("historyStrip");
  const row = $("historyRow");
  strip.innerHTML = "";
  if (history.length === 0) {
    row.classList.add("hidden");
    return;
  }
  row.classList.remove("hidden");
  for (const item of history) {
    const el = document.createElement("div");
    el.className = "history-item";
    el.style.backgroundImage = `url(${item.resultUrl})`;
    el.title = `${item.mode} · seed ${item.seed}\n"${item.prompt}"\nclick to load as input`;
    const badge = document.createElement("span");
    badge.className = "history-mode";
    badge.textContent = item.mode === "kontext" ? "K" : "F";
    el.appendChild(badge);
    el.addEventListener("click", () => loadHistoryEntry(item));
    strip.appendChild(el);
  }
}

function loadHistoryEntry(item) {
  // Re-wrap the stored result blob as a fresh File and feed it through
  // the canonical upload pipeline — exactly the same path as "use as
  // input" but from arbitrary history depth.
  const file = new File([item.resultBlob], `history-${item.id}.png`, { type: "image/png" });
  loadFile(file);
  $("prompt").value = item.prompt || "";
  $("seed").value = item.seed || "";
  if (item.mode === "kontext" || item.mode === "inpaint") {
    document.querySelectorAll(".seg-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === item.mode));
    state.mode = item.mode;
    $("brushRow").classList.toggle("hidden", item.mode !== "inpaint");
    maskCanvas.style.pointerEvents = item.mode === "inpaint" ? "auto" : "none";
  }
  if (item.steps) $("steps").value = item.steps;
  if (item.guidance) $("guidance").value = item.guidance;
  if (state.accelAvailable && item.useAccel !== null) {
    $("accelToggle").checked = item.useAccel;
  }
  // Drop the displayed result so it's clear the next edit starts fresh.
  $("result").removeAttribute("src");
  $("resultActions").classList.add("hidden");
  setStatus(`loaded · seed ${item.seed}`, "ok");
}

$("historyClear").addEventListener("click", () => {
  for (const item of history) URL.revokeObjectURL(item.resultUrl);
  history.length = 0;
  renderHistory();
});

// --- chain: use current result as the next input -----------------------
$("useAsInput").addEventListener("click", () => {
  const blob = state.lastResultBlob;
  if (!blob) return setStatus("no result to chain yet", "err");

  // Wrap the blob as a File so the existing upload pipeline (loadFile)
  // can ingest it identically to a fresh user upload.
  const stamped = `chained-${Date.now()}.png`;
  const file = new File([blob], stamped, { type: "image/png" });

  loadFile(file);
  // Clear UI surface for the next instruction. Keep mode + steps +
  // guidance untouched; clear prompt and seed because (a) the previous
  // prompt likely doesn't make sense on the new image, (b) the same
  // seed on a different input produces a different but pseudo-random
  // result anyway — better to start fresh.
  $("prompt").value = "";
  $("seed").value = "";
  // Drop the displayed result — it's now the input.
  $("result").removeAttribute("src");
  $("resultActions").classList.add("hidden");
  setStatus("chained — write the next instruction", "ok");
});

// --- progress polling ---------------------------------------------------
let progTimer = null;
let progIntervalMs = 0;

function startProgressPoll(intervalMs) {
  if (progTimer && progIntervalMs === intervalMs) return;  // already at that cadence
  if (progTimer) clearInterval(progTimer);
  progIntervalMs = intervalMs;
  pollOnce();  // immediate tick so the panel reflects state without delay
  progTimer = setInterval(pollOnce, intervalMs);
}

function stopProgressPoll() {
  if (progTimer) clearInterval(progTimer);
  progTimer = null;
  progIntervalMs = 0;
}

// Kick off idle polling at page load — confirms backend is alive and keeps
// GPU stats live even before the first edit.
startProgressPoll(2000);

async function pollOnce() {
  try {
    const r = await fetch("/api/progress");
    if (!r.ok) return;
    const { job, gpu } = await r.json();
    renderJob(job);
    renderGpu(gpu);
  } catch { /* ignore transient failures */ }
}

function renderJob(job) {
  const pct = job.percent || 0;
  $("progFill").style.width = `${pct}%`;
  const label = job.total
    ? `step ${job.step}/${job.total} · ${pct.toFixed(0)}%`
    : (job.active ? "warming up…" : "idle");
  $("progLabel").textContent = label;
  if (job.elapsed_s && job.total && job.step > 0) {
    const perStep = job.elapsed_s / job.step;
    const remaining = perStep * (job.total - job.step);
    $("progEta").textContent =
      `${job.elapsed_s.toFixed(1)}s elapsed · ~${remaining.toFixed(0)}s left · ${perStep.toFixed(1)}s/step`;
  } else if (job.elapsed_s) {
    $("progEta").textContent = `${job.elapsed_s.toFixed(1)}s elapsed`;
  } else {
    $("progEta").textContent = "";
  }
}

function renderGpu(gpu) {
  if (!gpu) {
    $("gpuName").textContent = "GPU (nvidia-smi unavailable)";
    return;
  }
  $("gpuName").textContent = gpu.name || "GPU";
  $("gpuUtilFill").style.width = `${gpu.util_percent}%`;
  $("gpuUtilLabel").textContent = `${gpu.util_percent}%`;
  $("gpuVramFill").style.width = `${gpu.vram_percent}%`;
  $("gpuVramLabel").textContent =
    `${(gpu.vram_used_mb / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(1)} GB`;
  $("gpuTemp").textContent = gpu.temperature_c != null ? `${gpu.temperature_c} °C` : "— °C";
  $("gpuPower").textContent = gpu.power_w != null ? `${gpu.power_w.toFixed(0)} W` : "— W";
}

function setStatus(msg, kind) {
  const s = $("status");
  s.textContent = msg;
  s.className = "muted" + (kind ? " " + kind : "");
}

// Diffusers expects a white-on-black mask: white = inpaint area.
async function maskBlob() {
  const out = document.createElement("canvas");
  out.width = maskCanvas.width;
  out.height = maskCanvas.height;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, out.width, out.height);
  const data = maskCtx.getImageData(0, 0, out.width, out.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) {
    if (px[i + 3] > 0) { px[i] = px[i + 1] = px[i + 2] = 255; px[i + 3] = 255; }
  }
  ctx.putImageData(data, 0, 0);
  return new Promise(res => out.toBlob(res, "image/png"));
}
