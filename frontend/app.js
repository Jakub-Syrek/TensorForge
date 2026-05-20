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
  pipelineMode: false,
  currentPipelineId: null,
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

  // Profile line in the header — at-a-glance "what's loaded" without
  // having to remember which launch profile was used.
  const profileParts = [];
  profileParts.push(h.quant ? `quant: ${h.quant}` : "quant: bf16");
  if (h.max_edge) profileParts.push(`max_edge: ${h.max_edge}`);
  if (h.accel && h.accel.repo) profileParts.push(`accel: ${h.accel.repo.split("/").pop()}`);
  $("profile").textContent = profileParts.join(" · ");
  // Pre-fill the max_edge input placeholder with the server default so
  // the user sees what 'empty = server' actually means.
  if (h.max_edge) $("maxEdge").placeholder = String(h.max_edge);

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

// --- pipeline mode toggle ----------------------------------------------
$("pipelineToggle").addEventListener("change", (e) => {
  state.pipelineMode = e.target.checked;
  document.querySelector(".pipeline-toggle").classList.toggle("on", state.pipelineMode);
  // Repurpose the run button label so the user knows what they're firing.
  $("run").textContent = state.pipelineMode ? "run pipeline" : "edit image";
  // Hint via placeholder when pipeline mode is on.
  $("prompt").placeholder = state.pipelineMode
    ? "one step per line. optional [mode] prefix, e.g.:\n[generate] a sunset over mountains\nadd a small cabin in the foreground\n[kontext] make it twilight"
    : "e.g. 'replace the sky with a dramatic sunset' (Kontext) — or for inpaint: describe only what should appear inside the mask";
});

// --- accel toggle: when on, auto-pin steps to 8 (Hyper-SD default);
// when off, auto-pin to 28 (full quality). User can still override manually.
$("accelToggle").addEventListener("change", (e) => {
  $("steps").value = e.target.checked ? "8" : "28";
});

// --- mode toggle ---------------------------------------------------------
// Per-mode sensible defaults. User can override afterwards via the inputs.
const MODE_DEFAULTS = {
  auto:     { steps: 28, guidance: 3.5 },   // matches kontext; generate path overridden server-side
  kontext:  { steps: 28, guidance: 3.5 },
  inpaint:  { steps: 28, guidance: 3.5 },
  qwen:     { steps: 50, guidance: 4.0 },   // Qwen-Image-Edit native defaults
  generate: { steps: 4,  guidance: 0.0 },   // Flux schnell — 4-step distilled, guidance baked in
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
  const prompt = $("prompt").value.trim();
  if (!prompt) return setStatus("prompt is empty", "err");

  // Pipeline mode hands off to a different code path entirely.
  if (state.pipelineMode) {
    return runPipeline(prompt);
  }

  // Image is required for every mode except generate; auto resolves
  // server-side so we accept it even without an image.
  const needsImage = !(state.mode === "generate" || (state.mode === "auto" && !state.imageFile));
  if (needsImage && !state.imageFile) return setStatus("upload an image first", "err");
  if (state.mode === "inpaint" && !state.maskDirty) return setStatus("paint a mask first", "err");

  const variantsCount = Math.max(1, Math.min(8, parseInt($("variants").value, 10) || 1));

  const fd = new FormData();
  if (state.imageFile) fd.append("image", state.imageFile);
  fd.append("mode", state.mode);
  fd.append("prompt", prompt);
  fd.append("steps", $("steps").value);
  fd.append("guidance", $("guidance").value);
  if ($("seed").value) fd.append("seed", $("seed").value);
  fd.append("sharpen_level", $("sharpen").value);
  if ($("maxEdge").value) fd.append("max_edge", $("maxEdge").value);
  if (state.accelAvailable) {
    fd.append("use_accel", $("accelToggle").checked ? "true" : "false");
  }
  fd.append("variants", String(variantsCount));
  if (state.mode === "inpaint") fd.append("mask", await maskBlob(), "mask.png");

  $("run").disabled = true;
  $("abort").classList.remove("hidden");
  state.userAborted = false;
  setStatus("submitting…", "");
  startProgressPoll(500);
  try {
    const r = await fetch("/api/tasks", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.text()) || r.statusText);
    const task = await r.json();
    state.currentTaskId = task.id;
    setStatus(`task ${task.id.slice(0, 8)} queued (${variantsCount} variant${variantsCount > 1 ? "s" : ""})`, "");
    const final = await pollTask(task.id);
    if (final === null) return; // user aborted; status set elsewhere
    renderTaskResult(final);
    addHistoryEntry({
      taskId: final.id,
      inputFile: state.imageFile,
      prompt,
      mode: state.mode,
      variants: final.variants,
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
    state.currentTaskId = null;
    startProgressPoll(2000);
  }
}

// --- pipeline submission + polling -------------------------------------
function parsePipelineSteps(raw) {
  const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
  return lines.map((line) => {
    const m = line.match(/^\[(\w+)\]\s+(.+)$/);
    if (m) return { mode: m[1].toLowerCase(), prompt: m[2] };
    return { mode: "auto", prompt: line };
  });
}

async function runPipeline(rawPrompt) {
  const steps = parsePipelineSteps(rawPrompt);
  if (steps.length === 0) return setStatus("no steps in pipeline", "err");
  if (steps.length > 10) return setStatus(`too many steps (${steps.length} > 10)`, "err");

  // Validate per-step modes client-side for a fast error before we round-trip.
  const VALID = new Set(["auto", "kontext", "inpaint", "qwen", "generate"]);
  for (let i = 0; i < steps.length; i++) {
    if (!VALID.has(steps[i].mode)) {
      return setStatus(`step ${i + 1}: invalid mode '${steps[i].mode}'`, "err");
    }
  }

  const fd = new FormData();
  if (state.imageFile) fd.append("image", state.imageFile);
  fd.append("steps_json", JSON.stringify(steps));
  if (state.mode === "inpaint" && state.maskDirty) {
    fd.append("mask", await maskBlob(), "mask.png");
  }

  $("run").disabled = true;
  $("abort").classList.remove("hidden");
  state.userAborted = false;
  setStatus(`submitting pipeline (${steps.length} steps)…`, "");
  startProgressPoll(500);
  try {
    const r = await fetch("/api/pipelines", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.text()) || r.statusText);
    const pipe = await r.json();
    state.currentPipelineId = pipe.pipeline_id;
    renderPipelineProgress(pipe);
    const final = await pollPipeline(pipe.pipeline_id);
    if (final === null) return;
    renderPipelineProgress(final);
    const lastDone = [...final.steps].reverse().find((t) => t.status === "done");
    if (lastDone) {
      const v = lastDone.variants.find((x) => x.status === "done");
      if (v && v.output_url) {
        state.lastResultUrl = v.output_url;
        state.lastVariantId = v.id;
        $("download").href = v.output_url;
      }
    }
    setStatus(`pipeline ${final.status} · ${final.steps.length} steps`, final.status === "done" ? "ok" : "err");
  } catch (e) {
    if (state.userAborted) setStatus("aborted", "err");
    else setStatus(String(e.message || e), "err");
  } finally {
    $("run").disabled = false;
    $("abort").classList.add("hidden");
    state.currentPipelineId = null;
    startProgressPoll(2000);
  }
}

async function pollPipeline(pid) {
  const TERMINAL = new Set(["done", "failed", "aborted"]);
  while (true) {
    if (state.userAborted) return null;
    let r;
    try {
      r = await fetch(`/api/pipelines/${pid}`);
    } catch {
      await new Promise((res) => setTimeout(res, 500));
      continue;
    }
    if (!r.ok) throw new Error(`pipeline fetch failed: ${r.status}`);
    const pipe = await r.json();
    renderPipelineProgress(pipe);
    if (TERMINAL.has(pipe.status)) return pipe;
    await new Promise((res) => setTimeout(res, 500));
  }
}

function renderPipelineProgress(pipe) {
  const wrap = $("resultWrap");
  wrap.innerHTML = "";
  const list = document.createElement("div");
  list.className = "pipeline-steps";
  pipe.steps.forEach((t, i) => {
    const row = document.createElement("div");
    row.className = `pstep ${t.status}`;
    const idx = document.createElement("span");
    idx.className = "pstep-idx";
    idx.textContent = String(i + 1);
    row.appendChild(idx);

    const body = document.createElement("div");
    body.className = "pstep-body";
    const promptLine = document.createElement("div");
    promptLine.className = "pstep-prompt";
    promptLine.textContent = `[${t.mode}] ${t.prompt}`;
    const meta = document.createElement("div");
    meta.className = "pstep-meta";
    const runtimes = t.variants.filter((v) => v.runtime_ms).map((v) => v.runtime_ms);
    const elapsed = runtimes.length ? `${(runtimes[0] / 1000).toFixed(1)} s` : "";
    meta.textContent = `${t.status}${elapsed ? " · " + elapsed : ""}${t.error ? " · " + t.error : ""}`;
    body.appendChild(promptLine);
    body.appendChild(meta);
    row.appendChild(body);

    const doneVariant = t.variants.find((v) => v.status === "done" && v.output_url);
    if (doneVariant) {
      const thumb = document.createElement("div");
      thumb.className = "pstep-thumb";
      thumb.style.backgroundImage = `url(${doneVariant.output_url})`;
      thumb.title = "click to use this step's output as input";
      thumb.addEventListener("click", (ev) => {
        ev.stopPropagation();
        state.lastResultUrl = doneVariant.output_url;
        state.lastVariantId = doneVariant.id;
      });
      row.appendChild(thumb);
    }
    list.appendChild(row);
  });
  wrap.appendChild(list);
  wrap.classList.remove("hidden");
  $("resultActions").classList.remove("hidden");
}

// Poll /api/tasks/:id until it reaches a terminal state (done/failed/aborted/approved).
// Returns null if the user aborted; otherwise the final task envelope.
async function pollTask(taskId) {
  const TERMINAL = new Set(["done", "failed", "aborted", "approved"]);
  while (true) {
    if (state.userAborted) return null;
    let r;
    try {
      r = await fetch(`/api/tasks/${taskId}`);
    } catch {
      // Transient network blip — retry next tick.
      await new Promise((res) => setTimeout(res, 500));
      continue;
    }
    if (!r.ok) throw new Error(`task fetch failed: ${r.status}`);
    const task = await r.json();
    const doneN = task.variants.filter((v) => v.status === "done").length;
    setStatus(`task ${taskId.slice(0, 8)} · ${task.status} · ${doneN}/${task.variants.length} variants`, "");
    if (TERMINAL.has(task.status)) return task;
    await new Promise((res) => setTimeout(res, 500));
  }
}

// Render the result panel for a task. Single variant → simple image;
// multi-variant → clickable grid with approve flow.
function renderTaskResult(task) {
  const wrap = $("resultWrap");
  wrap.innerHTML = "";
  state.currentTask = task;

  if (task.variants.length === 1) {
    const v = task.variants[0];
    if (v.status !== "done") {
      const ph = document.createElement("div");
      ph.className = "vplaceholder";
      ph.textContent = v.status + (v.error ? `: ${v.error}` : "");
      wrap.appendChild(ph);
    } else {
      const img = document.createElement("img");
      img.id = "result";
      img.src = v.output_url;
      wrap.appendChild(img);
      state.lastResultUrl = v.output_url;
      state.lastVariantId = v.id;
      $("seed").value = v.seed;
      $("download").href = v.output_url;
      setStatus(`done · seed ${v.seed}`, "ok");
    }
  } else {
    const grid = document.createElement("div");
    const cols = task.variants.length <= 4 ? 2 : task.variants.length <= 9 ? 3 : 4;
    grid.className = `variants-grid cols-${cols}`;
    task.variants.forEach((v) => {
      const cell = document.createElement("div");
      cell.className = "variant-cell";
      if (v.approved) cell.classList.add("approved");
      if (v.status === "done") {
        cell.innerHTML = `
          <img src="${v.output_url}" alt="" />
          <div class="vmeta"><span>seed ${v.seed}</span><span>${(v.runtime_ms || 0) / 1000 | 0}s</span></div>`;
        cell.addEventListener("click", () => selectVariant(task.id, v));
      } else {
        const ph = document.createElement("div");
        ph.className = "vplaceholder";
        ph.textContent = v.status;
        cell.appendChild(ph);
      }
      grid.appendChild(cell);
    });
    wrap.appendChild(grid);
    setStatus(`${task.variants.length} variants — click one to approve`, "");
  }

  wrap.classList.remove("hidden");
  $("resultActions").classList.remove("hidden");
}

// Approve a variant, mark it as the "winner" for chaining/download.
async function selectVariant(taskId, variant) {
  setStatus(`approving variant ${variant.seed}…`, "");
  let r;
  try {
    r = await fetch(`/api/tasks/${taskId}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant_id: variant.id }),
    });
  } catch (e) {
    return setStatus("approve failed: " + e.message, "err");
  }
  if (!r.ok) return setStatus(`approve failed: ${r.status}`, "err");
  const updated = await r.json();
  state.lastResultUrl = variant.output_url;
  state.lastVariantId = variant.id;
  $("seed").value = variant.seed;
  $("download").href = variant.output_url;
  renderTaskResult(updated);
  setStatus(`approved · seed ${variant.seed}`, "ok");
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
  $("resultWrap").innerHTML = "";
  $("resultWrap").classList.add("hidden");
  $("resultActions").classList.add("hidden");
  state.lastResultBlob = null;
  state.lastResultUrl = null;
  state.lastVariantId = null;
  state.currentTask = null;
  setStatus("", "");
});

// --- history: ring buffer of last HISTORY_MAX successful edits ---------
function addHistoryEntry(entry) {
  // Pick the approved variant if there is one; otherwise the first 'done'.
  const winner =
    (entry.variants || []).find((v) => v.approved) ||
    (entry.variants || []).find((v) => v.status === "done");
  if (!winner || !winner.output_url) return;

  const item = {
    id: entry.taskId || Date.now(),
    resultUrl: winner.output_url, // persistent server URL
    seed: winner.seed,
    prompt: entry.prompt,
    mode: entry.mode,
    steps: $("steps").value,
    guidance: $("guidance").value,
    useAccel: entry.useAccel,
  };
  history.unshift(item);
  while (history.length > HISTORY_MAX) history.pop();
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

async function loadHistoryEntry(item) {
  // Fetch the server URL as a blob, wrap as File, feed loadFile().
  let blob;
  try {
    const r = await fetch(item.resultUrl);
    if (!r.ok) throw new Error(`${r.status}`);
    blob = await r.blob();
  } catch (e) {
    return setStatus("history fetch failed: " + e.message, "err");
  }
  const file = new File([blob], `history-${item.id}.png`, { type: "image/png" });
  loadFile(file);
  $("prompt").value = item.prompt || "";
  $("seed").value = item.seed || "";
  if (item.mode === "kontext" || item.mode === "inpaint" || item.mode === "qwen") {
    document.querySelectorAll(".seg-btn").forEach((b) =>
      b.classList.toggle("active", b.dataset.mode === item.mode),
    );
    state.mode = item.mode;
    $("brushRow").classList.toggle("hidden", item.mode !== "inpaint");
    maskCanvas.style.pointerEvents = item.mode === "inpaint" ? "auto" : "none";
  }
  if (item.steps) $("steps").value = item.steps;
  if (item.guidance) $("guidance").value = item.guidance;
  if (state.accelAvailable && item.useAccel !== null && item.useAccel !== undefined) {
    $("accelToggle").checked = item.useAccel;
  }
  $("resultWrap").innerHTML = "";
  $("resultWrap").classList.add("hidden");
  $("resultActions").classList.add("hidden");
  state.lastResultUrl = null;
  state.lastVariantId = null;
  setStatus(`loaded · seed ${item.seed}`, "ok");
}

$("historyClear").addEventListener("click", () => {
  // Server-side URLs (no blob URLs to revoke now); just drop the in-memory list.
  history.length = 0;
  renderHistory();
});

// --- chain: use current result as the next input -----------------------
$("useAsInput").addEventListener("click", async () => {
  const url = state.lastResultUrl;
  if (!url) return setStatus("no result to chain yet", "err");

  // Fetch the variant PNG (server-side URL) and wrap as a File so the
  // existing upload pipeline (loadFile) ingests it identically.
  let blob;
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status}`);
    blob = await r.blob();
  } catch (e) {
    return setStatus("chain fetch failed: " + e.message, "err");
  }
  const stamped = `chained-${Date.now()}.png`;
  const file = new File([blob], stamped, { type: "image/png" });

  loadFile(file);
  // Clear UI surface for the next instruction. Keep mode + steps +
  // guidance untouched; clear prompt and seed because the previous
  // prompt likely doesn't apply to the new image.
  $("prompt").value = "";
  $("seed").value = "";
  $("resultWrap").innerHTML = "";
  $("resultWrap").classList.add("hidden");
  $("resultActions").classList.add("hidden");
  state.lastResultUrl = null;
  state.lastVariantId = null;
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
