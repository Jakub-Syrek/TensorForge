const $ = (id) => document.getElementById(id);

const state = {
  mode: "kontext",
  image: null,        // HTMLImageElement
  imageFile: null,    // File
  drawing: false,
  brush: 40,
  maskDirty: false,
};

const imgCanvas = $("imgCanvas");
const maskCanvas = $("maskCanvas");
const imgCtx = imgCanvas.getContext("2d");
const maskCtx = maskCanvas.getContext("2d");

// --- health check --------------------------------------------------------
fetch("/api/health").then(r => r.json()).then(h => {
  const dev = h.cuda_available ? `${h.device} · sm_${(h.capability || []).join("")}` : "CPU";
  $("health").textContent = `torch ${h.torch} · ${dev}`;
}).catch(() => { $("health").textContent = "health: server unreachable"; });

// --- file input + drop ---------------------------------------------------
const dz = $("dropzone");
const file = $("file");
dz.addEventListener("click", () => file.click());
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

// --- mode toggle ---------------------------------------------------------
document.querySelectorAll(".seg-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.mode = btn.dataset.mode;
    $("brushRow").classList.toggle("hidden", state.mode !== "inpaint");
    maskCanvas.style.pointerEvents = state.mode === "inpaint" ? "auto" : "none";
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
  if (state.mode === "inpaint") fd.append("mask", await maskBlob(), "mask.png");

  $("run").disabled = true;
  setStatus("running…", "");
  $("progressRow").classList.remove("hidden");
  startProgressPoll();
  try {
    const r = await fetch("/api/edit", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.text()) || r.statusText);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    $("result").src = url;
    const dl = $("download");
    dl.href = url;
    dl.classList.remove("hidden");
    setStatus("done", "ok");
  } catch (e) {
    setStatus(String(e.message || e), "err");
  } finally {
    $("run").disabled = false;
    stopProgressPoll();
  }
}

// --- progress polling ---------------------------------------------------
let progTimer = null;

function startProgressPoll() {
  stopProgressPoll();
  pollOnce();  // immediate first tick
  progTimer = setInterval(pollOnce, 500);
}

function stopProgressPoll() {
  if (progTimer) clearInterval(progTimer);
  progTimer = null;
}

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
