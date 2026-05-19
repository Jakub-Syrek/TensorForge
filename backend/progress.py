"""Shared progress state for the single concurrent edit job.

We don't queue jobs — the UI submits one at a time. So a module-level
singleton is enough; the pipeline writes step counts into it from the
diffusers callback, and /api/progress reads from it.

Also shells out to nvidia-smi for live GPU utilization + VRAM. Polling
at 500 ms is well within nvidia-smi's overhead budget (~30-50 ms per call).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class JobProgress:
    active: bool = False
    step: int = 0
    total: int = 0
    mode: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    last_error: str | None = None
    aborted: bool = False

    def start(self, total: int, mode: str) -> None:
        self.active = True
        self.step = 0
        self.total = total
        self.mode = mode
        self.started_at = time.monotonic()
        self.finished_at = None
        self.last_error = None
        self.aborted = False

    def advance(self, step_index_zero_based: int) -> None:
        # diffusers passes 0-based step index after each step ends, so step 0
        # firing means "first step done" -> show 1/total.
        self.step = step_index_zero_based + 1

    def request_abort(self) -> bool:
        """Signal abort to the running job. Returns True if an active job
        was signalled, False if there was nothing to abort."""
        if not self.active:
            return False
        self.aborted = True
        return True

    def finish(self, error: str | None = None) -> None:
        self.active = False
        self.finished_at = time.monotonic()
        self.last_error = error
        if error is None and self.total:
            self.step = self.total  # snap to 100% on clean finish

    def snapshot(self) -> dict:
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at if not self.active else time.monotonic()
            elapsed = end - self.started_at
        pct = (self.step / self.total * 100.0) if self.total else 0.0
        return {
            "active": self.active,
            "step": self.step,
            "total": self.total,
            "mode": self.mode,
            "elapsed_s": elapsed,
            "percent": round(pct, 1),
            "error": self.last_error,
            "aborted": self.aborted,
        }


job_progress = JobProgress()


@dataclass
class GpuStats:
    util_percent: int
    vram_used_mb: int
    vram_total_mb: int
    temperature_c: int | None = None
    power_w: float | None = None
    name: str = field(default="")

    def to_dict(self) -> dict:
        return {
            "util_percent": self.util_percent,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "vram_percent": round(self.vram_used_mb / self.vram_total_mb * 100.0, 1)
            if self.vram_total_mb
            else 0.0,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
            "name": self.name,
        }


_NVSMI_QUERY = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def parse_nvsmi_line(line: str) -> GpuStats | None:
    """Parse one CSV line from `nvidia-smi --format=csv,noheader,nounits`.

    Returns None on any parse failure — caller treats missing GPU stats as
    "unknown" rather than failing the whole request.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        name = parts[0]
        util = int(parts[1])
        used = int(parts[2])
        total = int(parts[3])
        temp = int(parts[4]) if len(parts) > 4 and parts[4] not in ("", "[N/A]") else None
        power = float(parts[5]) if len(parts) > 5 and parts[5] not in ("", "[N/A]") else None
        return GpuStats(
            util_percent=util,
            vram_used_mb=used,
            vram_total_mb=total,
            temperature_c=temp,
            power_w=power,
            name=name,
        )
    except (ValueError, IndexError):
        return None


def query_gpu_stats() -> GpuStats | None:
    # nosec B603,B607 — nvidia-smi is OS-installed at a well-known location and
    # already on PATH for any NVIDIA driver setup; args are static literals, no
    # untrusted input interpolated.
    try:
        out = subprocess.check_output(  # nosec B603 B607
            [
                "nvidia-smi",
                f"--query-gpu={_NVSMI_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = next((ln for ln in out.splitlines() if ln.strip()), "")
    return parse_nvsmi_line(line)
