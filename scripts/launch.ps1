<#
.SYNOPSIS
    Launch the TensorForge server with a chosen performance profile.

.DESCRIPTION
    Picks the right combination of FLUX_QUANT / FLUX_MAX_EDGE / FLUX_ACCEL_*
    env vars for one of four sensible workflows, verifies the venv exists,
    frees port 8000 if asked, and starts the server.

    Each profile trades quality against per-edit latency in a different way.
    Run `scripts\launch.ps1 -Help` to print the comparison table without
    actually launching.

.PARAMETER Profile
    One of: fast | hyper | quality | custom
      fast    - NF4 quantization, 28 steps, ~7-13 s/step. Default.
                Best for everyday edits on a 16 GB card.
      hyper   - NF4 + Hyper-SD 8-step LoRA, ~3 s/step x 8 steps = ~25 s/edit.
                Fastest profile. Quality slightly below 28-step baseline.
      quality - bf16, 28 steps, model_cpu_offload, ~230 s/step. Use ONLY
                when you want maximum fidelity and have time. PCIe-bound,
                GPU at ~65 W instead of 250 W - slow by design.
      custom  - start with no profile env vars. You set them yourself
                before invoking the script.

.PARAMETER MaxEdge
    Override FLUX_MAX_EDGE. Defaults: 1024 for NF4 profiles, 512 for quality.

.PARAMETER KillExisting
    If port 8000 is already bound, kill that process first.
    Without this switch the script exits with a clear error.

.PARAMETER Help
    Print the profile comparison table and exit. No launch.

.EXAMPLE
    scripts\launch.ps1
    # Equivalent to: scripts\launch.ps1 -Profile fast

.EXAMPLE
    scripts\launch.ps1 -Profile hyper -KillExisting
    # Kill any stale server, launch with Hyper-SD 8-step LoRA.

.EXAMPLE
    scripts\launch.ps1 -Profile quality -MaxEdge 768
    # bf16 final-render profile, but downscale input to 768 to stay under
    # the worst-case PCIe thrashing.
#>

[CmdletBinding()]
param(
    [ValidateSet('fast', 'hyper', 'quality', 'custom')]
    [string]$Profile = 'fast',

    [int]$MaxEdge,

    [switch]$KillExisting,

    [switch]$Help
)

# Don't set $ErrorActionPreference = 'Stop' globally — when we run the
# Python server via '&', uvicorn's INFO logs hit stderr, and PS 5.1 wraps
# every stderr line in a NativeCommandError record. With Stop, that
# terminates the whole script the moment the server starts logging. We
# use explicit `exit N` for our own error paths instead.

# Resolve repo paths from the script location, so the launcher works from
# any cwd. scripts/launch.ps1 -> repo root is one level up.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$ServerEntry = Join-Path $RepoRoot 'backend\server.py'

function Write-Section($title) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host $title -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray
}

function Get-OrDefault($value, $fallback) {
    # PS 5.1 lacks the ?? operator, so we wrap the null-coalesce explicitly.
    if ([string]::IsNullOrEmpty($value)) { return $fallback } else { return $value }
}

function Show-ProfileTable {
    Write-Section "TensorForge - profile comparison"
    Write-Host ""
    Write-Host "  profile   step time   VRAM      power   per-edit    quality"      -ForegroundColor Yellow
    Write-Host "  -------   ---------   ----      -----   --------    -------"      -ForegroundColor DarkGray
    Write-Host "  fast      ~7-13 s     ~14 GB    ~245 W   3-6 min     baseline"
    Write-Host "  hyper     ~3 s        ~14 GB    ~250 W   ~25 s       slightly off"
    Write-Host "  quality   ~230 s      ~15.4 GB  ~65 W    ~108 min    max"
    Write-Host "  custom    (whatever env vars you set)"
    Write-Host ""
    Write-Host "  fast    = NF4 quantization, 28 inference steps."
    Write-Host "            The everyday choice on a 16 GB card."
    Write-Host ""
    Write-Host "  hyper   = NF4 + Hyper-SD 8-step LoRA. ~9x faster than"
    Write-Host "            fast for slightly-different output. UI checkbox"
    Write-Host "            ('acceleration LoRA - 8 steps') will appear and"
    Write-Host "            can be toggled per edit."
    Write-Host ""
    Write-Host "  quality = bf16 + model_cpu_offload. Model (~21 GB) doesn't"
    Write-Host "            fit in 16 GB VRAM, so diffusers streams it across"
    Write-Host "            PCIe each step. Card is mostly waiting - only use"
    Write-Host "            this when you want bf16 fidelity and don't mind"
    Write-Host "            waiting 100+ minutes per edit."
    Write-Host ""
    Write-Host "  Override the input downscale cap with the MaxEdge parameter,"
    Write-Host "  e.g. '-Profile fast -MaxEdge 768' to favor speed over detail."
    Write-Host ""
}

if ($Help) {
    Show-ProfileTable
    exit 0
}

# Sanity-check venv. We rely on scripts/setup.py having been run first.
if (-not (Test-Path $VenvPython)) {
    Write-Host "venv not found at $VenvPython" -ForegroundColor Red
    Write-Host "Run scripts\setup.py first to create it." -ForegroundColor Yellow
    exit 1
}

# Port preflight. uvicorn binding fails late and noisy; catch it early.
$ExistingListener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($ExistingListener) {
    $existingPid = $ExistingListener.OwningProcess
    if ($KillExisting) {
        Write-Host "port 8000 is busy (PID $existingPid) - killing as requested" -ForegroundColor Yellow
        Stop-Process -Id $existingPid -Force
        Start-Sleep -Milliseconds 500
    } else {
        Write-Host "port 8000 is busy (PID $existingPid)." -ForegroundColor Red
        Write-Host "Pass -KillExisting to take it over, or stop that process manually." -ForegroundColor Yellow
        exit 2
    }
}

# Clear any leftover env from a previous launch in the same shell - we want
# the script to be the single source of truth for this run.
Remove-Item Env:FLUX_QUANT        -ErrorAction SilentlyContinue
Remove-Item Env:FLUX_ACCEL_REPO   -ErrorAction SilentlyContinue
Remove-Item Env:FLUX_ACCEL_WEIGHT -ErrorAction SilentlyContinue
Remove-Item Env:FLUX_ACCEL_SCALE  -ErrorAction SilentlyContinue
Remove-Item Env:FLUX_MAX_EDGE     -ErrorAction SilentlyContinue

switch ($Profile) {
    'fast' {
        $env:FLUX_QUANT = '4bit'
        if (-not $MaxEdge) { $MaxEdge = 1024 }
    }
    'hyper' {
        $env:FLUX_QUANT        = '4bit'
        $env:FLUX_ACCEL_REPO   = 'ByteDance/Hyper-SD'
        $env:FLUX_ACCEL_WEIGHT = 'Hyper-FLUX.1-dev-8steps-lora.safetensors'
        $env:FLUX_ACCEL_SCALE  = '0.125'
        if (-not $MaxEdge) { $MaxEdge = 1024 }
    }
    'quality' {
        # No FLUX_QUANT - bf16. Cap at 512 to keep the PCIe thrash bounded;
        # higher resolutions push step time past 5 minutes.
        if (-not $MaxEdge) { $MaxEdge = 512 }
    }
    'custom' {
        # Caller-controlled env vars. We still respect MaxEdge if passed.
    }
}
if ($MaxEdge) {
    $env:FLUX_MAX_EDGE = "$MaxEdge"
}

Write-Section "Launching profile: $Profile"
Write-Host ("  FLUX_QUANT        : " + (Get-OrDefault $env:FLUX_QUANT        '(unset, bf16)'))
Write-Host ("  FLUX_MAX_EDGE     : " + (Get-OrDefault $env:FLUX_MAX_EDGE     '(server default)'))
Write-Host ("  FLUX_ACCEL_REPO   : " + (Get-OrDefault $env:FLUX_ACCEL_REPO   '(no LoRA)'))
Write-Host ("  FLUX_ACCEL_WEIGHT : " + (Get-OrDefault $env:FLUX_ACCEL_WEIGHT '-'))
Write-Host ("  FLUX_ACCEL_SCALE  : " + (Get-OrDefault $env:FLUX_ACCEL_SCALE  '-'))
Write-Host ""
Write-Host "  UI: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  Health probe to verify env vars stuck:" -ForegroundColor DarkGray
Write-Host "    curl -s http://127.0.0.1:8000/api/health"
Write-Host ""
Write-Host "  Stop with Ctrl+C." -ForegroundColor DarkGray
Write-Host ""

& $VenvPython $ServerEntry
