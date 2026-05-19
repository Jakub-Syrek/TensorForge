"""Report and clean up cache + temp files that accumulate over time.

The big offenders for a Flux setup:
- Hugging Face cache (~/.cache/huggingface) — Flux Kontext + Fill are ~33 GB
  each (transformer + T5 + CLIP + VAE + config blobs). Models stay forever
  unless explicitly deleted.
- torch.compile / triton cache — kernel cache, grows with model variants.
- Python __pycache__ in the repo — small but tidies up on demand.
- Local outputs/ folder — kept out of git but accumulates if you save runs.

Run modes
---------
- No flags (default):       dry-run. Reports sizes, deletes nothing.
- --apply --pyc:            remove every __pycache__ under the repo.
- --apply --outputs:        remove the outputs/ folder.
- --apply --hf <repo_id>:   delete one HF cache repo by id, e.g.
                            'black-forest-labs/FLUX.1-Kontext-dev'.
- --apply --torch-compile:  remove torch.compile / triton cache.

Examples
--------
    python scripts/clean.py                    # show what's eating disk
    python scripts/clean.py --apply --pyc
    python scripts/clean.py --apply --hf black-forest-labs/FLUX.1-Kontext-dev
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
HF_CACHE = HOME / ".cache" / "huggingface"
TORCH_INDUCTOR_CACHE = HOME / ".cache" / "torch_inductor"
TRITON_CACHE = HOME / ".triton"


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def report_hf_cache() -> None:
    print(f"\n[HF cache] {HF_CACHE}")
    if not HF_CACHE.exists():
        print("  (no HF cache yet)")
        return

    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print(f"  total: {fmt_size(dir_size(HF_CACHE))}")
        print("  (install huggingface_hub for per-repo breakdown)")
        return

    info = scan_cache_dir()
    print(f"  total: {info.size_on_disk_str}")
    if not info.repos:
        return
    print("  per repo:")
    for repo in sorted(info.repos, key=lambda r: -r.size_on_disk):
        print(f"    {repo.repo_id:55} {repo.size_on_disk_str}")


def report_torch_caches() -> None:
    for label, path in (
        ("torch.compile (inductor)", TORCH_INDUCTOR_CACHE),
        ("triton kernels", TRITON_CACHE),
    ):
        if path.exists():
            print(f"\n[{label}] {path}\n  {fmt_size(dir_size(path))}")


def report_repo_caches() -> None:
    pycache_dirs = list(REPO_ROOT.rglob("__pycache__"))
    pycache_dirs = [p for p in pycache_dirs if ".venv" not in p.parts]
    if pycache_dirs:
        total = sum(dir_size(p) for p in pycache_dirs)
        print(f"\n[__pycache__ in repo] {len(pycache_dirs)} dirs, {fmt_size(total)}")

    outputs = REPO_ROOT / "outputs"
    if outputs.exists():
        print(f"\n[outputs/] {fmt_size(dir_size(outputs))}")


def delete_path(path: Path, label: str) -> None:
    if not path.exists():
        print(f"  [skip] {label}: doesn't exist")
        return
    size = dir_size(path) if path.is_dir() else path.stat().st_size
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    print(f"  [done] {label}: freed {fmt_size(size)}")


def delete_hf_repo(repo_id: str) -> None:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print("  huggingface_hub not installed; cannot resolve repo path")
        return

    info = scan_cache_dir()
    matched = [r for r in info.repos if r.repo_id == repo_id]
    if not matched:
        print(f"  [skip] HF repo '{repo_id}' not in cache")
        return
    repo = matched[0]
    print(f"  deleting {repo.repo_id} ({repo.size_on_disk_str})")
    strategy = info.delete_revisions(*(rev.commit_hash for rev in repo.revisions))
    strategy.execute()
    print("  [done]")


def clean_pycache() -> None:
    pycache_dirs = list(REPO_ROOT.rglob("__pycache__"))
    pycache_dirs = [p for p in pycache_dirs if ".venv" not in p.parts]
    if not pycache_dirs:
        print("  no __pycache__ under repo")
        return
    freed = 0
    for p in pycache_dirs:
        freed += dir_size(p)
        shutil.rmtree(p, ignore_errors=True)
    print(f"  [done] removed {len(pycache_dirs)} __pycache__ dirs, freed {fmt_size(freed)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true", help="actually delete (default is dry-run)")
    parser.add_argument("--pyc", action="store_true", help="remove __pycache__ under the repo")
    parser.add_argument("--outputs", action="store_true", help="remove the outputs/ folder")
    parser.add_argument("--hf", metavar="REPO_ID", help="delete an HF cache repo by id")
    parser.add_argument(
        "--torch-compile", action="store_true", help="remove torch.compile + triton caches"
    )
    args = parser.parse_args()

    if not args.apply:
        print("=== Dry run — no files will be deleted. Use --apply to act. ===")
        report_hf_cache()
        report_torch_caches()
        report_repo_caches()
        print()
        return 0

    print("=== Applying cleanup ===")
    if args.pyc:
        print("\n[__pycache__]")
        clean_pycache()
    if args.outputs:
        print("\n[outputs/]")
        delete_path(REPO_ROOT / "outputs", "outputs/")
    if args.hf:
        print(f"\n[HF cache] {args.hf}")
        delete_hf_repo(args.hf)
    if args.torch_compile:
        print("\n[torch caches]")
        delete_path(TORCH_INDUCTOR_CACHE, "torch_inductor")
        delete_path(TRITON_CACHE, "triton")
    if not (args.pyc or args.outputs or args.hf or args.torch_compile):
        print("Nothing selected. Pass --pyc / --outputs / --hf REPO_ID / --torch-compile.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
