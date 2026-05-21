"""Local filesystem storage for task inputs, masks, and outputs.

Layout (under DATA_DIR, default ./data):
    data/
      aipic.db                          — SQLite, see backend/db.py
      tasks/{task_id}/
        input{ext}                      — uploaded image
        mask.png                        — painted mask (inpaint only)
        variants/{variant_id}{ext}      — model output per variant

Pure local — single machine, single GPU. The interface (save_bytes, path_for,
delete_task_dir) is narrow enough that swapping for S3 / MinIO later is one
module away: same signatures, different backend.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DATA_DIR = Path(os.environ.get("AIPIC_DATA_DIR", "data")).resolve()


def init() -> None:
    """Create the storage tree if missing. Safe to call repeatedly."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "tasks").mkdir(parents=True, exist_ok=True)


def task_dir(task_id: str) -> Path:
    p = DATA_DIR / "tasks" / task_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def variants_dir(task_id: str) -> Path:
    p = task_dir(task_id) / "variants"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_input(task_id: str, data: bytes, ext: str = ".png") -> Path:
    path = task_dir(task_id) / f"input{ext}"
    path.write_bytes(data)
    return path


def save_mask(task_id: str, data: bytes) -> Path:
    path = task_dir(task_id) / "mask.png"
    path.write_bytes(data)
    return path


def save_face_image(task_id: str, data: bytes) -> Path:
    """Persist a face reference photo for PuLID identity-preserving
    generation. Stored next to the input/mask in the task dir."""
    path = task_dir(task_id) / "face.png"
    path.write_bytes(data)
    return path


def save_variant_output(task_id: str, variant_id: str, data: bytes, ext: str = ".png") -> Path:
    path = variants_dir(task_id) / f"{variant_id}{ext}"
    path.write_bytes(data)
    return path


def delete_task_dir(task_id: str) -> None:
    """Wipe a task's storage tree. Idempotent — no error if it's already gone."""
    p = DATA_DIR / "tasks" / task_id
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
