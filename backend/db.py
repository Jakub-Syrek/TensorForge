"""SQLite-backed task persistence.

One Task (edit request) holds N Variants (one per requested seed). Single
table set, single SQLite file at data/aipic.db. SQLAlchemy 2.x style.

Status enum kept as plain strings to stay friendly to migrations:
  queued       — waiting in queue
  running      — picked up by worker
  done         — inference succeeded, output_path set
  failed       — inference raised; error stored
  aborted      — user-cancelled via /api/abort
  approved     — for Variant only: user picked this one
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from backend.storage import DATA_DIR


def _uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(32), primary_key=True, default=_uuid)
    status = Column(String(16), default="queued", nullable=False, index=True)
    mode = Column(String(16), nullable=False)
    prompt = Column(String, nullable=False)
    input_path = Column(String, nullable=False)
    mask_path = Column(String, nullable=True)
    original_width = Column(Integer, nullable=False)
    original_height = Column(Integer, nullable=False)
    params = Column(JSON, default=dict)
    variants_requested = Column(Integer, default=1, nullable=False)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    variants = relationship(
        "Variant",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="Variant.created_at",
    )


class Variant(Base):
    __tablename__ = "variants"

    id = Column(String(32), primary_key=True, default=_uuid)
    task_id = Column(
        String(32), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seed = Column(Integer, nullable=False)
    status = Column(String(16), default="queued", nullable=False, index=True)
    output_path = Column(String, nullable=True)
    runtime_ms = Column(Integer, nullable=True)
    error = Column(String, nullable=True)
    approved = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)

    task = relationship("Task", back_populates="variants")


DB_PATH = Path(os.environ.get("AIPIC_DB_PATH", DATA_DIR / "aipic.db"))


def _make_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False — we share the SQLite connection between the
    # FastAPI worker threadpool and the dedicated background queue worker.
    # SQLAlchemy + SQLite handle the locking internally for this pattern.
    return create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        future=True,
    )


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """Create tables if they don't exist. No migrations yet — schema is
    additive enough that we can ALTER manually for now. Switch to Alembic
    when the schema starts moving."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()
