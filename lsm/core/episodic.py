"""Append-only episodic log.

Every turn (user input, composed prompt, assistant reply, tool call, feedback,
etc.) is written here as a row. This is the raw audit record that outlives all
derived state. Later levels read from it for replay and eval.

Schema (SQLite):

    turns(turn_id TEXT PK, session_id TEXT, role TEXT, content TEXT,
          ts REAL, meta TEXT)

A small amount of metadata (backend name, model, option hash, reward, etc.)
lives in the JSON ``meta`` column so schema changes don't require migrations
for the common case.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlite_utils

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TurnRecord:
    """A single row in the episodic log."""

    session_id: str
    role: str
    content: str
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


class EpisodicLog:
    """Thin wrapper around a SQLite ``turns`` table.

    Writes are durable (WAL mode) but not fsync-per-insert; turn throughput is
    well under the rate SQLite can absorb safely.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite_utils.Database(str(self.path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        turns = self._db["turns"]
        if not turns.exists():
            turns.create(
                {
                    "turn_id": str,
                    "session_id": str,
                    "role": str,
                    "content": str,
                    "ts": float,
                    "meta": str,
                },
                pk="turn_id",
            )
            turns.create_index(["session_id", "ts"])
        self._db.executescript("PRAGMA journal_mode=WAL;")

    def append(self, record: TurnRecord) -> str:
        """Insert a turn and return its id."""
        self._db["turns"].insert(
            {
                "turn_id": record.turn_id,
                "session_id": record.session_id,
                "role": record.role,
                "content": record.content,
                "ts": record.ts,
                "meta": json.dumps(record.meta),
            }
        )
        return record.turn_id

    def session(self, session_id: str, limit: int | None = None) -> list[TurnRecord]:
        """Return rows for a session, chronological."""
        q = "SELECT * FROM turns WHERE session_id = ? ORDER BY ts ASC"
        params: tuple[Any, ...] = (session_id,)
        if limit is not None:
            q += " LIMIT ?"
            params = (session_id, limit)
        rows = list(self._db.execute(q, params).fetchall())
        cols = ["turn_id", "session_id", "role", "content", "ts", "meta"]
        out: list[TurnRecord] = []
        for row in rows:
            d = dict(zip(cols, row, strict=True))
            out.append(
                TurnRecord(
                    turn_id=d["turn_id"],
                    session_id=d["session_id"],
                    role=d["role"],
                    content=d["content"],
                    ts=d["ts"],
                    meta=json.loads(d["meta"]) if d["meta"] else {},
                )
            )
        return out

    def count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM turns").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._db.conn.close()
