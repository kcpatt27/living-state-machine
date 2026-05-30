"""SQLite telemetry for verifier decisions and scoring context snapshots.

Every gated path (writer, reply, summarizer) appends a row to
``$LSM_HOME/verifier_log.sqlite``. Event rows are stored in ``verifier_events``.
Optional context snapshots (facts / identity / corrections / recent user turns)
are stored in ``verifier_context`` keyed by event id.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lsm.core.verifier import Verdict, VerifierContext, Violation


@dataclass(frozen=True, slots=True)
class VerifierEvent:
    """One persisted verifier scoring event."""

    id: str
    ts: float
    source: str
    candidate: str
    energy: float
    decision: str
    violations: tuple[dict[str, Any], ...]
    reprompt_used: bool


@dataclass(frozen=True, slots=True)
class VerifierContextRecord:
    """Serialized verifier context linked to one event id."""

    event_id: str
    facts_snapshot: tuple[dict[str, Any], ...]
    identity_snapshot: dict[str, Any] | None
    recent_corrections: tuple[dict[str, Any], ...]
    recent_user_turns: tuple[str, ...]
    ctx_hash: str


def _violations_to_json(violations: tuple[Violation, ...]) -> str:
    rows = [
        {
            "kind": v.kind,
            "detail": v.detail,
            "weight": v.weight,
            "evidence": v.evidence,
        }
        for v in violations
    ]
    return json.dumps(rows)


class VerifierLog:
    """Append-only SQLite log of verifier events."""

    SCHEMA = (
        "CREATE TABLE IF NOT EXISTS verifier_events ("
        " id TEXT PRIMARY KEY,"
        " ts REAL NOT NULL,"
        " source TEXT NOT NULL,"
        " candidate TEXT NOT NULL,"
        " energy REAL NOT NULL,"
        " decision TEXT NOT NULL,"
        " violations_json TEXT NOT NULL,"
        " reprompt_used INTEGER NOT NULL"
        ")"
    )
    SCHEMA_CONTEXT = (
        "CREATE TABLE IF NOT EXISTS verifier_context ("
        " event_id TEXT PRIMARY KEY,"
        " facts_snapshot_json TEXT NOT NULL,"
        " identity_snapshot_json TEXT NOT NULL,"
        " recent_corrections_json TEXT NOT NULL,"
        " recent_user_turns_json TEXT NOT NULL,"
        " ctx_hash TEXT NOT NULL,"
        " FOREIGN KEY(event_id) REFERENCES verifier_events(id)"
        ")"
    )

    def __init__(self, path: Path) -> None:
        import sqlite3

        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(self.SCHEMA)
        self._conn.execute(self.SCHEMA_CONTEXT)
        self._conn.commit()

    def append(
        self,
        *,
        source: str,
        candidate: str,
        verdict: Verdict,
        context: VerifierContextRecord | None = None,
        reprompt_used: bool = False,
        event_id: str | None = None,
        ts: float | None = None,
    ) -> str:
        """Persist one event. Returns the row id."""
        eid = event_id or str(uuid.uuid4())
        now = float(ts if ts is not None else time.time())
        self._conn.execute("BEGIN")
        self._conn.execute(
            "INSERT INTO verifier_events "
            "(id, ts, source, candidate, energy, decision, violations_json, reprompt_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                eid,
                now,
                source,
                candidate,
                float(verdict.energy),
                verdict.decision,
                _violations_to_json(verdict.violations),
                1 if reprompt_used else 0,
            ),
        )
        if context is not None:
            self._conn.execute(
                "INSERT OR REPLACE INTO verifier_context "
                "(event_id, facts_snapshot_json, identity_snapshot_json, "
                "recent_corrections_json, recent_user_turns_json, ctx_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    eid,
                    json.dumps(list(context.facts_snapshot)),
                    json.dumps(context.identity_snapshot or {}),
                    json.dumps(list(context.recent_corrections)),
                    json.dumps(list(context.recent_user_turns)),
                    context.ctx_hash,
                ),
            )
        self._conn.commit()
        return eid

    def recent(self, *, limit: int = 20) -> list[VerifierEvent]:
        rows = self._conn.execute(
            "SELECT id, ts, source, candidate, energy, decision, violations_json, "
            "reprompt_used FROM verifier_events ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM verifier_events").fetchone()
        return int(row[0] or 0)

    def context_for(self, event_id: str) -> VerifierContextRecord | None:
        row = self._conn.execute(
            "SELECT event_id, facts_snapshot_json, identity_snapshot_json, "
            "recent_corrections_json, recent_user_turns_json, ctx_hash "
            "FROM verifier_context WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_context(row)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


def _row_to_event(row: tuple[Any, ...]) -> VerifierEvent:
    raw = json.loads(row[6] or "[]")
    violations = tuple(raw) if isinstance(raw, list) else ()
    return VerifierEvent(
        id=str(row[0]),
        ts=float(row[1]),
        source=str(row[2]),
        candidate=str(row[3]),
        energy=float(row[4]),
        decision=str(row[5]),
        violations=violations,  # type: ignore[arg-type]
        reprompt_used=bool(row[7]),
    )


def _row_to_context(row: tuple[Any, ...]) -> VerifierContextRecord:
    raw_facts = json.loads(row[1] or "[]")
    raw_identity = json.loads(row[2] or "{}")
    raw_corr = json.loads(row[3] or "[]")
    raw_turns = json.loads(row[4] or "[]")
    facts = tuple(raw_facts) if isinstance(raw_facts, list) else ()
    identity = raw_identity if isinstance(raw_identity, dict) and raw_identity else None
    corrections = tuple(raw_corr) if isinstance(raw_corr, list) else ()
    turns = tuple(str(x) for x in raw_turns) if isinstance(raw_turns, list) else ()
    return VerifierContextRecord(
        event_id=str(row[0]),
        facts_snapshot=facts,  # type: ignore[arg-type]
        identity_snapshot=identity,
        recent_corrections=corrections,  # type: ignore[arg-type]
        recent_user_turns=turns,
        ctx_hash=str(row[5]),
    )


def build_context_record(event_id: str, ctx: VerifierContext) -> VerifierContextRecord:
    """Freeze a live :class:`VerifierContext` into JSON-safe primitives."""
    facts_snapshot: list[dict[str, Any]] = []
    if ctx.facts is not None:
        try:
            for fact in ctx.facts.all_facts():
                facts_snapshot.append(
                    {
                        "id": str(fact.id),
                        "owner": str(fact.owner),
                        "category": str(fact.category),
                        "subject": str(fact.subject or ""),
                        "value": str(fact.value),
                    }
                )
        except Exception:
            facts_snapshot = []
    identity_snapshot: dict[str, Any] | None = None
    if ctx.identity is not None:
        identity_snapshot = {
            "name": str(ctx.identity.name or ""),
            "description": str(ctx.identity.description or ""),
        }
    recent_corrections = tuple(
        {
            "id": str(entry.id),
            "text": str(entry.text),
            "source": str(entry.source),
            "tags": list(entry.tags),
        }
        for entry in ctx.recent_corrections
    )
    recent_turns = tuple(str(t) for t in ctx.recent_user_turns)
    hash_payload = {
        "facts": facts_snapshot,
        "identity": identity_snapshot or {},
        "corrections": list(recent_corrections),
        "recent_user_turns": list(recent_turns),
    }
    ctx_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return VerifierContextRecord(
        event_id=event_id,
        facts_snapshot=tuple(facts_snapshot),
        identity_snapshot=identity_snapshot,
        recent_corrections=recent_corrections,
        recent_user_turns=recent_turns,
        ctx_hash=ctx_hash,
    )


__all__ = [
    "VerifierContextRecord",
    "VerifierEvent",
    "VerifierLog",
    "build_context_record",
]
