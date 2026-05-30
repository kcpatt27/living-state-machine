"""Mine pseudo-negative verifier examples from user correction turns."""

from __future__ import annotations

import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsm.core.verifier_log import VerifierContextRecord, VerifierLog

_CORRECTION_RE = re.compile(
    r"\b("
    r"no\b|wrong\b|actually\b|that's not\b|that is not\b|i said\b|"
    r"you mean\b|not\s+.*\bit(?:'s|\s+is)\b"
    r")",
    re.IGNORECASE,
)


@dataclass(slots=True)
class MinedExample:
    """Pseudo-labeled negative from correction mining."""

    event_id: str
    source: str
    candidate: str
    label: str
    label_source: str = "correction_mining"
    context: VerifierContextRecord | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def mine_correction_examples(home: Path, *, limit: int = 200) -> list[MinedExample]:
    """Find assistant turns immediately corrected by user turns.

    Heuristic: for each user turn matching correction phrases, grab the nearest
    prior assistant turn in the same session and label it as ``reject``.
    """
    epis_path = home / "episodes.sqlite"
    if not epis_path.exists():
        return []
    context_by_candidate = _load_context_by_candidate(home / "verifier_log.sqlite")
    conn = sqlite3.connect(str(epis_path))
    try:
        rows = conn.execute(
            "SELECT turn_id, session_id, role, content, ts FROM turns ORDER BY session_id, ts ASC"
        ).fetchall()
    finally:
        conn.close()
    by_session: dict[str, list[tuple[str, str, str, float]]] = {}
    for turn_id, session_id, role, content, ts in rows:
        by_session.setdefault(str(session_id), []).append(
            (str(turn_id), str(role), str(content), float(ts))
        )
    out: list[MinedExample] = []
    for session_id, turns in by_session.items():
        last_assistant: tuple[str, str, str, float] | None = None
        for turn in turns:
            turn_id, role, content, _ts = turn
            if role == "assistant":
                last_assistant = turn
                continue
            if role != "user":
                continue
            if _CORRECTION_RE.search(content) is None:
                continue
            if last_assistant is None:
                continue
            a_turn_id, _a_role, a_text, _a_ts = last_assistant
            event_id = f"mined-{a_turn_id}-{uuid.uuid4().hex[:8]}"
            out.append(
                MinedExample(
                    event_id=event_id,
                    source="reply",
                    candidate=a_text,
                    label="reject",
                    context=context_by_candidate.get(a_text),
                    meta={
                        "session_id": session_id,
                        "assistant_turn_id": a_turn_id,
                        "correction_turn_id": turn_id,
                        "correction_text": content,
                    },
                )
            )
            if len(out) >= max(1, limit):
                return out
    return out


def _load_context_by_candidate(path: Path) -> dict[str, VerifierContextRecord]:
    if not path.exists():
        return {}
    vlog = VerifierLog(path)
    try:
        events = vlog.recent(limit=2000)
        out: dict[str, VerifierContextRecord] = {}
        for ev in events:
            ctx = vlog.context_for(ev.id)
            if ctx is None:
                continue
            out.setdefault(ev.candidate, ctx)
        return out
    finally:
        vlog.close()
