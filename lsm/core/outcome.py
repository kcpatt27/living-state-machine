"""Outcome scoring loop — Stentor primitive #5.

After every assistant turn the pipeline calls an :class:`OutcomeScorer`
which rates how useful the response was on a ``[0, 1]`` scale. The score
(plus optional user feedback) feeds two downstream controls:

- **Writer gate.** The writer sees the critic score in
  :class:`~lsm.core.writer.WriteCandidate.self_critic_score`; high-trust
  turns are easier to keep, low-trust ones get dropped.
- **Plasticity scheduling (L10).** Per-entry trust values move toward the
  critic's opinion; slow-and-sure entries resist rewrites, fast entries
  churn.

Implementations
===============

- :class:`NullOutcomeScorer` — ablation; always returns ``0.5``.
- :class:`HeuristicOutcomeScorer` — cheap rule-based proxy. Penalises empty
  replies, literal copies of the user message, one-liner 'ok' responses,
  and rewards replies that cite retrieved content or mention numbers /
  names that appeared in the user turn.
- :class:`LLMSelfCriticScorer` — delegates to an LLM backend with a
  structured grading prompt and strict JSON parsing.

All implementations are pure; the log of outcomes lives in
:class:`OutcomeLog` (SQLite) so we can compute correlations after the fact.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend

log = logging.getLogger(__name__)


@dataclass(slots=True)
class OutcomeRecord:
    """One scored outcome."""

    turn_id: str
    session_id: str
    score: float
    source: str  # "heuristic" | "self_critic" | "user"
    reasons: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def trust_delta(self) -> float:
        """Centered score in ``[-0.5, +0.5]`` for plasticity updates."""
        return self.score - 0.5


@runtime_checkable
class OutcomeScorer(Protocol):
    """Grade an assistant reply."""

    name: str

    async def score(
        self,
        *,
        user_msg: str,
        assistant_reply: str,
        retrieved: Sequence[str] = (),
        state_projection: str = "",
    ) -> OutcomeRecord: ...


# ---------------------------------------------------------------------------
# Null
# ---------------------------------------------------------------------------


class NullOutcomeScorer:
    """Ablation: always returns neutral score, no side effects."""

    name = "null"

    async def score(
        self,
        *,
        user_msg: str,
        assistant_reply: str,
        retrieved: Sequence[str] = (),
        state_projection: str = "",
    ) -> OutcomeRecord:
        _ = user_msg, assistant_reply, retrieved, state_projection
        return OutcomeRecord(turn_id="", session_id="", score=0.5, source="null")


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HeuristicOutcomeConfig:
    min_reply_chars: int = 8
    min_reply_tokens: int = 3


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class HeuristicOutcomeScorer:
    """Fast, LLM-free outcome grader. Runs on every turn by default."""

    name = "heuristic"

    def __init__(self, config: HeuristicOutcomeConfig | None = None) -> None:
        self.config = config or HeuristicOutcomeConfig()

    async def score(
        self,
        *,
        user_msg: str,
        assistant_reply: str,
        retrieved: Sequence[str] = (),
        state_projection: str = "",
    ) -> OutcomeRecord:
        reply = assistant_reply or ""
        reasons: list[str] = []
        features: dict[str, float] = {}

        reply_tokens = _tokenize(reply)
        user_tokens = set(_tokenize(user_msg))
        retrieved_tokens = set(tok for r in retrieved for tok in _tokenize(r))
        state_tokens = set(_tokenize(state_projection))

        features["length_chars"] = float(len(reply))
        features["length_tokens"] = float(len(reply_tokens))

        if not reply.strip():
            reasons.append("empty_reply")
            return OutcomeRecord(
                turn_id="",
                session_id="",
                score=0.0,
                source=self.name,
                reasons=reasons,
                features=features,
            )

        if len(reply) < self.config.min_reply_chars:
            reasons.append("too_short")
        if len(reply_tokens) < self.config.min_reply_tokens:
            reasons.append("too_few_tokens")

        # Does the reply reuse content from the user or retrieved context?
        reply_set = set(reply_tokens)
        user_overlap = len(reply_set & user_tokens) / max(len(reply_set), 1)
        retrieved_overlap = len(reply_set & retrieved_tokens) / max(len(reply_set), 1)
        state_overlap = len(reply_set & state_tokens) / max(len(reply_set), 1)

        features["user_overlap"] = user_overlap
        features["retrieved_overlap"] = retrieved_overlap
        features["state_overlap"] = state_overlap

        # Exact echo of the user is a failure mode we want to punish.
        if user_msg and reply.strip().lower() == user_msg.strip().lower():
            reasons.append("echoes_user")

        if reply_set <= {"ok", "okay", "sure", "yes", "no", "thanks", "got", "it"}:
            reasons.append("filler_reply")

        # Base score = clipped logistic of length + retrieval reuse signal.
        length_score = min(len(reply_tokens) / 40.0, 1.0)
        grounding = 0.5 * retrieved_overlap + 0.2 * state_overlap
        score = 0.45 + 0.35 * length_score + 0.25 * grounding

        if "empty_reply" in reasons:
            score = 0.0
        if "echoes_user" in reasons:
            score *= 0.3
        if "filler_reply" in reasons:
            score *= 0.3
        if "too_short" in reasons:
            score *= 0.7
        if "too_few_tokens" in reasons:
            score *= 0.7

        score = max(0.0, min(1.0, score))
        return OutcomeRecord(
            turn_id="",
            session_id="",
            score=float(score),
            source=self.name,
            reasons=reasons,
            features=features,
        )


# ---------------------------------------------------------------------------
# LLM self-critic
# ---------------------------------------------------------------------------


_SELF_CRITIC_SYSTEM = (
    "You are a strict reviewer grading an AI assistant's reply for "
    "usefulness and factual grounding. Respond with ONLY a JSON object of "
    'the form {"score": 0.0 to 1.0, "reason": "..."}. Scores above 0.7 '
    "require that the reply clearly answers the user, uses the retrieved "
    "context where relevant, and is factually consistent with it. Scores "
    "below 0.3 indicate empty, evasive, or wrong replies. Do not include "
    "any text outside the JSON object."
)


class LLMSelfCriticScorer:
    """Delegates grading to a frozen LLM backend with a structured prompt."""

    name = "self_critic"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        max_tokens: int = 120,
        temperature: float = 0.0,
        fallback: OutcomeScorer | None = None,
    ) -> None:
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback = fallback or HeuristicOutcomeScorer()

    async def score(
        self,
        *,
        user_msg: str,
        assistant_reply: str,
        retrieved: Sequence[str] = (),
        state_projection: str = "",
    ) -> OutcomeRecord:
        prompt_parts: list[str] = []
        if state_projection:
            prompt_parts.append(f"[State projection]\n{state_projection}")
        if retrieved:
            joined = "\n- " + "\n- ".join(retrieved)
            prompt_parts.append(f"[Retrieved memories]{joined}")
        prompt_parts.append(f"[User]\n{user_msg}")
        prompt_parts.append(f"[Assistant]\n{assistant_reply}")
        user_content = "\n\n".join(prompt_parts)

        messages = [
            ChatMessage(role="system", content=_SELF_CRITIC_SYSTEM),
            ChatMessage(role="user", content=user_content),
        ]
        options = GenerateOptions(max_tokens=self.max_tokens, temperature=self.temperature)
        try:
            raw = await self.backend.generate(messages, options)
        except Exception:  # noqa: BLE001
            log.exception("self-critic backend call failed; falling back")
            return await self.fallback.score(
                user_msg=user_msg,
                assistant_reply=assistant_reply,
                retrieved=retrieved,
                state_projection=state_projection,
            )

        score, reason = _parse_self_critic_json(raw)
        if score is None:
            log.warning("self-critic produced unparseable output: %r", raw[:200])
            return await self.fallback.score(
                user_msg=user_msg,
                assistant_reply=assistant_reply,
                retrieved=retrieved,
                state_projection=state_projection,
            )
        return OutcomeRecord(
            turn_id="",
            session_id="",
            score=float(score),
            source=self.name,
            reasons=[reason] if reason else [],
            features={},
        )


_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_self_critic_json(text: str) -> tuple[float | None, str | None]:
    if not text:
        return None, None
    match = _JSON_BLOCK.search(text)
    if not match:
        return None, None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None
    try:
        score = float(payload.get("score"))
    except (TypeError, ValueError):
        return None, None
    if score < 0.0 or score > 1.0:
        return None, None
    reason = payload.get("reason")
    return score, str(reason) if reason is not None else None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class OutcomeLog:
    """Append-only SQLite log of outcome records.

    Lives in the state directory next to the episodic log. Designed for
    analysis after the fact (correlations between critic score and downstream
    task accuracy).
    """

    SCHEMA = (
        "CREATE TABLE IF NOT EXISTS outcomes ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " turn_id TEXT,"
        " session_id TEXT,"
        " score REAL,"
        " source TEXT,"
        " reasons_json TEXT,"
        " features_json TEXT,"
        " created_at REAL"
        ")"
    )

    def __init__(self, path: Path) -> None:
        import sqlite3

        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(self.SCHEMA)
        self._conn.commit()

    def append(self, record: OutcomeRecord) -> int:
        cur = self._conn.execute(
            "INSERT INTO outcomes (turn_id, session_id, score, source, reasons_json,"
            " features_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.turn_id,
                record.session_id,
                float(record.score),
                record.source,
                json.dumps(record.reasons),
                json.dumps(record.features),
                float(record.created_at),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def extend(self, records: Iterable[OutcomeRecord]) -> None:
        for r in records:
            self.append(r)

    def session(self, session_id: str, *, limit: int = 100) -> list[OutcomeRecord]:
        rows = self._conn.execute(
            "SELECT turn_id, session_id, score, source, reasons_json, features_json,"
            " created_at FROM outcomes WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, int(limit)),
        ).fetchall()
        return [
            OutcomeRecord(
                turn_id=r[0],
                session_id=r[1],
                score=float(r[2]),
                source=r[3],
                reasons=json.loads(r[4] or "[]"),
                features=json.loads(r[5] or "{}"),
                created_at=float(r[6]),
            )
            for r in rows
        ]

    def mean_score(self, session_id: str | None = None) -> float:
        if session_id is None:
            row = self._conn.execute("SELECT AVG(score) FROM outcomes").fetchone()
        else:
            row = self._conn.execute(
                "SELECT AVG(score) FROM outcomes WHERE session_id = ?", (session_id,)
            ).fetchone()
        return float(row[0] or 0.0)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_scorer(kind: str, **kwargs: object) -> OutcomeScorer:
    """Factory. ``kind`` is one of: null, heuristic, self_critic."""
    kind = kind.lower()
    if kind in ("none", "null", "disabled"):
        return NullOutcomeScorer()
    if kind == "heuristic":
        return HeuristicOutcomeScorer(HeuristicOutcomeConfig(**kwargs))  # type: ignore[arg-type]
    if kind == "self_critic":
        backend = kwargs.pop("backend", None)
        if backend is None:
            raise ValueError("self_critic scorer requires a 'backend' kwarg")
        return LLMSelfCriticScorer(backend=backend, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown outcome scorer kind: {kind!r}")


__all__ = [
    "HeuristicOutcomeConfig",
    "HeuristicOutcomeScorer",
    "LLMSelfCriticScorer",
    "NullOutcomeScorer",
    "OutcomeLog",
    "OutcomeRecord",
    "OutcomeScorer",
    "build_scorer",
]


# Make dataclass asdict handy for API layer.
def record_to_dict(record: OutcomeRecord) -> dict[str, Any]:
    return asdict(record)
