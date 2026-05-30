"""Hidden state ``s_t``.

At L3 ``s_t`` is a dual representation:

- **Text projection** — a compact natural-language summary the LLM can read in
  the system prompt. It is the primary, canonical form.
- **Vector** — the fastembed-encoded version of the text projection. Used as a
  retrieval-conditioning signal in L4+, and as the input to the Phase-2
  custom recurrent core (L14).

Both are persisted atomically: write to ``*.tmp``, then ``os.replace``. Reads
tolerate missing files (first run) and fall back to empty state.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from lsm.core.embeddings import Embedder
from lsm.core.facts import (
    CAT_USER_MEDIA,
    CAT_USER_PREFERENCES,
    Fact,
    FactStore,
    extract_facts_from_user_turn,
    format_structured_facts_block,
)
from lsm.core.identity import IdentityProfile
from lsm.core.memory import MemoryEntry, MemoryStore
from lsm.core.projection import FactProjection, ProjectionContext
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend

log = logging.getLogger(__name__)


@dataclass(slots=True)
class State:
    """Hidden state ``s_t`` (text + vector)."""

    text: str = ""
    vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    updated_at: float = 0.0
    revision: int = 0
    notes: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.text and self.vector.size == 0

    def project(self, max_chars: int = 1200) -> str:
        """Return a prompt-friendly textual projection."""
        if not self.text:
            return ""
        return self.text if len(self.text) <= max_chars else self.text[: max_chars - 1] + "…"


class StateStore:
    """Filesystem-backed store for one :class:`State` instance."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.json_path = self.root / "s_t.json"
        self.npy_path = self.root / "s_t.npy"

    def load(self) -> State:
        if not self.json_path.exists():
            return State()
        meta = json.loads(self.json_path.read_text(encoding="utf-8"))
        vec = (
            np.load(self.npy_path).astype(np.float32)
            if self.npy_path.exists()
            else np.zeros(0, dtype=np.float32)
        )
        return State(
            text=meta.get("text", ""),
            vector=vec,
            updated_at=float(meta.get("updated_at", 0.0)),
            revision=int(meta.get("revision", 0)),
            notes=dict(meta.get("notes", {})),
        )

    def save(self, state: State) -> None:
        tmp_json = self.json_path.with_suffix(".json.tmp")
        tmp_npy = self.npy_path.with_suffix(".npy.tmp")

        meta = {
            "text": state.text,
            "updated_at": state.updated_at,
            "revision": state.revision,
            "notes": state.notes,
            "vector_dim": int(state.vector.size),
        }
        tmp_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Use a file handle so numpy writes to this exact path.
        # np.save(path, ...) auto-appends ".npy" if the path doesn't end in it,
        # which would produce "s_t.npy.tmp.npy" and break the os.replace below.
        with open(tmp_npy, "wb") as f:
            np.save(f, state.vector, allow_pickle=False)
        os.replace(tmp_json, self.json_path)
        os.replace(tmp_npy, self.npy_path)

    def snapshot(self, tag: str) -> Path:
        """Snapshot the current state directory under ``snapshots/<tag>/``."""
        if not tag:
            raise ValueError("tag must be non-empty")
        dest = self.root / "snapshots" / tag
        if dest.exists():
            raise FileExistsError(f"snapshot {tag!r} already exists at {dest}")
        dest.mkdir(parents=True)
        for f in (self.json_path, self.npy_path):
            if f.exists():
                shutil.copy2(f, dest / f.name)
        return dest

    def restore(self, tag: str) -> None:
        src = self.root / "snapshots" / tag
        if not src.exists():
            raise FileNotFoundError(f"no snapshot {tag!r}")
        for f in ("s_t.json", "s_t.npy"):
            s = src / f
            if s.exists():
                shutil.copy2(s, self.root / f)

    def list_snapshots(self) -> list[str]:
        d = self.root / "snapshots"
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())


@runtime_checkable
class StateUpdater(Protocol):
    """Produces the next :class:`State` from the previous one plus the turn."""

    async def update(self, prev: State, user_msg: str, assistant_reply: str) -> State: ...


class NullStateUpdater:
    """Ablation: never updates state. ``s_t`` stays at its initial empty value."""

    async def update(self, prev: State, user_msg: str, assistant_reply: str) -> State:
        return prev


class PassthroughStateUpdater:
    """Cheap baseline: concatenate turn into the text and re-embed.

    Keeps at most ``max_chars`` of the most-recent content. Useful as a cost
    baseline against the LLM-written summarizer.
    """

    def __init__(self, embedder: Embedder, max_chars: int = 1200) -> None:
        self._embedder = embedder
        self._max_chars = max_chars

    async def update(self, prev: State, user_msg: str, assistant_reply: str) -> State:
        joined = f"{prev.text}\nuser: {user_msg}\nassistant: {assistant_reply}".strip()
        if len(joined) > self._max_chars:
            joined = joined[-self._max_chars :]
        vec = (
            self._embedder.embed(joined)
            if joined
            else np.zeros(self._embedder.dim, dtype=np.float32)
        )
        return State(
            text=joined,
            vector=vec,
            updated_at=time.time(),
            revision=prev.revision + 1,
            notes=prev.notes,
        )


class FactsBackedUpdater:
    """Deterministic updater: render ``s_t.text`` from structured facts.

    This is F.2's production default. It does not ask the LLM to author state.
    """

    def __init__(
        self,
        embedder: Embedder,
        projection: FactProjection,
        *,
        identity: IdentityProfile | None = None,
        memory: MemoryStore | None = None,
        fact_store: FactStore | None = None,
        correction_top_k: int = 4,
        plasticity_signal: Callable[[], float] | None = None,
        max_chars: int = 600,
    ) -> None:
        self._embedder = embedder
        self._projection = projection
        self._identity = identity
        self._memory = memory
        self._fact_store = fact_store
        self._correction_top_k = correction_top_k
        self._plasticity_signal = plasticity_signal
        self._max_chars = max_chars

    def _correction_entries(self) -> list[MemoryEntry]:
        if self._memory is None or self._correction_top_k <= 0:
            return []
        try:
            return self._memory.query_by_tag("correction", top_k=self._correction_top_k)
        except Exception:
            log.warning("failed to query correction memories", exc_info=True)
            return []

    async def update(self, prev: State, user_msg: str, assistant_reply: str) -> State:
        _ = assistant_reply  # kept for protocol compatibility.
        facts = self._fact_store.all_facts() if self._fact_store is not None else []
        notes = dict(prev.notes)
        rolling = notes.get("rolling", "")
        surprise = 1.0
        if self._plasticity_signal is not None:
            try:
                surprise = float(self._plasticity_signal())
            except Exception:
                surprise = 1.0
        is_correction = _looks_like_correction(user_msg)
        # Derive structural touched categories from the same extraction
        # rules the pipeline used. This is a cheap re-run (no DB writes)
        # but lets the projection's correction-reorder branch be domain-
        # agnostic instead of pattern-matching specific entity names.
        touched: frozenset[str] = frozenset()
        if is_correction and user_msg:
            try:
                touched = frozenset(f.category for f in extract_facts_from_user_turn(user_msg))
            except Exception:
                log.debug("touched-category derivation failed", exc_info=True)
                touched = frozenset()
        ctx = ProjectionContext(
            facts=facts,
            identity=self._identity,
            recent_corrections=self._correction_entries(),
            rolling_notes=rolling,
            correction_just_happened=is_correction,
            recent_surprise=surprise,
            max_chars=self._max_chars,
            touched_categories=touched,
        )
        text = (await self._projection.render(ctx)).strip()
        if not text:
            return prev
        vec = self._embedder.embed(text)
        return State(
            text=text,
            vector=vec,
            updated_at=time.time(),
            revision=prev.revision + 1,
            notes=notes,
        )


SUMMARIZER_SYSTEM = (
    "You maintain a compact running summary of a conversation between a user "
    "and an AI assistant. Merge the latest exchange into the prior summary.\n\n"
    "Source of truth:\n"
    "- Only the user's own messages are facts. The assistant's reply is an "
    "unverified paraphrase; treat it as facts only if a later user turn "
    "explicitly affirms it.\n"
    '- Facts in the "Known prior corrections" block (if present) override '
    "anything in the prior summary or assistant reply; never re-merge an "
    "older claim that a correction contradicts.\n"
    '- Facts in the "Structured facts" block (if present) are authoritative '
    "for user-owned slots; do not contradict or merge away listed facts.\n"
    "- Never treat the assistant's name, persona, or vocative (e.g. \"hi "
    'Elsum") as a user fact, product, game, person, or hobby.\n\n'
    "Slot definitions (omit empty bullets):\n"
    "- identity/role: the assistant's pinned identity only. Echo the pinned "
    "block verbatim when provided. Never the user's name.\n"
    "- preferences/likes: user tastes only. When the user says both X and Y, "
    "keep separate items (e.g. yakiniku and Korean BBQ are not the same).\n"
    "- dislikes: user dislikes only.\n"
    "- traits/personality: enduring descriptions about the user (creator, "
    "loves life). Not actionable help requests.\n"
    "- games/media: split title vs genre; never call a title a genre or vice "
    "versa; corrections replace prior genre claims.\n"
    "- people/names: humans the user mentions, including the user's own name. "
    "Never the assistant's name.\n"
    "- ongoing tasks: actionable help requests or open loops only (help me, "
    "remind me). Not personality, not the assistant's mission.\n"
    "- open questions\n\n"
    "Example: identity/role: Elsum (assistant) | people/names: Reishi (user) | "
    "traits/personality: creator\n\n"
    "Additive facts: when the user says both X and Y, or X and Y, or as well as, "
    "preserve separate items; never parenthesize one as the other.\n\n"
    "Correction handling: when the new user message contradicts a fact in "
    "the prior summary, replace the contradicted fact in place. Do not "
    "keep both versions.\n\n"
    "Style: \u2264 120 words. Short declarative sentences. Drop small talk. "
    "If unsure, omit. Never invent. Output only the summary, no preamble."
)

_IDENTITY_ROLE_LINE = re.compile(
    r"^identity/role\s*:\s*.*$",
    re.MULTILINE | re.IGNORECASE,
)

_CORRECTIONS_HEADER = "Known prior corrections (authoritative ground truth; never un-merge):"

_PEOPLE_NAMES_LINE = re.compile(
    r"^people/names\s*:\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)
_PREFERENCES_LINE = re.compile(
    r"^preferences/likes\s*:\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)
_GAMES_MEDIA_LINE = re.compile(
    r"^games/media\s*:\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)

_CORRECTION_PATTERNS = re.compile(
    r"\b("
    r"actually"
    r"|not\s+(?:a|an|the)?\s*"
    r"|i\s+meant"
    r"|correction"
    r"|to\s+clarify"
    r"|that'?s\s+wrong"
    r"|isn'?t"
    r"|please\s+make\s+it"
    r"|no,\s*please"
    r")\b",
    re.IGNORECASE,
)

_ADDITIVE_PATTERNS = re.compile(
    r"\b(" r"both\b" r"|as\s+well\s+as" r"|also\s+like" r")\b",
    re.IGNORECASE,
)


def _pin_identity_line(text: str, profile: IdentityProfile) -> str:
    """Force ``identity/role`` to the pinned profile (post-LLM safety net)."""
    line = profile.identity_role_line()
    if _IDENTITY_ROLE_LINE.search(text):
        return _IDENTITY_ROLE_LINE.sub(line, text, count=1)
    return f"{line}\n{text}" if text else line


def _format_corrections_block(entries: Sequence[MemoryEntry]) -> str:
    """Build the corrections prompt block from memory entries."""
    lines = [_CORRECTIONS_HEADER]
    for entry in entries:
        lines.append(f"- {entry.text}")
    return "\n".join(lines)


_QUOTED_REPLACEMENT = re.compile(
    r'(?:make\s+it|please\s+make\s+it)\s*["\']',
    re.IGNORECASE,
)


def _looks_like_correction(text: str) -> bool:
    """Heuristic: does this user turn read as a correction to a prior fact.

    Boolean only, no NLP. Used to add a single override instruction to the
    summarizer prompt so the LLM prefers the new user message over the prior
    summary where they conflict.
    """
    if not text:
        return False
    return bool(_CORRECTION_PATTERNS.search(text) or _QUOTED_REPLACEMENT.search(text))


def _looks_like_additive(text: str) -> bool:
    """Heuristic: user is adding another item, not replacing (both/and)."""
    if not text:
        return False
    return bool(
        _ADDITIVE_PATTERNS.search(text)
        or re.search(r"\b\w+\s+and\s+\w+\s+restaurants?", text, re.IGNORECASE)
        or re.search(r'make\s+it\s*["\'][^"\']*\band\b', text, re.IGNORECASE)
    )


def _sanitize_people_names(text: str, assistant_name: str) -> str:
    """Remove the assistant name from ``people/names`` if the LLM misplaced it."""
    if not assistant_name or not _PEOPLE_NAMES_LINE.search(text):
        return text

    def _repl(m: re.Match[str]) -> str:
        body = m.group(1)
        parts = [p.strip() for p in re.split(r",|\band\b", body) if p.strip()]
        cleaned: list[str] = []
        for p in parts:
            if assistant_name.lower() in p.lower() and "user" not in p.lower():
                continue
            cleaned.append(p)
        if not cleaned:
            return "people/names: (none)"
        return "people/names: " + ", ".join(cleaned)

    return _PEOPLE_NAMES_LINE.sub(_repl, text, count=1)


def _sanitize_slots_from_facts(text: str, facts: Sequence[Fact]) -> str:
    """Ensure structured facts appear in the matching ``s_t`` slots."""
    text = _merge_preference_facts_into_text(text, facts)
    return _merge_media_facts_into_text(text, facts)


def _merge_preference_facts_into_text(text: str, facts: Sequence[Fact]) -> str:
    pref_facts = [f for f in facts if f.category == CAT_USER_PREFERENCES]
    if not pref_facts:
        return text

    def _merge_prefs(m: re.Match[str]) -> str:
        existing = m.group(1).lower()
        missing = [f.value for f in pref_facts if f.value.strip().lower() not in existing]
        if not missing:
            return m.group(0)
        merged = m.group(1).rstrip()
        if merged:
            merged += ", " + ", ".join(missing)
        else:
            merged = ", ".join(f.value for f in pref_facts)
        return f"preferences/likes: {merged}"

    if _PREFERENCES_LINE.search(text):
        return _PREFERENCES_LINE.sub(_merge_prefs, text, count=1)
    lines = [f"preferences/likes: {', '.join(f.value for f in pref_facts)}"]
    return "\n".join(lines) + "\n" + text if text else lines[0]


def _merge_media_facts_into_text(text: str, facts: Sequence[Fact]) -> str:
    media_facts = [f for f in facts if f.category == CAT_USER_MEDIA and f.subject]
    if not media_facts:
        return text

    rendered = "games/media: " + "; ".join(f"{f.subject} ({f.value})" for f in media_facts)

    if _GAMES_MEDIA_LINE.search(text):
        text = _GAMES_MEDIA_LINE.sub(rendered, text, count=1)
    else:
        text = rendered + "\n" + text if text else rendered
    # Drop stale genre wording when facts say action RPG (correction path).
    for f in media_facts:
        if "action" in f.value.lower() and "rpg" in f.value.lower():
            text = re.sub(r"\bstrategy\s+game\b", f.value, text, flags=re.IGNORECASE)
    return text


class LLMSummarizerUpdater:
    """L3 default: ask the LLM itself to merge the new turn into ``s_t``.

    The same backend is reused (a fresh non-streaming call per turn). An
    embedder projects the text into the vector slot.

    Args:
        assistant_as_facts: if ``True``, the assistant reply is treated as
            authoritative (legacy behaviour, preserved for ablation runs).
            Default ``False``: only the user message is authoritative; the
            assistant reply is labelled "unverified paraphrase" in the prompt.
        identity: pinned assistant profile; when set, ``identity/role`` is
            injected into the prompt and forced in the output.
        memory: durable store used to pull correction-tagged entries (Phase A.7).
        correction_top_k: max correction memories to inject per turn.
        fact_store: structured facts ground truth (Phase F.1).
    """

    def __init__(
        self,
        backend: LLMBackend,
        embedder: Embedder,
        *,
        max_words: int = 120,
        options: GenerateOptions | None = None,
        assistant_as_facts: bool = False,
        identity: IdentityProfile | None = None,
        memory: MemoryStore | None = None,
        correction_top_k: int = 4,
        fact_store: FactStore | None = None,
        verifier: object | None = None,
        verifier_reprompt_enabled: bool = True,
        verifier_log: object | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._max_words = max_words
        self._options = options or GenerateOptions(temperature=0.2, max_tokens=256)
        self._assistant_as_facts = assistant_as_facts
        self._identity = identity
        self._memory = memory
        self._correction_top_k = correction_top_k
        self._fact_store = fact_store
        # ``verifier`` is typed as ``object`` here to keep state.py free of
        # an import on verifier.py at module load (state.py is imported by
        # the verifier indirectly via facts.py); ducktyping is enough.
        self._verifier = verifier
        self._verifier_reprompt_enabled = verifier_reprompt_enabled
        self._verifier_log = verifier_log

    def _correction_entries(self) -> list[MemoryEntry]:
        if self._memory is None or self._correction_top_k <= 0:
            return []
        try:
            return self._memory.query_by_tag("correction", top_k=self._correction_top_k)
        except Exception:
            log.warning("failed to query correction memories", exc_info=True)
            return []

    def _build_verifier_context(self, user_msg: str):
        """Build a :class:`VerifierContext` for the summarizer gate."""
        from lsm.core.verifier import VerifierContext  # local import: optional dep

        return VerifierContext(
            facts=self._fact_store,
            identity=self._identity,
            recent_corrections=tuple(self._correction_entries()),
            recent_user_turns=(user_msg,) if user_msg else (),
        )

    def _build_body(self, prev: State, user_msg: str, assistant_reply: str) -> str:
        if self._assistant_as_facts:
            return (
                "Prior summary:\n"
                f"{prev.text or '(empty)'}\n\n"
                "Latest exchange:\n"
                f"user: {user_msg}\n"
                f"assistant: {assistant_reply}\n\n"
                f"Produce the new summary (\u2264 {self._max_words} words)."
            )

        parts: list[str] = []
        if self._identity is not None:
            parts.extend(
                [
                    "Pinned identity (do NOT alter; echo verbatim as identity/role):",
                    f"name: {self._identity.name}",
                    f"description: {self._identity.description}",
                    "",
                ]
            )
        corrections = self._correction_entries()
        if corrections:
            parts.extend([_format_corrections_block(corrections), ""])
        if self._fact_store is not None:
            block = format_structured_facts_block(self._fact_store.all_facts())
            if block:
                parts.extend([block, ""])
        parts.extend(
            [
                "Prior summary:",
                prev.text or "(empty)",
                "",
                "New user message (authoritative):",
                user_msg or "(none)",
                "",
                (
                    "Assistant reply (unverified paraphrase \u2014 do not "
                    "treat as facts unless user has affirmed):"
                ),
                assistant_reply or "(none)",
            ]
        )
        if _looks_like_correction(user_msg):
            parts.extend(
                [
                    "",
                    (
                        "User is correcting prior summary; prefer the new "
                        "user message over prior summary where they conflict."
                    ),
                ]
            )
        if _looks_like_additive(user_msg):
            parts.extend(
                [
                    "",
                    (
                        "User is adding items (both/and); keep separate items in "
                        "preferences; do not synonym-collapse similar categories."
                    ),
                ]
            )
        parts.extend(
            [
                "",
                f"Produce the new summary (\u2264 {self._max_words} words).",
            ]
        )
        return "\n".join(parts)

    async def update(self, prev: State, user_msg: str, assistant_reply: str) -> State:
        body = self._build_body(prev, user_msg, assistant_reply)

        prompt_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=SUMMARIZER_SYSTEM),
            ChatMessage(role="user", content=body),
        ]
        try:
            new_text = (await self._backend.generate(prompt_messages, self._options)).strip()
        except Exception:
            log.warning("state updater failed; keeping prior state", exc_info=True)
            return prev

        if not new_text:
            return prev

        # V.1 summarizer gate. Up to one re-prompt with fix hints. If the
        # second draft is still rejected (or worse than the first), we
        # keep the prior s_t as the safest fallback.
        if self._verifier is not None:
            ctx = self._build_verifier_context(user_msg)
            verdict = self._verifier.score(new_text, ctx)  # type: ignore[attr-defined]
            reprompt_used = False
            if verdict.decision != "accept":
                if self._verifier_reprompt_enabled and verdict.fix_hints:
                    reprompt_used = True
                    bullets = "\n".join(f"- {h}" for h in verdict.fix_hints)
                    prompt_messages.append(
                        ChatMessage(
                            role="system",
                            content=(
                                "Your previous summary failed verification. "
                                "Fix these issues before re-emitting:\n"
                                f"{bullets}"
                            ),
                        )
                    )
                    try:
                        retry = (
                            await self._backend.generate(prompt_messages, self._options)
                        ).strip()
                    except Exception:
                        log.warning("summarizer re-prompt failed", exc_info=True)
                        retry = ""
                    if retry:
                        retry_verdict = self._verifier.score(retry, ctx)  # type: ignore[attr-defined]
                        if retry_verdict.decision == "accept":
                            new_text = retry
                            verdict = retry_verdict
                # Plan-spec semantics: if we're still not at ``accept`` after
                # at most one retry, keep prior s_t. Safer than committing a
                # known-bad draft when we have a fallback.
                if verdict.decision != "accept":
                    log.info(
                        "summarizer gate kept prior s_t (decision=%s, " "energy=%.2f)",
                        verdict.decision,
                        verdict.energy,
                    )
                    if self._verifier_log is not None:
                        try:
                            from lsm.core.verifier_log import build_context_record

                            event_id = str(uuid.uuid4())
                            self._verifier_log.append(  # type: ignore[attr-defined]
                                source="summarizer",
                                candidate=new_text,
                                verdict=verdict,
                                context=build_context_record(event_id, ctx),
                                reprompt_used=reprompt_used,
                                event_id=event_id,
                            )
                        except Exception:
                            log.debug("summarizer verifier telemetry failed", exc_info=True)
                    return prev
            if self._verifier_log is not None:
                try:
                    from lsm.core.verifier_log import build_context_record

                    event_id = str(uuid.uuid4())
                    self._verifier_log.append(  # type: ignore[attr-defined]
                        source="summarizer",
                        candidate=new_text,
                        verdict=verdict,
                        context=build_context_record(event_id, ctx),
                        reprompt_used=reprompt_used,
                        event_id=event_id,
                    )
                except Exception:
                    log.debug("summarizer verifier telemetry failed", exc_info=True)

        if self._identity is not None:
            new_text = _pin_identity_line(new_text, self._identity)
            new_text = _sanitize_people_names(new_text, self._identity.name)
        if self._fact_store is not None:
            new_text = _sanitize_slots_from_facts(new_text, self._fact_store.all_facts())

        vec = self._embedder.embed(new_text)
        return State(
            text=new_text,
            vector=vec,
            updated_at=time.time(),
            revision=prev.revision + 1,
            notes=prev.notes,
        )


def state_to_dict(s: State) -> dict[str, object]:
    """JSON-friendly view of a state (omits the raw vector)."""
    d = asdict(s)
    d.pop("vector", None)
    d["vector_dim"] = int(s.vector.size)
    return d
