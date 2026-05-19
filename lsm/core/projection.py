"""Facts-backed state projection (F.2).

Deterministic rendering is the default path: build ``s_t.text`` from facts,
identity, and light dynamic signals (recency, correction, surprise) without
asking an LLM to author state.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from lsm.core.facts import (
    CAT_ASSISTANT_NOTES,
    CAT_USER_DISLIKES,
    CAT_USER_MEDIA,
    CAT_USER_OPEN_QUESTIONS,
    CAT_USER_PEOPLE,
    CAT_USER_PREFERENCES,
    CAT_USER_TASKS,
    CAT_USER_TRAITS,
    Fact,
)
from lsm.core.identity import IdentityProfile
from lsm.core.memory import MemoryEntry
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend

log = logging.getLogger(__name__)

# Maps fact categories to projection slot names. Used when a correction
# turn happens to float the affected slots earlier in the render order
# without hardcoding domain entities (Crimson Desert, yakiniku, ...).
_SLOT_BY_CATEGORY: dict[str, str] = {
    CAT_USER_PEOPLE: "people/names",
    CAT_USER_PREFERENCES: "preferences/likes",
    CAT_USER_DISLIKES: "dislikes",
    CAT_USER_TRAITS: "traits/personality",
    CAT_USER_MEDIA: "games/media",
    CAT_USER_TASKS: "ongoing tasks",
    CAT_USER_OPEN_QUESTIONS: "open questions",
    CAT_ASSISTANT_NOTES: "assistant notes",
}


@dataclass(slots=True, frozen=True)
class ProjectionContext:
    """Inputs used to render ``s_t.text``."""

    facts: list[Fact]
    identity: IdentityProfile | None = None
    recent_corrections: list[MemoryEntry] | None = None
    rolling_notes: str = ""
    correction_just_happened: bool = False
    recent_surprise: float = 1.0
    max_chars: int = 600
    # Categories whose facts changed on the most recent turn. Used by the
    # correction reorder branch to float touched slots upward. Carry the
    # set of fact-category strings (e.g. ``CAT_USER_MEDIA``) rather than
    # slot names so the source is structural, not presentation-layer.
    touched_categories: frozenset[str] = field(default_factory=frozenset)


@runtime_checkable
class FactProjection(Protocol):
    """Render ``s_t.text`` from structured inputs."""

    async def render(self, ctx: ProjectionContext) -> str: ...


def _render_value(value: str, *, emphasize: bool) -> str:
    v = value.strip()
    if not v:
        return ""
    return f"*{v}" if emphasize else v


class DeterministicProjection:
    """Deterministic ``FactProjection`` implementation.

    Dynamic behavior:
    - Facts updated in the recency window are emphasized with a leading ``*``.
    - If this turn is a correction, correction-heavy slots are floated earlier.
    - If surprise is above threshold, lower-priority slots are dropped first.
    - Hard budget trims from the bottom while preserving identity/people/media.
    """

    def __init__(
        self,
        *,
        recency_window_s: float = 60.0,
        surprise_trim_threshold: float = 1.5,
        notes_max_chars: int = 150,
    ) -> None:
        self._recency_window_s = recency_window_s
        self._surprise_trim_threshold = surprise_trim_threshold
        self._notes_max_chars = notes_max_chars

    async def render(self, ctx: ProjectionContext) -> str:
        now = time.time()
        by_cat: dict[str, list[Fact]] = {}
        for fact in ctx.facts:
            by_cat.setdefault(fact.category, []).append(fact)

        # Stable recency ordering: most recently updated facts first. The
        # recency-window check is applied separately to drive emphasis.
        for facts in by_cat.values():
            facts.sort(key=lambda f: f.updated_at, reverse=True)

        lines: dict[str, str] = {}
        if ctx.identity is not None:
            lines["identity/role"] = ctx.identity.identity_role_line()

        people = by_cat.get(CAT_USER_PEOPLE, [])
        if people:
            vals = self._dedupe_values(
                _render_value(
                    f.value,
                    emphasize=f.updated_at >= now - self._recency_window_s,
                )
                for f in people
            )
            if vals:
                lines["people/names"] = f"people/names: {', '.join(vals)}"

        prefs = by_cat.get(CAT_USER_PREFERENCES, [])
        if prefs:
            grouped: dict[str, list[str]] = {}
            general: list[str] = []
            for f in prefs:
                val = _render_value(
                    f.value,
                    emphasize=f.updated_at >= now - self._recency_window_s,
                )
                if not val:
                    continue
                if f.subject:
                    grouped.setdefault(f.subject, []).append(val)
                else:
                    general.append(val)
            chunks: list[str] = []
            for subject, vals in grouped.items():
                dvals = self._dedupe_values(vals)
                if dvals:
                    chunks.append(f"{subject}: {', '.join(dvals)}")
            dgeneral = self._dedupe_values(general)
            chunks.extend(dgeneral)
            if chunks:
                lines["preferences/likes"] = f"preferences/likes: {'; '.join(chunks)}"

        dislikes = by_cat.get(CAT_USER_DISLIKES, [])
        if dislikes:
            vals = self._dedupe_values(
                _render_value(
                    f.value,
                    emphasize=f.updated_at >= now - self._recency_window_s,
                )
                for f in dislikes
            )
            if vals:
                lines["dislikes"] = f"dislikes: {', '.join(vals)}"

        traits = by_cat.get(CAT_USER_TRAITS, [])
        if traits:
            vals = self._dedupe_values(
                _render_value(
                    f.value,
                    emphasize=f.updated_at >= now - self._recency_window_s,
                )
                for f in traits
            )
            if vals:
                lines["traits/personality"] = f"traits/personality: {', '.join(vals)}"

        media = [f for f in by_cat.get(CAT_USER_MEDIA, []) if f.subject]
        if media:
            vals = self._dedupe_values(
                _render_value(
                    f"{f.subject} ({f.value})",
                    emphasize=f.updated_at >= now - self._recency_window_s,
                )
                for f in media
            )
            if vals:
                lines["games/media"] = f"games/media: {'; '.join(vals)}"

        tasks = by_cat.get(CAT_USER_TASKS, [])
        if tasks:
            vals = self._dedupe_values(
                _render_value(
                    f.value,
                    emphasize=f.updated_at >= now - self._recency_window_s,
                )
                for f in tasks
            )
            if vals:
                lines["ongoing tasks"] = f"ongoing tasks: {', '.join(vals)}"

        open_questions = by_cat.get(CAT_USER_OPEN_QUESTIONS, [])
        if open_questions:
            vals = self._dedupe_values(
                _render_value(
                    f.value,
                    emphasize=f.updated_at >= now - self._recency_window_s,
                )
                for f in open_questions
            )
            if vals:
                lines["open questions"] = f"open questions: {'; '.join(vals)}"

        assistant_notes = by_cat.get(CAT_ASSISTANT_NOTES, [])
        if assistant_notes:
            vals = self._dedupe_values(
                _render_value(
                    f.value,
                    emphasize=f.updated_at >= now - self._recency_window_s,
                )
                for f in assistant_notes
            )
            if vals:
                lines["assistant notes"] = f"assistant notes: {'; '.join(vals)}"

        rolling = ctx.rolling_notes.strip()
        if rolling:
            if len(rolling) > self._notes_max_chars:
                rolling = rolling[: self._notes_max_chars - 1] + "…"
            lines["notes"] = f"notes: {rolling}"

        order = [
            "identity/role",
            "people/names",
            "preferences/likes",
            "dislikes",
            "traits/personality",
            "games/media",
            "ongoing tasks",
            "open questions",
            "assistant notes",
            "notes",
        ]

        if ctx.correction_just_happened:
            correction_slots = {
                _SLOT_BY_CATEGORY[cat] for cat in ctx.touched_categories if cat in _SLOT_BY_CATEGORY
            }
            dynamic = [slot for slot in order if slot != "identity/role"]
            original_rank = {slot: idx for idx, slot in enumerate(dynamic)}
            dynamic.sort(key=lambda s: (s not in correction_slots, original_rank[s]))
            order = ["identity/role", *dynamic]

        if ctx.recent_surprise > self._surprise_trim_threshold:
            lines.pop("traits/personality", None)
            lines.pop("notes", None)

        rendered = [lines[s] for s in order if s in lines]
        if not rendered:
            return ""

        protected = {"identity/role", "people/names", "games/media"}
        line_by_slot = {slot: line for slot, line in lines.items()}
        removable = [s for s in reversed(order) if s not in protected and s in line_by_slot]

        text = "\n".join(rendered)
        while len(text) > ctx.max_chars and removable:
            drop = removable.pop(0)
            line_by_slot.pop(drop, None)
            rendered = [line_by_slot[s] for s in order if s in line_by_slot]
            text = "\n".join(rendered)
        if len(text) > ctx.max_chars:
            log.warning(
                "projection exceeded max_chars=%d (got %d); protected slots " "alone over budget",
                ctx.max_chars,
                len(text),
            )
        return text

    @staticmethod
    def _dedupe_values(values: Iterable[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            v = str(raw).strip()
            if not v:
                continue
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out


class LLMPolishedProjection:
    """Optional rephrase pass over deterministic projection.

    This wrapper never introduces new facts. If the polished text does not
    preserve the same slot->value payload as the deterministic body, it falls
    back to the deterministic body.
    """

    def __init__(
        self,
        *,
        base: FactProjection,
        backend: LLMBackend,
        enabled: bool = False,
        options: GenerateOptions | None = None,
    ) -> None:
        self._base = base
        self._backend = backend
        self._enabled = enabled
        self._options = options or GenerateOptions(temperature=0.0, max_tokens=192)

    async def render(self, ctx: ProjectionContext) -> str:
        base = (await self._base.render(ctx)).strip()
        if not base or not self._enabled:
            return base
        prompt = (
            "Rephrase for fluency while keeping exactly the same slots and facts.\n"
            "Rules:\n"
            "- Do not add or remove slots.\n"
            "- Do not add, remove, or change any fact values.\n"
            "- Keep one line per slot in '<slot>: <value>' format.\n"
            "- Keep the same slot order.\n\n"
            "Text:\n"
            f"{base}"
        )
        try:
            polished = (
                await self._backend.generate(
                    [
                        ChatMessage(role="system", content="You are a strict reformatter."),
                        ChatMessage(role="user", content=prompt),
                    ],
                    self._options,
                )
            ).strip()
        except Exception:
            return base
        return polished if self._same_payload(base, polished) else base

    @staticmethod
    def _same_payload(a: str, b: str) -> bool:
        pa = LLMPolishedProjection._parse_payload(a)
        pb = LLMPolishedProjection._parse_payload(b)
        return pa == pb

    @staticmethod
    def _normalize_token(s: str) -> str:
        """Normalize a slot or value for content-equality comparison.

        We strip ``*`` recency markers wherever they appear and collapse
        whitespace so a polished rephrase that drops the asterisk artifact
        still counts as the same payload. We intentionally do NOT strip
        punctuation broadly -- a polish that introduces or removes a comma
        between values is a real content change.
        """
        out = s.replace("*", " ").strip().lower()
        # Collapse internal runs of whitespace to a single space.
        out = " ".join(out.split())
        return out

    @staticmethod
    def _parse_payload(text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for line in text.splitlines():
            if ":" not in line:
                continue
            slot, value = line.split(":", 1)
            out.append(
                (
                    LLMPolishedProjection._normalize_token(slot),
                    LLMPolishedProjection._normalize_token(value),
                )
            )
        return out
