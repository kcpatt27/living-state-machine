"""Controlled synthetic negatives for verifier distillation."""

from __future__ import annotations

import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from lsm.core.verifier_log import VerifierContextRecord, VerifierEvent

_ALNUM_WORD = re.compile(r"\b([A-Za-z][A-Za-z0-9_\-]{2,})\b")


@dataclass(slots=True)
class PerturbationExample:
    """Synthetic candidate + label generated from an observed event."""

    event_id: str
    source: str
    candidate: str
    label: str
    label_source: str = "synthetic_perturb"
    context: VerifierContextRecord | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def perturb_from_events(
    events: list[tuple[VerifierEvent, VerifierContextRecord | None]],
    *,
    limit: int = 500,
    seed: int = 0,
) -> list[PerturbationExample]:
    """Generate synthetic negatives from accepted / low-energy rows."""
    rng = random.Random(seed)
    out: list[PerturbationExample] = []
    for ev, ctx in events:
        if ev.decision != "accept":
            continue
        mutated = _mutate_candidate(ev.candidate, ctx, rng)
        if not mutated or mutated == ev.candidate:
            continue
        out.append(
            PerturbationExample(
                event_id=f"perturb-{ev.id}-{uuid.uuid4().hex[:8]}",
                source=ev.source,
                candidate=mutated,
                label="reject",
                context=ctx,
                meta={"parent_event_id": ev.id},
            )
        )
        if len(out) >= max(1, limit):
            break
    return out


def _mutate_candidate(
    candidate: str,
    ctx: VerifierContextRecord | None,
    rng: random.Random,
) -> str:
    if not candidate.strip():
        return ""
    # 1) Fact-value swap if a known fact appears in the candidate.
    if ctx is not None:
        for fact in ctx.facts_snapshot:
            subject = str(fact.get("subject", "")).strip()
            value = str(fact.get("value", "")).strip()
            if not subject or not value:
                continue
            if subject.lower() not in candidate.lower():
                continue
            if value.lower() not in candidate.lower():
                continue
            wrong = _replace_value(value, rng)
            return re.sub(re.escape(value), wrong, candidate, count=1, flags=re.IGNORECASE)
        # 2) Identity drift injection.
        if ctx.identity_snapshot is not None:
            name = str(ctx.identity_snapshot.get("name", "")).strip()
            if name:
                return f"I am {name}. {candidate}"
    # 3) List merge drop: keep one side of "A and B".
    m = re.search(
        r"\b([A-Za-z][A-Za-z0-9'\-]{1,30})\s+and\s+([A-Za-z][A-Za-z0-9'\-]{1,30})\b", candidate
    )
    if m:
        a, b = m.group(1), m.group(2)
        keep = a if rng.random() < 0.5 else b
        return re.sub(rf"\b{re.escape(a)}\s+and\s+{re.escape(b)}\b", keep, candidate, count=1)
    # 4) Fallback generic contradiction suffix.
    token = next(iter(_ALNUM_WORD.findall(candidate)), "that")
    return f"{candidate} {token} is not correct."


def _replace_value(value: str, rng: random.Random) -> str:
    words = _ALNUM_WORD.findall(value)
    if not words:
        return f"not-{value}"
    first = words[0]
    return value.replace(first, f"{first}{rng.randint(100, 999)}", 1)
