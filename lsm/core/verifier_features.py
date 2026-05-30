"""Shared feature extraction for learned verifier models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from lsm.core.facts import Fact
from lsm.core.verifier import RuleBasedVerifier, Verdict, VerifierContext, ViolationKind

_VIOLATION_KINDS: tuple[ViolationKind, ...] = (
    "fact_contradiction",
    "identity_drift",
    "correction_violation",
    "list_merge",
    "length_overflow",
)

_SOURCE_BUCKETS: tuple[str, ...] = ("writer", "reply", "stream", "summarizer", "manual")


def feature_names() -> list[str]:
    """Stable feature order used by training and inference."""
    names: list[str] = []
    for kind in _VIOLATION_KINDS:
        names.append(f"violation_present:{kind}")
    for kind in _VIOLATION_KINDS:
        names.append(f"violation_count:{kind}")
    names.extend(
        [
            "candidate_chars",
            "candidate_lines",
            "slot_line_count",
            "facts_count",
            "recent_corrections_count",
            "recent_user_turns_count",
            "has_identity",
            "has_fact_subject_match",
            "rule_energy",
            "rule_is_accept",
            "rule_is_reprompt",
            "rule_is_reject",
        ]
    )
    for bucket in _SOURCE_BUCKETS:
        names.append(f"source_is:{bucket}")
    names.append("source_is:other")
    return names


def extract_features(
    candidate: str,
    ctx: VerifierContext,
    *,
    source: str = "manual",
    rule_verdict: Verdict | None = None,
) -> np.ndarray:
    """Extract deterministic numeric features from candidate + context."""
    verdict = rule_verdict or RuleBasedVerifier().score(candidate, ctx)
    v_counts = Counter(v.kind for v in verdict.violations)
    facts = _safe_all_facts(ctx)
    source_key = source.strip().lower()
    lines = candidate.splitlines()
    slot_lines = sum(1 for line in lines if ":" in line)
    has_subject = _has_fact_subject_match(candidate, facts)
    values: list[float] = []
    for kind in _VIOLATION_KINDS:
        values.append(1.0 if v_counts.get(kind, 0) > 0 else 0.0)
    for kind in _VIOLATION_KINDS:
        values.append(float(v_counts.get(kind, 0)))
    values.extend(
        [
            float(len(candidate)),
            float(len(lines) or 1),
            float(slot_lines),
            float(len(facts)),
            float(len(ctx.recent_corrections)),
            float(len(ctx.recent_user_turns)),
            1.0 if ctx.identity is not None else 0.0,
            1.0 if has_subject else 0.0,
            float(verdict.energy),
            1.0 if verdict.decision == "accept" else 0.0,
            1.0 if verdict.decision == "reprompt" else 0.0,
            1.0 if verdict.decision == "reject" else 0.0,
        ]
    )
    for bucket in _SOURCE_BUCKETS:
        values.append(1.0 if source_key == bucket else 0.0)
    values.append(0.0 if source_key in _SOURCE_BUCKETS else 1.0)
    return np.asarray(values, dtype=np.float32)


def feature_dict(
    candidate: str,
    ctx: VerifierContext,
    *,
    source: str = "manual",
    rule_verdict: Verdict | None = None,
) -> dict[str, float]:
    """Dictionary form of :func:`extract_features` keyed by feature name."""
    vec = extract_features(candidate, ctx, source=source, rule_verdict=rule_verdict)
    return {k: float(v) for k, v in zip(feature_names(), vec.tolist(), strict=True)}


def _safe_all_facts(ctx: VerifierContext) -> list[Fact]:
    if ctx.facts is None:
        return []
    try:
        return list(ctx.facts.all_facts())
    except Exception:
        return []


def _has_fact_subject_match(candidate: str, facts: Sequence[Fact]) -> bool:
    low = candidate.lower()
    for fact in facts:
        subj = (fact.subject or "").strip().lower()
        if subj and subj in low:
            return True
    return False
