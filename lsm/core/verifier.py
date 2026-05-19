"""V.1 Verifier layer.

A :class:`Verifier` scores a candidate string against a context and returns a
:class:`Verdict` (energy + violations + decision). Verifiers are the project's
first explicit scoring-not-authoring layer; the architectural framing lives in
``memory-bank/research-notes.md``.

This module ships:

- :class:`Verifier` Protocol (single ``score`` method).
- :class:`NullVerifier` — always accept; CI default, ablation baseline.
- :class:`RuleBasedVerifier` — deterministic sum of five weighted energy
  components (fact contradiction, identity drift, correction violation,
  list merge, length overflow). No extra LLM call.

V.1 does not ship a learned scorer. The Protocol shape is identical to what a
learned scorer would expose, so a contrastive / EBM-style scorer can drop in
later as a clean ablation. Telemetry (sqlite log) is deferred to a follow-on.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from lsm.core.facts import (
    CAT_USER_MEDIA,
    CAT_USER_PREFERENCES,
    MAX_FACT_VALUE_LEN,
    Fact,
    FactStore,
)
from lsm.core.identity import IdentityProfile
from lsm.core.memory import MemoryEntry
from lsm.core.normalize import TermNormalizer, clean_term

ViolationKind = Literal[
    "fact_contradiction",
    "identity_drift",
    "correction_violation",
    "list_merge",
    "length_overflow",
]

DecisionKind = Literal["accept", "reprompt", "reject"]

DEFAULT_WEIGHTS: dict[ViolationKind, float] = {
    "fact_contradiction": 1.5,
    "identity_drift": 1.5,
    "correction_violation": 1.5,
    "list_merge": 0.5,
    "length_overflow": 0.2,
}


@dataclass(frozen=True, slots=True)
class Violation:
    """One reason a candidate failed verification."""

    kind: ViolationKind
    detail: str
    weight: float
    evidence: str = ""

    def fix_hint(self) -> str:
        """Human-readable repair instruction passed back to the author."""
        return f"[{self.kind}] {self.detail}"


@dataclass(frozen=True, slots=True)
class VerifierContext:
    """Inputs the verifier scores against."""

    facts: FactStore | None = None
    identity: IdentityProfile | None = None
    recent_corrections: Sequence[MemoryEntry] = ()
    recent_user_turns: Sequence[str] = ()
    normalizer: TermNormalizer | None = None


@dataclass(frozen=True, slots=True)
class Verdict:
    """Verifier output: energy, violations, decision, fix hints."""

    energy: float
    violations: tuple[Violation, ...] = ()
    decision: DecisionKind = "accept"
    fix_hints: tuple[str, ...] = ()


@runtime_checkable
class Verifier(Protocol):
    """Score a candidate against a context."""

    def score(self, candidate: str, ctx: VerifierContext) -> Verdict: ...


class NullVerifier:
    """Always accept. CI default; ablation baseline."""

    def score(self, candidate: str, ctx: VerifierContext) -> Verdict:
        _ = candidate, ctx
        return Verdict(energy=0.0)


# --------------------------------------------------------------------------- #
# Rule-based verifier
# --------------------------------------------------------------------------- #


_ASSERTION_AFTER_SUBJECT = re.compile(
    # Matches subject-asserts-value separators:
    #   "Crimson Desert is an action RPG"
    #   "Crimson Desert = action RPG"
    #   "Crimson Desert: action RPG"
    #   "Crimson Desert (action RPG)"        <- F.2 DeterministicProjection
    # The article group requires trailing whitespace so it doesn't eat the
    # leading letter of words like "action" (which previously left "ction").
    r"(?:is|was|are|were|=|:|\()\s*(?:(?:a|an|the)\s+)?([^.?!,;:)\n]{2,80})",
    re.IGNORECASE,
)

# Identity ownership leaders: phrases that put the following noun in the
# user's possession. If the candidate has "<assistant_name>" right after one
# of these, the candidate is misattributing the assistant identity.
_IDENTITY_OWNERSHIP_LEADERS = (
    "i am",
    "i'm",
    "my name is",
    "my friend",
    "i like",
    "i love",
    "i enjoy",
    "my partner",
    "my pet",
)

# Correction memory parsing: "<term> is not <stale>" / "actually <canon>".
_NOT_PHRASE = re.compile(
    r"\bnot\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9\-'\s]{2,40})",
    re.IGNORECASE,
)
_IT_IS_PHRASE = re.compile(
    r"\b(?:it(?:'?s|\s+is|\s+was)|actually(?:\s+a|an|the)?)\s+(?:a|an|the)?\s*"
    r"([A-Za-z][A-Za-z0-9\-'\s]{2,40})",
    re.IGNORECASE,
)

# List-merge: paired-and phrases in a recent user turn.
_PAIRED_AND = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\-']{1,30})\s+and\s+([A-Za-z][A-Za-z0-9\-']{1,30})",
    re.IGNORECASE,
)

# Slot lines for length overflow check ("slot: value").
_SLOT_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9\s/_-]{1,30}):\s*(.+?)\s*$")

# Stop tokens that should never count as a "preserved item" by themselves
# when checking list-merge. Avoids spurious passes for things like "and a".
_LIST_MERGE_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "i",
        "we",
        "you",
        "he",
        "she",
        "it",
        "to",
        "of",
        "for",
        "with",
    }
)


def _canon(value: str, normalizer: TermNormalizer | None, scope: str) -> str:
    """Best-effort canonical form for a phrase."""
    base = clean_term(value)
    if not base:
        return ""
    if normalizer is None:
        return base.lower()
    try:
        return normalizer.canonicalize(scope, base).lower()
    except Exception:
        return base.lower()


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^\w]+", s.lower()) if t}


def _value_present_in(phrase: str, value: str) -> bool:
    """Crude content match: every non-stop token of ``value`` appears in ``phrase``."""
    val_tokens = _tokens(value) - {"a", "an", "the"}
    if not val_tokens:
        return False
    phrase_tokens = _tokens(phrase)
    return val_tokens.issubset(phrase_tokens)


def _check_fact_contradiction(
    candidate: str,
    facts: Sequence[Fact],
    normalizer: TermNormalizer | None,
) -> list[Violation]:
    out: list[Violation] = []
    low = candidate.lower()
    for fact in facts:
        if fact.category not in (CAT_USER_MEDIA, CAT_USER_PREFERENCES):
            continue
        subject = (fact.subject or "").strip()
        if not subject:
            continue
        subj_low = subject.lower()
        # Subject must appear in candidate to be at risk of contradiction.
        if subj_low not in low:
            continue
        # Find every assertion immediately following the subject mention.
        pattern = re.compile(
            re.escape(subject) + r"\s+" + _ASSERTION_AFTER_SUBJECT.pattern,
            re.IGNORECASE,
        )
        canon_value = _canon(fact.value, normalizer, scope=fact.category)
        contradicted = False
        for m in pattern.finditer(candidate):
            tail = m.group(1).strip()
            if not tail:
                continue
            tail_canon = _canon(tail, normalizer, scope=fact.category)
            if tail_canon == canon_value:
                contradicted = False
                break
            if _value_present_in(tail, fact.value):
                contradicted = False
                break
            contradicted = True
        if contradicted:
            out.append(
                Violation(
                    kind="fact_contradiction",
                    detail=(
                        f"{subject}: candidate asserts a value that "
                        f"contradicts facts ({fact.value!r})"
                    ),
                    weight=DEFAULT_WEIGHTS["fact_contradiction"],
                    evidence=fact.id,
                )
            )
    return out


def _check_identity_drift(
    candidate: str,
    identity: IdentityProfile | None,
) -> list[Violation]:
    if identity is None or not identity.name:
        return []
    name = identity.name.strip()
    if not name:
        return []
    low = candidate.lower()
    name_low = name.lower()
    out: list[Violation] = []
    for leader in _IDENTITY_OWNERSHIP_LEADERS:
        # leader + small filler + name pattern (e.g. "i'm Elsum", "my friend Elsum")
        pat = re.compile(
            r"\b"
            + re.escape(leader)
            + r"\b[\s,]+(?:a|an|the|named|called)?\s*"
            + re.escape(name_low)
            + r"\b",
            re.IGNORECASE,
        )
        if pat.search(low):
            out.append(
                Violation(
                    kind="identity_drift",
                    detail=(
                        f"assistant name {name!r} appears in a user-ownership "
                        f"slot (after {leader!r})"
                    ),
                    weight=DEFAULT_WEIGHTS["identity_drift"],
                    evidence=name,
                )
            )
            break  # one violation is enough; don't stack the same drift
    return out


def _extract_correction_pair(text: str) -> tuple[str, str] | None:
    """Best-effort (stale, canonical) pair from a correction memory text.

    Returns ``None`` if the text does not look like a typical "X is not Y, it
    is Z" correction.
    """
    not_m = _NOT_PHRASE.search(text)
    is_m = _IT_IS_PHRASE.search(text)
    if not_m is None or is_m is None:
        return None
    stale = clean_term(not_m.group(1))
    canon = clean_term(is_m.group(1))
    if not stale or not canon or stale.lower() == canon.lower():
        return None
    return stale, canon


def _check_correction_violation(
    candidate: str,
    corrections: Sequence[MemoryEntry],
) -> list[Violation]:
    out: list[Violation] = []
    low = candidate.lower()
    for entry in corrections:
        pair = _extract_correction_pair(entry.text)
        if pair is None:
            continue
        stale, canon = pair
        stale_low = stale.lower()
        canon_low = canon.lower()
        if stale_low in low and canon_low not in low:
            out.append(
                Violation(
                    kind="correction_violation",
                    detail=(
                        f"candidate restates pre-correction form {stale!r}; "
                        f"corrected form is {canon!r}"
                    ),
                    weight=DEFAULT_WEIGHTS["correction_violation"],
                    evidence=entry.id,
                )
            )
    return out


def _check_list_merge(
    candidate: str,
    recent_user_turns: Sequence[str],
) -> list[Violation]:
    if not recent_user_turns:
        return []
    low = candidate.lower()
    seen_pairs: set[tuple[str, str]] = set()
    out: list[Violation] = []
    # Walk most-recent turn first; first violation is enough per pair.
    for turn in reversed(list(recent_user_turns)):
        for m in _PAIRED_AND.finditer(turn):
            a = m.group(1).strip().lower()
            b = m.group(2).strip().lower()
            if not a or not b or a == b:
                continue
            if a in _LIST_MERGE_STOP or b in _LIST_MERGE_STOP:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            a_in = a in low
            b_in = b in low
            if a_in != b_in:
                missing = b if not b_in else a
                kept = a if not b_in else b
                out.append(
                    Violation(
                        kind="list_merge",
                        detail=(
                            f"user paired {a!r} and {b!r}; candidate kept "
                            f"{kept!r} but dropped {missing!r}"
                        ),
                        weight=DEFAULT_WEIGHTS["list_merge"],
                        evidence=turn[:80],
                    )
                )
    return out


def _check_length_overflow(candidate: str) -> list[Violation]:
    out: list[Violation] = []
    for line in candidate.splitlines():
        m = _SLOT_LINE.match(line)
        if m is None:
            continue
        slot = m.group(1).strip()
        value = m.group(2).strip()
        if len(value) > MAX_FACT_VALUE_LEN:
            out.append(
                Violation(
                    kind="length_overflow",
                    detail=(
                        f"slot {slot!r} value is {len(value)} chars " f"(max {MAX_FACT_VALUE_LEN})"
                    ),
                    weight=DEFAULT_WEIGHTS["length_overflow"],
                    evidence=slot,
                )
            )
    return out


class RuleBasedVerifier:
    """Deterministic verifier; sum-of-weighted-components energy.

    Thresholds are computed from energy:

    - ``energy < threshold_reprompt`` -> ``accept``
    - ``energy < threshold_reject``   -> ``reprompt``
    - otherwise                       -> ``reject``

    No LLM calls. No I/O. Determinism is a hard requirement so the verifier
    can act as a stable training label source for a future learned scorer.
    """

    def __init__(
        self,
        *,
        threshold_reprompt: float = 1.0,
        threshold_reject: float = 2.5,
        weights: dict[ViolationKind, float] | None = None,
    ) -> None:
        self._threshold_reprompt = float(threshold_reprompt)
        self._threshold_reject = float(threshold_reject)
        merged = dict(DEFAULT_WEIGHTS)
        if weights:
            merged.update(weights)
        self._weights: dict[ViolationKind, float] = merged

    def _reweight(self, vs: Sequence[Violation]) -> list[Violation]:
        if not vs:
            return []
        out: list[Violation] = []
        for v in vs:
            w = self._weights.get(v.kind, v.weight)
            if w == v.weight:
                out.append(v)
                continue
            out.append(
                Violation(
                    kind=v.kind,
                    detail=v.detail,
                    weight=w,
                    evidence=v.evidence,
                )
            )
        return out

    def score(self, candidate: str, ctx: VerifierContext) -> Verdict:
        if not candidate or not candidate.strip():
            return Verdict(energy=0.0, decision="accept")

        violations: list[Violation] = []

        facts: list[Fact] = []
        if ctx.facts is not None:
            try:
                facts = list(ctx.facts.all_facts())
            except Exception:
                facts = []
        violations.extend(_check_fact_contradiction(candidate, facts, ctx.normalizer))
        violations.extend(_check_identity_drift(candidate, ctx.identity))
        violations.extend(_check_correction_violation(candidate, ctx.recent_corrections))
        violations.extend(_check_list_merge(candidate, ctx.recent_user_turns))
        violations.extend(_check_length_overflow(candidate))

        violations = self._reweight(violations)
        energy = sum(v.weight for v in violations)

        if energy < self._threshold_reprompt:
            decision: DecisionKind = "accept"
        elif energy < self._threshold_reject:
            decision = "reprompt"
        else:
            decision = "reject"

        fix_hints = tuple(v.fix_hint() for v in violations)
        return Verdict(
            energy=energy,
            violations=tuple(violations),
            decision=decision,
            fix_hints=fix_hints,
        )
