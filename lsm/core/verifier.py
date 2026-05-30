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
later as a clean ablation. Telemetry is persisted via :mod:`lsm.core.verifier_log`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

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

    @property
    def threshold_reprompt(self) -> float:
        return self._threshold_reprompt

    @property
    def threshold_reject(self) -> float:
        return self._threshold_reject

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


class LinearLearnedVerifier:
    """Multinomial linear classifier over hand-crafted verifier features."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        fallback: RuleBasedVerifier | None = None,
        temperature: float = 1.0,
    ) -> None:
        from lsm.core.verifier_features import feature_names

        artifact = _load_joblib_or_json(Path(model_path))
        coef = np.asarray(artifact["coef"], dtype=np.float32)
        intercept = np.asarray(artifact["intercept"], dtype=np.float32)
        classes_raw = artifact.get("classes", [])
        feat_raw = artifact.get("feature_names", feature_names())
        self._coef = coef
        self._intercept = intercept
        self._classes = [str(c) for c in classes_raw] if isinstance(classes_raw, list) else []
        self._feature_names = (
            [str(n) for n in feat_raw] if isinstance(feat_raw, list) else feature_names()
        )
        temp = artifact.get("temperature", temperature)
        self._temperature = (
            float(temp) if isinstance(temp, int | float | str) else float(temperature)
        )
        self._fallback = fallback or RuleBasedVerifier()
        if self._coef.ndim != 2:
            raise ValueError("linear verifier coef must be 2D")
        if self._intercept.ndim != 1:
            raise ValueError("linear verifier intercept must be 1D")
        if self._coef.shape[0] != len(self._classes):
            raise ValueError("linear verifier classes mismatch coef rows")

    def score(self, candidate: str, ctx: VerifierContext) -> Verdict:
        from lsm.core.verifier_features import feature_dict

        base = self._fallback.score(candidate, ctx)
        f = feature_dict(candidate, ctx, rule_verdict=base)
        x = np.asarray([float(f.get(name, 0.0)) for name in self._feature_names], dtype=np.float32)
        logits = (self._coef @ x) + self._intercept
        probs = _softmax(logits / max(self._temperature, 1e-6))
        decision = self._classes[int(np.argmax(probs))]
        p_accept = float(probs[self._class_index("accept")])
        energy = -math.log(max(p_accept, 1e-6))
        return Verdict(
            energy=energy,
            decision=decision,  # type: ignore[arg-type]
            violations=base.violations,
            fix_hints=base.fix_hints,
        )

    def _class_index(self, label: str) -> int:
        try:
            return self._classes.index(label)
        except ValueError:
            return 0


class EncoderLearnedVerifier:
    """ONNX-backed cross-encoder verifier over ``candidate [SEP] context``."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        tokenizer_path: Path | str,
        fallback: RuleBasedVerifier | None = None,
        max_length: int = 384,
    ) -> None:
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "EncoderLearnedVerifier requires onnxruntime + transformers installed."
            ) from exc
        self._ort = ort
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._max_length = int(max_length)
        self._classes = ["accept", "reprompt", "reject"]
        self._fallback = fallback or RuleBasedVerifier()

    def score(self, candidate: str, ctx: VerifierContext) -> Verdict:
        base = self._fallback.score(candidate, ctx)
        ctx_text = _context_as_text(ctx)
        enc = self._tokenizer(
            candidate,
            ctx_text,
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
            return_tensors="np",
        )
        ort_inputs = {
            k: v for k, v in enc.items() if k in {"input_ids", "attention_mask", "token_type_ids"}
        }
        logits = np.asarray(self._session.run(None, ort_inputs)[0], dtype=np.float32)[0]
        probs = _softmax(logits)
        decision = self._classes[int(np.argmax(probs))]
        p_accept = float(probs[0])
        energy = -math.log(max(p_accept, 1e-6))
        return Verdict(
            energy=energy,
            decision=decision,  # type: ignore[arg-type]
            violations=base.violations,
            fix_hints=base.fix_hints,
        )


class HybridVerifier:
    """Rule safety floor + learned head for low-energy decisions."""

    def __init__(
        self,
        learned: Verifier,
        *,
        rule: RuleBasedVerifier | None = None,
        reject_floor: float | None = None,
    ) -> None:
        self._learned = learned
        self._rule = rule or RuleBasedVerifier()
        self._reject_floor = (
            float(reject_floor) if reject_floor is not None else self._rule.threshold_reject
        )

    def score(self, candidate: str, ctx: VerifierContext) -> Verdict:
        rule_verdict = self._rule.score(candidate, ctx)
        if rule_verdict.energy >= self._reject_floor:
            return rule_verdict
        learned_verdict = self._learned.score(candidate, ctx)
        return Verdict(
            energy=learned_verdict.energy,
            decision=learned_verdict.decision,
            violations=rule_verdict.violations,
            fix_hints=rule_verdict.fix_hints,
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    x = x - np.max(x)
    exp = np.exp(x)
    z = np.sum(exp)
    if z <= 0:
        return np.zeros_like(exp)
    return exp / z


def _load_joblib_or_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import joblib

        data = joblib.load(path)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("linear verifier artifact must be dict-like")
    return raw


def _context_as_text(ctx: VerifierContext) -> str:
    parts: list[str] = []
    if ctx.identity is not None:
        parts.append(f"identity: {ctx.identity.name} - {ctx.identity.description}")
    if ctx.facts is not None:
        try:
            facts = list(ctx.facts.all_facts())
        except Exception:
            facts = []
        if facts:
            parts.append("facts:")
            for fact in facts[:24]:
                subj = f"{fact.subject}: " if fact.subject else ""
                parts.append(f"- [{fact.category}] {subj}{fact.value}")
    if ctx.recent_corrections:
        parts.append("recent_corrections:")
        for item in ctx.recent_corrections[:8]:
            parts.append(f"- {item.text}")
    if ctx.recent_user_turns:
        parts.append("recent_user_turns:")
        for turn in ctx.recent_user_turns[:8]:
            parts.append(f"- {turn}")
    return "\n".join(parts)
