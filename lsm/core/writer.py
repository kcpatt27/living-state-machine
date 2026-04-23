"""Gated memory writer — Stentor primitive #3.

A memory writer decides whether an incoming candidate turn should be
committed to durable memory. Without gating every turn gets written and
retrieval fills with filler ("ok", "thanks") which poisons downstream
scoring.

The writer is a pure classifier: ``decide(candidate) -> WriteDecision``. The
pipeline stays responsible for the actual ``memory.put`` call. This keeps
the writer testable without a store, and lets us A/B several gates behind
the same protocol.

Implementations
===============

- :class:`AlwaysWriter` — ablation baseline. Keeps the pre-L7 behaviour of
  writing every non-empty turn. Use this as the "writer-off" comparison.
- :class:`NeverWriter` — useful for extreme ablations (zero-memory runs).
- :class:`RuleBasedWriter` — hand-written heuristics. Writes if the turn
  looks like a durable fact (regex matches on "my X is Y", "I prefer Y",
  named entities, long descriptive statements). Rejects filler, greetings,
  and very short / question-only utterances.
- :class:`LearnedWriter` — a tiny logistic classifier trained on a
  synthesised labelled set. Falls back to :class:`RuleBasedWriter` when the
  classifier is not yet trained.

Every writer also exposes a ``features`` method so we can log per-candidate
signals into the run log (useful for post-hoc analysis and for bootstrapping
the learned writer's training set).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

# Regexes that indicate a durable personal fact. These are intentionally
# conservative; the goal is high precision on "write", not high recall.
_DURABLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmy\s+[a-z][a-z\s\-']{1,40}\s+is\s+[^.\n?!]{1,80}", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am)\s+(?:a|an|the)?\s*[a-z][a-z\s\-']{1,40}", re.IGNORECASE),
    re.compile(r"\bi\s+(?:prefer|like|love|hate|avoid)\s+[^.\n?!]{2,80}", re.IGNORECASE),
    re.compile(r"\bi\s+work\s+(?:at|as|for)\s+[^.\n?!]{2,80}", re.IGNORECASE),
    re.compile(r"\bi\s+live\s+in\s+[^.\n?!]{2,80}", re.IGNORECASE),
    re.compile(r"\bcall\s+me\s+[^.\n?!]{1,40}", re.IGNORECASE),
    re.compile(r"\bremember\s+(?:that\s+)?[^.\n?!]{3,120}", re.IGNORECASE),
    re.compile(r"\bnote\s+(?:that\s+)?[^.\n?!]{3,120}", re.IGNORECASE),
)

# Filler patterns (short affirmations / thanks / greetings) — reject on match.
_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(ok|okay|cool|nice|sure|yep|yes|yeah|no|nope|thanks|thank you|"
        r"thx|ty|hi|hey|hello|bye|goodbye|great|awesome|lol|lmao|"
        r"got it|makes sense|alright|got you|understood|k|kk)\s*[.!?]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(hmm|uh|um|ah|oh|huh)\s*[.!?]*\s*$", re.IGNORECASE),
)

_QUESTION_ONLY = re.compile(r"^\s*[^.]*\?\s*$")


@dataclass(slots=True)
class WriteCandidate:
    """Input to the writer: one candidate turn."""

    text: str
    role: str = "user"
    source: str = "conversation"
    tags: list[str] = field(default_factory=list)
    # Optional extra signals from upstream; all optional.
    novelty: float | None = None  # 1 - max_cosine(candidate, existing memory)
    self_critic_score: float | None = None  # from the outcome loop (L9+)


@dataclass(slots=True)
class WriteDecision:
    """Output of the writer. ``decision`` is one of: write / skip / merge."""

    write: bool
    reason: str
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    # Optional hint to the store: "new" (default), "merge-with:<id>", "overwrite:<id>".
    # L7 only uses "new" and skip; L8 will exercise merge.
    action: str = "new"


@runtime_checkable
class MemoryWriter(Protocol):
    """Protocol for any writer. Stateless w.r.t. the store."""

    name: str

    def decide(self, candidate: WriteCandidate) -> WriteDecision: ...

    def features(self, candidate: WriteCandidate) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Feature extraction shared by all writers
# ---------------------------------------------------------------------------


def _token_count(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9']+", text)))


def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in ".,!?;:")
    return punct / max(len(text), 1)


def _has_question_mark(text: str) -> bool:
    return "?" in text


def _fraction_alpha(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / max(len(text), 1)


def _durable_hits(text: str) -> int:
    return sum(1 for p in _DURABLE_PATTERNS if p.search(text))


def _is_filler(text: str) -> bool:
    return any(p.search(text) for p in _FILLER_PATTERNS)


def _extract_features(candidate: WriteCandidate) -> dict[str, float]:
    text = candidate.text or ""
    n_tokens = _token_count(text)
    novelty = 0.5 if candidate.novelty is None else float(candidate.novelty)
    critic = 0.5 if candidate.self_critic_score is None else float(candidate.self_critic_score)
    return {
        "length_chars": float(len(text)),
        "length_tokens": float(n_tokens),
        "log_tokens": math.log1p(n_tokens),
        "has_question": float(_has_question_mark(text)),
        "question_only": float(bool(_QUESTION_ONLY.match(text)) and _has_question_mark(text)),
        "punct_ratio": _punct_ratio(text),
        "fraction_alpha": _fraction_alpha(text),
        "durable_hits": float(_durable_hits(text)),
        "is_filler": float(_is_filler(text)),
        "is_user": float(candidate.role == "user"),
        "novelty": novelty,
        "self_critic": critic,
    }


# ---------------------------------------------------------------------------
# Concrete writers
# ---------------------------------------------------------------------------


class AlwaysWriter:
    """Ablation baseline: write anything with non-empty text."""

    name = "always"

    def decide(self, candidate: WriteCandidate) -> WriteDecision:
        text = (candidate.text or "").strip()
        if not text:
            return WriteDecision(write=False, reason="empty", features={})
        return WriteDecision(
            write=True,
            reason="always",
            score=1.0,
            features=_extract_features(candidate),
        )

    def features(self, candidate: WriteCandidate) -> dict[str, float]:
        return _extract_features(candidate)


class NeverWriter:
    """Extreme ablation: drop every candidate."""

    name = "never"

    def decide(self, candidate: WriteCandidate) -> WriteDecision:
        return WriteDecision(
            write=False,
            reason="never",
            features=_extract_features(candidate),
        )

    def features(self, candidate: WriteCandidate) -> dict[str, float]:
        return _extract_features(candidate)


@dataclass(slots=True)
class RuleBasedWriterConfig:
    """Thresholds for the heuristic writer."""

    min_tokens: int = 4
    min_chars: int = 12
    min_fraction_alpha: float = 0.5
    min_novelty: float = 0.15
    assistant_requires_durable_hit: bool = True


class RuleBasedWriter:
    """Heuristic writer. High-precision keep, aggressive reject on filler."""

    name = "rules"

    def __init__(self, config: RuleBasedWriterConfig | None = None) -> None:
        self.config = config or RuleBasedWriterConfig()

    def features(self, candidate: WriteCandidate) -> dict[str, float]:
        return _extract_features(candidate)

    def decide(self, candidate: WriteCandidate) -> WriteDecision:
        feats = self.features(candidate)
        text = (candidate.text or "").strip()

        if not text:
            return WriteDecision(write=False, reason="empty", features=feats)
        if feats["is_filler"] >= 1.0:
            return WriteDecision(write=False, reason="filler", features=feats)
        if feats["length_tokens"] < self.config.min_tokens:
            return WriteDecision(write=False, reason="too_short", features=feats)
        if feats["length_chars"] < self.config.min_chars:
            return WriteDecision(write=False, reason="too_short_chars", features=feats)
        if feats["fraction_alpha"] < self.config.min_fraction_alpha:
            return WriteDecision(write=False, reason="mostly_non_alpha", features=feats)
        if feats["question_only"] >= 1.0:
            # Questions are worth keeping only if they also introduce a durable
            # fact (e.g. "my name is K; what should I call you?"). Otherwise
            # skip; they'd just echo back on next retrieval.
            if feats["durable_hits"] < 1.0:
                return WriteDecision(write=False, reason="question_only", features=feats)
        if candidate.novelty is not None and candidate.novelty < self.config.min_novelty:
            return WriteDecision(write=False, reason="low_novelty", features=feats)

        if candidate.role == "assistant" and self.config.assistant_requires_durable_hit:
            if feats["durable_hits"] < 1.0:
                return WriteDecision(write=False, reason="assistant_no_durable_hit", features=feats)

        score = (
            0.4 * min(feats["durable_hits"], 3.0) / 3.0
            + 0.25 * min(feats["log_tokens"], 5.0) / 5.0
            + 0.2 * feats["novelty"]
            + 0.15 * feats["self_critic"]
        )
        return WriteDecision(
            write=True,
            reason="kept_by_rules",
            score=float(score),
            features=feats,
            action="new",
        )


# ---------------------------------------------------------------------------
# Tiny learned writer (logistic regression over the feature dict)
# ---------------------------------------------------------------------------


_FEATURE_ORDER: tuple[str, ...] = (
    "log_tokens",
    "durable_hits",
    "is_filler",
    "has_question",
    "question_only",
    "fraction_alpha",
    "punct_ratio",
    "is_user",
    "novelty",
    "self_critic",
)


@dataclass(slots=True)
class LearnedWriterConfig:
    threshold: float = 0.5
    fallback: RuleBasedWriterConfig = field(default_factory=RuleBasedWriterConfig)


@dataclass(slots=True)
class LearnedWriterModel:
    """Weights for a logistic classifier.

    ``weights`` are aligned with :data:`_FEATURE_ORDER`; ``bias`` is added
    before the sigmoid.
    """

    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0

    def predict_proba(self, features: dict[str, float]) -> float:
        if not self.weights:
            return 0.5
        z = self.bias
        for name, w in self.weights.items():
            z += w * features.get(name, 0.0)
        return 1.0 / (1.0 + math.exp(-z))


class LearnedWriter:
    """Logistic writer with :class:`RuleBasedWriter` as a graceful fallback.

    Train offline with :func:`train_learned_writer`; the resulting
    :class:`LearnedWriterModel` can be persisted via :meth:`save` / loaded
    via :meth:`load`. If no model is loaded, :meth:`decide` delegates to
    :class:`RuleBasedWriter`.
    """

    name = "learned"

    def __init__(
        self,
        config: LearnedWriterConfig | None = None,
        model: LearnedWriterModel | None = None,
    ) -> None:
        self.config = config or LearnedWriterConfig()
        self.model = model
        self._fallback = RuleBasedWriter(self.config.fallback)

    def features(self, candidate: WriteCandidate) -> dict[str, float]:
        return _extract_features(candidate)

    def decide(self, candidate: WriteCandidate) -> WriteDecision:
        feats = self.features(candidate)
        if self.model is None or not self.model.weights:
            decision = self._fallback.decide(candidate)
            decision.reason = f"fallback:{decision.reason}"
            return decision

        # Hard override: if the cheap rule says it's empty / filler, trust the
        # rule (the model can be noisy on those edges).
        text = (candidate.text or "").strip()
        if not text:
            return WriteDecision(write=False, reason="empty", features=feats)
        if feats["is_filler"] >= 1.0:
            return WriteDecision(write=False, reason="filler", features=feats)

        prob = self.model.predict_proba(feats)
        write = prob >= self.config.threshold
        return WriteDecision(
            write=write,
            reason=f"learned:p={prob:.3f}",
            score=float(prob),
            features=feats,
            action="new",
        )

    def save(self, path: Path) -> None:
        if self.model is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threshold": self.config.threshold,
            "bias": self.model.bias,
            "weights": self.model.weights,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.config.threshold = float(payload.get("threshold", self.config.threshold))
        self.model = LearnedWriterModel(
            weights={str(k): float(v) for k, v in payload.get("weights", {}).items()},
            bias=float(payload.get("bias", 0.0)),
        )


# ---------------------------------------------------------------------------
# Training loop for the learned writer (batch gradient descent on logistic loss)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LabeledCandidate:
    """Training example: a candidate and its gold keep/skip label (1 or 0)."""

    candidate: WriteCandidate
    label: int


def train_learned_writer(
    examples: Sequence[LabeledCandidate],
    *,
    epochs: int = 400,
    lr: float = 0.5,
    l2: float = 1e-3,
    feature_order: Iterable[str] = _FEATURE_ORDER,
) -> LearnedWriterModel:
    """Train a tiny logistic regression with plain gradient descent.

    No numpy dependency to keep this path usable in extremely stripped-down
    environments; we only ever have a handful of features and a few hundred
    examples so a Python loop is fine (< 1s on the benchmarks below).
    """
    order = list(feature_order)
    weights = {name: 0.0 for name in order}
    bias = 0.0
    n = max(len(examples), 1)

    # Precompute feature vectors.
    feats: list[tuple[dict[str, float], int]] = [
        (_extract_features(ex.candidate), int(ex.label)) for ex in examples
    ]

    for _ in range(epochs):
        grad_w = {name: 0.0 for name in order}
        grad_b = 0.0
        for feat, y in feats:
            z = bias
            for name in order:
                z += weights[name] * feat.get(name, 0.0)
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - y
            grad_b += err
            for name in order:
                grad_w[name] += err * feat.get(name, 0.0)
        bias -= lr * grad_b / n
        for name in order:
            step = lr * (grad_w[name] / n + l2 * weights[name])
            weights[name] -= step

    return LearnedWriterModel(weights=weights, bias=bias)


def build_writer(kind: str, **kwargs: object) -> MemoryWriter:
    """Factory. ``kind`` is one of: always, never, rules, learned."""
    kind = kind.lower()
    if kind == "always":
        return AlwaysWriter()
    if kind == "never":
        return NeverWriter()
    if kind == "rules":
        return RuleBasedWriter(RuleBasedWriterConfig(**kwargs))  # type: ignore[arg-type]
    if kind == "learned":
        return LearnedWriter(LearnedWriterConfig(**kwargs))  # type: ignore[arg-type]
    raise ValueError(f"unknown writer kind: {kind!r}")
