"""Association — Stentor primitive #2.

A sparse, symmetric Hebbian co-occurrence matrix over memory keys. When two
entries are retrieved together, the edge between them strengthens. During
scoring, an entry's score gets a boost proportional to its accumulated edge
weight with the items currently in context.

Update rule (on co-hit of a set K of keys):

    for i, j in pairs(K): A_ij <- A_ij + eta
    (applied symmetrically; weights stored on sorted-pair keys)

Decay (applied lazily between calls):

    A_ij <- A_ij * exp(-lambda * dt)

Boost applied during scoring, given context keys C:

    support = mean_{c in C} A_{entry,c}
    boosted = base + boost_strength * tanh(support)

``NullAssociation`` is the ablation: pass-through score, no-op updates.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Protocol, runtime_checkable

from lsm.core.memory import MemoryEntry

ScoreCombiner = Callable[[float, MemoryEntry, Sequence[str]], tuple[float, dict[str, float]]]


@dataclass(slots=True)
class AssociationConfig:
    """Parameters for :class:`HebbianAssociation`."""

    eta: float = 0.25
    lambda_per_second: float = 1e-5
    boost_strength: float = 0.35


@runtime_checkable
class Association(Protocol):
    """Hebbian-style association interface."""

    def co_hit(self, keys: Iterable[str], now: float | None = None) -> None: ...

    def weight(self, a: str, b: str, now: float | None = None) -> float: ...

    def combiner(
        self, base: float, entry: MemoryEntry, context: Sequence[str]
    ) -> tuple[float, dict[str, float]]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


class NullAssociation:
    """Ablation: no association."""

    def co_hit(self, keys: Iterable[str], now: float | None = None) -> None:
        return None

    def weight(self, a: str, b: str, now: float | None = None) -> float:
        return 0.0

    def combiner(
        self, base: float, entry: MemoryEntry, context: Sequence[str]
    ) -> tuple[float, dict[str, float]]:
        return base, {"association": 0.0}

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> None:
        return None


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class HebbianAssociation:
    """Sparse symmetric co-occurrence matrix with exponential decay."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._weights: dict[tuple[str, str], float] = {}
        self._last_ts: float = time.time()

    def _decay_all(self, now: float) -> None:
        # If the caller's clock is at or behind our anchor (e.g. tests using
        # synthetic timestamps while __init__ set _last_ts = time.time()),
        # just re-anchor to the caller's clock — we only ever decay forward.
        if now <= self._last_ts:
            self._last_ts = now
            return
        dt = now - self._last_ts
        factor = math.exp(-self.config.lambda_per_second * dt)
        if factor == 1.0:
            self._last_ts = now
            return
        for k in list(self._weights.keys()):
            self._weights[k] *= factor
            if self._weights[k] < 1e-6:
                del self._weights[k]
        self._last_ts = now

    def co_hit(self, keys: Iterable[str], now: float | None = None) -> None:
        t = now if now is not None else time.time()
        self._decay_all(t)
        unique = sorted(set(k for k in keys if k))
        if len(unique) < 2:
            return
        for a, b in combinations(unique, 2):
            key = _pair(a, b)
            self._weights[key] = self._weights.get(key, 0.0) + self.config.eta

    def weight(self, a: str, b: str, now: float | None = None) -> float:
        if a == b:
            return 0.0
        t = now if now is not None else time.time()
        self._decay_all(t)
        return self._weights.get(_pair(a, b), 0.0)

    def combiner(
        self, base: float, entry: MemoryEntry, context: Sequence[str]
    ) -> tuple[float, dict[str, float]]:
        if not context:
            return base, {"association": 0.0, "association_support": 0.0}
        now = time.time()
        self._decay_all(now)
        supports = [self._weights.get(_pair(entry.id, c), 0.0) for c in context if c != entry.id]
        support = sum(supports) / max(len(supports), 1)
        boost = self.config.boost_strength * math.tanh(support)
        return base + boost, {
            "association": boost,
            "association_support": support,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "eta": self.config.eta,
                "lambda_per_second": self.config.lambda_per_second,
                "boost_strength": self.config.boost_strength,
            },
            "last_ts": self._last_ts,
            # dict keys must be strings in JSON; use '||' separator (neither char is allowed in uuids)
            "weights": {f"{a}||{b}": w for (a, b), w in self._weights.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        self.config = AssociationConfig(
            eta=float(cfg.get("eta", self.config.eta)),
            lambda_per_second=float(cfg.get("lambda_per_second", self.config.lambda_per_second)),
            boost_strength=float(cfg.get("boost_strength", self.config.boost_strength)),
        )
        self._last_ts = float(payload.get("last_ts", time.time()))
        self._weights = {}
        for k, v in payload.get("weights", {}).items():
            a, b = k.split("||", 1)
            self._weights[_pair(a, b)] = float(v)


def chain_combiners(*combiners: ScoreCombiner) -> ScoreCombiner:
    """Compose multiple combiners left-to-right.

    Each combiner receives the running score and returns (new_score, components).
    Components dicts are merged; later components overwrite earlier ones on key
    collision (which is why we prefix keys like ``habituation`` and
    ``association``).
    """

    def composed(
        base: float, entry: MemoryEntry, context: Sequence[str]
    ) -> tuple[float, dict[str, float]]:
        score = base
        merged: dict[str, float] = {"cosine": base}
        for fn in combiners:
            score, comps = fn(score, entry, context)
            merged.update(comps)
        return score, merged

    return composed
