"""Meta-memory and plasticity scheduling — Stentor primitive #6.

Every memory entry carries a ``trust`` in ``[0, 1]`` (already on
:class:`~lsm.core.memory.MemoryEntry`) and an implicit per-entry
``plasticity`` stored here in the plasticity manager's own table.

Update semantics
================

When the outcome loop scores an assistant reply, we know which memories
were retrieved to produce that reply. The :class:`PlasticityManager`:

1. Pushes the trust of each retrieved memory toward the outcome score at a
   rate proportional to that memory's current plasticity::

       trust' = (1 - eta_k) * trust + eta_k * target

2. Shrinks plasticity when trust is stable (consistent evidence) and
   expands it when trust is volatile (contradictory evidence)::

       eta_k' = clip(eta_k * (1 - rho) + rho * |trust - target|, eta_min, eta_max)

   where ``rho`` is the meta-learning rate.

This is the closest direct analogue to Stentor's transient-vs-durable
distinction: repeatedly-confirmed entries become *slow* and resist
rewrites; volatile entries stay *fast* and churn.

Interaction with other primitives
=================================

- **Scoring.** ``trust_boost_combiner`` is a :data:`ScoreCombiner` that
  multiplies cosine similarity by ``(1 + boost * (trust - 0.5))``, so
  high-trust entries outrank competitors at tie-break.
- **Writer.** (Future hook.) Low-plasticity, high-trust entries should
  prefer merge-with-existing over new insertion; L7 already accepts an
  ``action`` hint on :class:`~lsm.core.writer.WriteDecision`.
- **Multi-timescale promotion.** (Future hook.) Durable-tier entries
  should require ``trust >= threshold``.

The manager persists to JSON so state survives restarts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from lsm.core.memory import MemoryEntry


@dataclass(slots=True)
class PlasticityConfig:
    """Knobs for :class:`ExpDecayPlasticity`."""

    eta_init: float = 0.3  # initial per-entry plasticity
    eta_min: float = 0.02
    eta_max: float = 0.6
    rho: float = 0.15  # meta-learning rate on plasticity itself
    trust_boost: float = 0.4  # strength of trust's score bump


@dataclass(slots=True)
class PlasticityState:
    """Per-entry meta-state."""

    trust: float = 0.5
    plasticity: float = 0.3
    updates: int = 0
    last_update: float = field(default_factory=time.time)


@runtime_checkable
class Plasticity(Protocol):
    """Per-entry plasticity + trust interface."""

    config: PlasticityConfig

    def observe(
        self, entry_ids: Sequence[str], target: float, now: float | None = None
    ) -> None: ...

    def state_of(self, entry_id: str) -> PlasticityState: ...

    def trust(self, entry_id: str) -> float: ...

    def combiner(
        self, base: float, entry: MemoryEntry, context: Sequence[str]
    ) -> tuple[float, dict[str, float]]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


class NullPlasticity:
    """Ablation: constant trust, no updates, no boost."""

    name = "null"

    def __init__(self, config: PlasticityConfig | None = None) -> None:
        self.config = config or PlasticityConfig()

    def observe(
        self, entry_ids: Sequence[str], target: float, now: float | None = None
    ) -> None:
        return None

    def state_of(self, entry_id: str) -> PlasticityState:
        return PlasticityState(trust=0.5, plasticity=self.config.eta_init)

    def trust(self, entry_id: str) -> float:
        return 0.5

    def combiner(
        self, base: float, entry: MemoryEntry, context: Sequence[str]
    ) -> tuple[float, dict[str, float]]:
        _ = entry, context
        return base, {"trust": 0.5, "plasticity": self.config.eta_init}

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> None:
        return None


class ExpDecayPlasticity:
    """Per-entry trust updates with self-scheduling plasticity."""

    name = "exp_decay"

    def __init__(self, config: PlasticityConfig | None = None) -> None:
        self.config = config or PlasticityConfig()
        self._state: dict[str, PlasticityState] = {}

    def _ensure(self, entry_id: str) -> PlasticityState:
        st = self._state.get(entry_id)
        if st is None:
            st = PlasticityState(
                trust=0.5, plasticity=self.config.eta_init, updates=0, last_update=time.time()
            )
            self._state[entry_id] = st
        return st

    def observe(
        self, entry_ids: Sequence[str], target: float, now: float | None = None
    ) -> None:
        target = max(0.0, min(1.0, float(target)))
        t = now if now is not None else time.time()
        for eid in entry_ids:
            st = self._ensure(eid)
            eta = st.plasticity
            # Trust update.
            new_trust = (1.0 - eta) * st.trust + eta * target
            # Plasticity update: shrink when trust was consistent with target
            # (|delta| small), expand when not.
            delta = abs(st.trust - target)
            eta_next = eta * (1.0 - self.config.rho) + self.config.rho * delta
            st.trust = float(max(0.0, min(1.0, new_trust)))
            st.plasticity = float(
                max(self.config.eta_min, min(self.config.eta_max, eta_next))
            )
            st.updates += 1
            st.last_update = t

    def state_of(self, entry_id: str) -> PlasticityState:
        return self._ensure(entry_id)

    def trust(self, entry_id: str) -> float:
        return self._ensure(entry_id).trust

    def combiner(
        self, base: float, entry: MemoryEntry, context: Sequence[str]
    ) -> tuple[float, dict[str, float]]:
        _ = context
        st = self._state.get(entry.id)
        trust = st.trust if st is not None else float(entry.trust)
        plasticity = st.plasticity if st is not None else self.config.eta_init
        boost = self.config.trust_boost * (trust - 0.5)
        scaled = base * (1.0 + boost)
        return scaled, {
            "trust": trust,
            "plasticity": plasticity,
            "trust_boost": boost,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "eta_init": self.config.eta_init,
                "eta_min": self.config.eta_min,
                "eta_max": self.config.eta_max,
                "rho": self.config.rho,
                "trust_boost": self.config.trust_boost,
            },
            "state": {
                eid: {
                    "trust": st.trust,
                    "plasticity": st.plasticity,
                    "updates": st.updates,
                    "last_update": st.last_update,
                }
                for eid, st in self._state.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        self.config = PlasticityConfig(
            eta_init=float(cfg.get("eta_init", self.config.eta_init)),
            eta_min=float(cfg.get("eta_min", self.config.eta_min)),
            eta_max=float(cfg.get("eta_max", self.config.eta_max)),
            rho=float(cfg.get("rho", self.config.rho)),
            trust_boost=float(cfg.get("trust_boost", self.config.trust_boost)),
        )
        self._state.clear()
        for eid, s in payload.get("state", {}).items():
            self._state[eid] = PlasticityState(
                trust=float(s.get("trust", 0.5)),
                plasticity=float(s.get("plasticity", self.config.eta_init)),
                updates=int(s.get("updates", 0)),
                last_update=float(s.get("last_update", time.time())),
            )


__all__ = [
    "ExpDecayPlasticity",
    "NullPlasticity",
    "Plasticity",
    "PlasticityConfig",
    "PlasticityState",
]
