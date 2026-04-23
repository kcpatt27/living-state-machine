"""Habituation — Stentor primitive #1.

Each memory key accumulates a habituation level that grows on hit and decays
exponentially over time. Retrieval scores for habituated entries are damped
so repeated, low-information inputs stop dominating the context window.

State update (continuous-time analogue, computed lazily on each access):

    h_k(t) = h_k(t_last) * exp(-(t - t_last) / tau) + eta_on_hit

Damping applied during scoring:

    damped = base * (1 - damping_strength * tanh(h_k))

Persistence: a JSON blob at ``state/habituation.json`` with per-key
``(count, last_ts)``.

Ablation: :class:`NullHabituation` is a pass-through, identical to no
habituation. The ``ablation`` flag in experiment config selects which
implementation is wired into the pipeline.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from lsm.core.memory import MemoryEntry


@dataclass(slots=True)
class HabituationConfig:
    """Parameters for :class:`ExpDecayHabituation`."""

    tau_seconds: float = 3600.0
    eta: float = 1.0
    damping_strength: float = 0.35


@runtime_checkable
class Habituation(Protocol):
    """Observe keys, damp their retrieval score, decay over time."""

    def observe(self, key: str, now: float | None = None) -> None: ...

    def value(self, key: str, now: float | None = None) -> float: ...

    def combiner(
        self,
        base: float,
        entry: MemoryEntry,
        context: Sequence[str],
        now: float | None = None,
    ) -> tuple[float, dict[str, float]]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


class NullHabituation:
    """Ablation: no habituation. Retrieval scores pass through unchanged."""

    def observe(self, key: str, now: float | None = None) -> None:
        return None

    def value(self, key: str, now: float | None = None) -> float:
        return 0.0

    def combiner(
        self,
        base: float,
        entry: MemoryEntry,
        context: Sequence[str],
        now: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        return base, {"cosine": base, "habituation": 0.0}

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> None:
        return None


@dataclass(slots=True)
class _Cell:
    count: float
    last_ts: float


class ExpDecayHabituation:
    """Standard implementation: exponential decay with on-hit increment."""

    def __init__(self, config: HabituationConfig | None = None) -> None:
        self.config = config or HabituationConfig()
        self._cells: dict[str, _Cell] = {}

    def _decayed(self, cell: _Cell, now: float) -> float:
        dt = max(now - cell.last_ts, 0.0)
        return cell.count * math.exp(-dt / max(self.config.tau_seconds, 1e-6))

    def observe(self, key: str, now: float | None = None) -> None:
        t = now if now is not None else time.time()
        cell = self._cells.get(key)
        if cell is None:
            self._cells[key] = _Cell(count=self.config.eta, last_ts=t)
            return
        cell.count = self._decayed(cell, t) + self.config.eta
        cell.last_ts = t

    def value(self, key: str, now: float | None = None) -> float:
        t = now if now is not None else time.time()
        cell = self._cells.get(key)
        if cell is None:
            return 0.0
        return self._decayed(cell, t)

    def combiner(
        self,
        base: float,
        entry: MemoryEntry,
        context: Sequence[str],
        now: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        h = self.value(entry.id, now=now)
        damp = self.config.damping_strength * math.tanh(h)
        # Subtract damp * max(base, 0) instead of base * (1 - damp).
        # The old form accidentally *boosts* negatively-scored items
        # (e.g. a cosine around -0.3 from a hash embedder), which can
        # invert the intended ranking effect of habituation.
        damped = base - damp * max(base, 0.0)
        return damped, {"cosine": base, "habituation": h, "habituation_damp": damp}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "tau_seconds": self.config.tau_seconds,
                "eta": self.config.eta,
                "damping_strength": self.config.damping_strength,
            },
            "cells": {k: {"count": c.count, "last_ts": c.last_ts} for k, c in self._cells.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        self.config = HabituationConfig(
            tau_seconds=float(cfg.get("tau_seconds", self.config.tau_seconds)),
            eta=float(cfg.get("eta", self.config.eta)),
            damping_strength=float(cfg.get("damping_strength", self.config.damping_strength)),
        )
        self._cells = {
            k: _Cell(count=float(v["count"]), last_ts=float(v["last_ts"]))
            for k, v in payload.get("cells", {}).items()
        }

    def summary(self, now: float | None = None, top_n: int = 20) -> list[tuple[str, float]]:
        """Return the top-habituated keys — used by ``lsm state show`` later."""
        t = now if now is not None else time.time()
        ranked = sorted(
            ((k, self._decayed(c, t)) for k, c in self._cells.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ranked[:top_n]
