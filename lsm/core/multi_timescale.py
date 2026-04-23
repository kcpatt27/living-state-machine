"""Multi-timescale memory — Stentor primitive #4.

Three tiers, three decay constants:

- **fast** (``tau_seconds ~ minutes``): holds everything written in the
  current conversation window; aggressive expiry.
- **session** (``tau_seconds ~ hours``): things that came up more than once
  in a session get promoted here.
- **durable** (``tau_seconds ~ months``): high-trust, reliably useful
  entries. Written only by explicit promotion (or, under the writer gate,
  by user-driven ingestion).

Promotion rules
===============

Every query reports its hits; for each hit we increment ``hit_count`` on
the entry. When ``hit_count >= tier.promote_hits`` the entry is moved to
the next tier (fast -> session -> durable) and its counter reset. Moves
preserve the entry's ``id`` so association/habituation state stays valid.

Decay & expiry
==============

Every entry tracks ``last_touch`` (put or hit). During :meth:`tick`, the
entry's *vitality* ``v = exp(-(now - last_touch) / tau)`` is computed. If
``v < tier.vitality_floor`` the entry is either demoted (fast -> dropped,
session -> still allowed to survive if promoted previously) or removed. The
durable tier never auto-expires in Phase 1; it only shrinks on explicit
user deletion.

Storage
=======

:class:`MultiTimescaleMemory` wraps three :class:`MemoryStore` instances.
It itself satisfies :class:`MemoryStore` so the rest of the pipeline does
not change. Under the hood it queries all tiers, applies a tier score
weight (fast=1.0, session=tier_weight_session, durable=tier_weight_durable)
and merges top-k. Tier weights are tunable so we can ablate them.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from lsm.core.embeddings import Embedder
from lsm.core.memory import (
    InMemoryStore,
    MemoryEntry,
    MemoryStore,
    ScoreCombiner,
    ScoredEntry,
)

log = logging.getLogger(__name__)

TIER_FAST = "fast"
TIER_SESSION = "session"
TIER_DURABLE = "durable"
TIERS: tuple[str, ...] = (TIER_FAST, TIER_SESSION, TIER_DURABLE)


@dataclass(slots=True)
class TierConfig:
    tau_seconds: float
    promote_hits: int
    vitality_floor: float = 1e-3
    max_entries: int = 1024
    score_weight: float = 1.0


@dataclass(slots=True)
class MultiTimescaleConfig:
    fast: TierConfig = field(
        default_factory=lambda: TierConfig(
            tau_seconds=5 * 60.0, promote_hits=2, score_weight=1.0, max_entries=256
        )
    )
    session: TierConfig = field(
        default_factory=lambda: TierConfig(
            tau_seconds=4 * 3600.0, promote_hits=3, score_weight=0.9, max_entries=1024
        )
    )
    durable: TierConfig = field(
        default_factory=lambda: TierConfig(
            tau_seconds=180 * 24 * 3600.0,
            promote_hits=10**9,  # no auto-promotion out of durable
            score_weight=0.85,
            max_entries=65536,
            vitality_floor=0.0,  # durable does not auto-expire
        )
    )

    def for_tier(self, tier: str) -> TierConfig:
        return getattr(self, tier)


@dataclass(slots=True)
class _EntryState:
    """Per-entry bookkeeping (tracked independently of the store)."""

    tier: str
    hit_count: int = 0
    last_touch: float = field(default_factory=time.time)

    def vitality(self, now: float, tau: float) -> float:
        dt = max(now - self.last_touch, 0.0)
        return math.exp(-dt / tau) if tau > 0 else 1.0


class MultiTimescaleMemory:
    """A :class:`MemoryStore` that fans out across three tiers."""

    def __init__(
        self,
        embedder: Embedder,
        config: MultiTimescaleConfig | None = None,
        fast: MemoryStore | None = None,
        session: MemoryStore | None = None,
        durable: MemoryStore | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config or MultiTimescaleConfig()
        self._stores: dict[str, MemoryStore] = {
            TIER_FAST: fast or InMemoryStore(embedder=embedder),
            TIER_SESSION: session or InMemoryStore(embedder=embedder),
            TIER_DURABLE: durable or InMemoryStore(embedder=embedder),
        }
        self._state: dict[str, _EntryState] = {}

    # --- MemoryStore protocol ------------------------------------------------

    def put(
        self,
        text: str,
        *,
        source: str = "conversation",
        tags: Iterable[str] = (),
        meta: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> MemoryEntry:
        """New entries always start in the fast tier."""
        m = dict(meta or {})
        m.setdefault("tier", TIER_FAST)
        entry = self._stores[TIER_FAST].put(text, source=source, tags=tags, meta=m)
        t = now if now is not None else time.time()
        self._state[entry.id] = _EntryState(tier=TIER_FAST, last_touch=t)
        self._enforce_capacity(TIER_FAST)
        return entry

    def put_many(self, texts: Sequence[str], **kw: Any) -> list[MemoryEntry]:
        return [self.put(t, **kw) for t in texts]

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        context_keys: Sequence[str] = (),
        combiner: ScoreCombiner | None = None,
    ) -> list[ScoredEntry]:
        merged: dict[str, ScoredEntry] = {}
        for tier in TIERS:
            store = self._stores[tier]
            weight = self.config.for_tier(tier).score_weight
            hits = store.query(
                text, top_k=top_k, context_keys=context_keys, combiner=combiner
            )
            for h in hits:
                comps = dict(h.components)
                comps["tier"] = weight
                weighted = ScoredEntry(
                    entry=h.entry, score=h.score * weight, components=comps
                )
                existing = merged.get(h.entry.id)
                if existing is None or weighted.score > existing.score:
                    merged[h.entry.id] = weighted
        ordered = sorted(merged.values(), key=lambda s: s.score, reverse=True)
        top = ordered[:top_k]
        self._on_hits([h.entry.id for h in top])
        return top

    def get(self, entry_id: str) -> MemoryEntry | None:
        for tier in TIERS:
            e = self._stores[tier].get(entry_id)
            if e is not None:
                return e
        return None

    def delete(self, ids: Sequence[str]) -> int:
        n = 0
        for tier in TIERS:
            n += self._stores[tier].delete(ids)
        for i in ids:
            self._state.pop(i, None)
        return n

    def delete_by_source(self, source: str) -> int:
        n = 0
        for tier in TIERS:
            n += self._stores[tier].delete_by_source(source)
        # Clean any orphaned state entries — easier to iterate lazily.
        for eid in list(self._state.keys()):
            if self.get(eid) is None:
                del self._state[eid]
        return n

    def count(self) -> int:
        return sum(self._stores[t].count() for t in TIERS)

    def close(self) -> None:
        for tier in TIERS:
            try:
                self._stores[tier].close()
            except Exception:  # noqa: BLE001
                log.exception("closing tier %s failed", tier)

    # --- Tier bookkeeping ---------------------------------------------------

    def tier_of(self, entry_id: str) -> str | None:
        st = self._state.get(entry_id)
        return st.tier if st is not None else None

    def tier_counts(self) -> dict[str, int]:
        return {t: self._stores[t].count() for t in TIERS}

    def _on_hits(self, ids: Sequence[str], now: float | None = None) -> None:
        t = now if now is not None else time.time()
        for eid in ids:
            st = self._state.get(eid)
            if st is None:
                # Unknown id (e.g. store was seeded outside of put); track it.
                tier = self._find_tier(eid)
                if tier is None:
                    continue
                st = _EntryState(tier=tier, last_touch=t)
                self._state[eid] = st
            st.hit_count += 1
            st.last_touch = t
            promote_hits = self.config.for_tier(st.tier).promote_hits
            if st.hit_count >= promote_hits:
                self._promote(eid, now=t)

    def _find_tier(self, entry_id: str) -> str | None:
        for tier in TIERS:
            if self._stores[tier].get(entry_id) is not None:
                return tier
        return None

    def _next_tier(self, tier: str) -> str | None:
        idx = TIERS.index(tier)
        return TIERS[idx + 1] if idx + 1 < len(TIERS) else None

    def _promote(self, entry_id: str, *, now: float) -> None:
        st = self._state.get(entry_id)
        if st is None:
            return
        src = st.tier
        dst = self._next_tier(src)
        if dst is None:
            return
        entry = self._stores[src].get(entry_id)
        if entry is None:
            return
        new_meta = dict(entry.meta)
        new_meta["tier"] = dst
        new_meta["promoted_from"] = src
        new_meta["promoted_at"] = now
        promoted = MemoryEntry(
            id=entry.id,
            text=entry.text,
            source=entry.source,
            tags=list(entry.tags),
            created_at=entry.created_at,
            updated_at=now,
            trust=entry.trust,
            habituation=entry.habituation,
            meta=new_meta,
        )
        self._stores[dst].put_entry(promoted)
        self._stores[src].delete([entry_id])
        st.tier = dst
        st.hit_count = 0
        st.last_touch = now
        self._enforce_capacity(dst)

    def tick(self, now: float | None = None) -> dict[str, int]:
        """Apply decay; demote/drop entries below the vitality floor.

        Returns a dict of tier -> count_dropped for observability.
        """
        t = now if now is not None else time.time()
        dropped: dict[str, int] = {x: 0 for x in TIERS}

        for tier in (TIER_FAST, TIER_SESSION, TIER_DURABLE):
            cfg = self.config.for_tier(tier)
            if cfg.vitality_floor <= 0.0:
                continue
            ids = list(self._state.keys())
            for eid in ids:
                st = self._state.get(eid)
                if st is None or st.tier != tier:
                    continue
                v = st.vitality(t, cfg.tau_seconds)
                if v < cfg.vitality_floor:
                    self._stores[tier].delete([eid])
                    del self._state[eid]
                    dropped[tier] += 1
        return dropped

    def _enforce_capacity(self, tier: str) -> None:
        cfg = self.config.for_tier(tier)
        store = self._stores[tier]
        excess = store.count() - cfg.max_entries
        if excess <= 0:
            return
        # Drop the oldest (by last_touch), tie-break by entry id.
        ids_in_tier = [eid for eid, st in self._state.items() if st.tier == tier]
        ids_in_tier.sort(key=lambda i: (self._state[i].last_touch, i))
        for eid in ids_in_tier[:excess]:
            store.delete([eid])
            del self._state[eid]
