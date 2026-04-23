"""multi_session_consistency_v1 — facts survive across separated sessions.

The test simulates three stages per item:

1. **Session A** (anchor): a single durable fact is written, followed by a
   large batch of unrelated filler turns.
2. **Mid-gap**: we call ``memory.tick(now=session_a_end + gap_seconds)`` to
   age the store. Fast-tier entries expire; session/durable survive
   depending on promotion and decay parameters.
3. **Session C** (query): we retrieve against a question that should
   surface the anchor fact.

Metric: accuracy of the anchor being in the top-k. Reported separately for
the multi-timescale store vs. a flat store — the flat store has no tiered
decay so large filler drowns the anchor; tiered memory should preserve it.

This is a retrieval-only task; no LLM call.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lsm.core.embeddings import Embedder
from lsm.core.memory import InMemoryStore, MemoryStore
from lsm.core.multi_timescale import (
    MultiTimescaleConfig,
    MultiTimescaleMemory,
    TIER_DURABLE,
    TierConfig,
)

TASK_NAME = "multi_session_consistency_v1"


_FACTS: list[tuple[str, str]] = [
    ("my backup drive is nicknamed 'obsidian-03'", "what is my backup drive called"),
    ("our deploy canary server is ember-7", "what is our deploy canary server"),
    ("my recovery phrase ends with the word 'willow'", "what word ends my recovery phrase"),
    ("i keep my tax folder in the lunar filing cabinet", "where is my tax folder"),
    ("our family car's licence plate is APX 4782", "what is our car's licence plate"),
    ("my locker combination code is 17-42-05", "what is my locker combination"),
    ("my dentist works on tuesdays at cedar clinic", "when does my dentist work"),
    ("our home wifi ssid is 'myko-net'", "what is our home wifi ssid"),
]

_FILLER: list[str] = [
    "the weather is pleasant today",
    "any interesting news",
    "what time is it now",
    "let's discuss something else",
    "that's an interesting point",
    "i'm not sure i follow",
    "could you explain that again",
    "i'll think about it",
    "tell me more about that",
    "that was a good meeting",
    "let's catch up tomorrow",
    "send me the link please",
    "did you see the email",
    "the project is on track",
    "quick question about yesterday",
]


@dataclass(slots=True)
class ConsistencyConfig:
    num_items: int = 12
    filler_turns: int = 40
    top_k: int = 5
    gap_seconds: float = 6 * 3600.0  # > fast tau, < session tau by default
    seed: int = 0


@dataclass(slots=True)
class ConsistencyItem:
    item_id: str
    anchor_text: str
    query: str


def generate(cfg: ConsistencyConfig) -> list[ConsistencyItem]:
    rng = random.Random(cfg.seed)
    items: list[ConsistencyItem] = []
    for i in range(cfg.num_items):
        fact, query = rng.choice(_FACTS)
        items.append(
            ConsistencyItem(
                item_id=f"multi_session_consistency_v1-{cfg.seed}-{i:03d}",
                anchor_text=fact,
                query=query,
            )
        )
    return items


def _run_item_tiered(
    item: ConsistencyItem,
    cfg: ConsistencyConfig,
    embedder: Embedder,
    mtm_cfg: MultiTimescaleConfig,
    rng: random.Random,
) -> bool:
    memory = MultiTimescaleMemory(embedder=embedder, config=mtm_cfg)
    t0 = 1_700_000_000.0

    anchor = memory.put(item.anchor_text, source="anchor", meta={"t": t0})
    # promote anchor deliberately (simulates outcome loop L9 endorsing the
    # fact) — we do it by querying the anchor's own topic enough times to
    # trip the session -> durable thresholds.
    hits_needed = (
        mtm_cfg.for_tier("fast").promote_hits
        + mtm_cfg.for_tier("session").promote_hits
    )
    for _ in range(hits_needed):
        memory.query(item.anchor_text, top_k=1)

    # Fill session A with unrelated turns so the flat baseline is flooded.
    for i in range(cfg.filler_turns):
        memory.put(rng.choice(_FILLER), source="filler", meta={"t": t0 + i})

    # Age the store.
    memory.tick(now=t0 + cfg.gap_seconds)

    hits = memory.query(item.query, top_k=cfg.top_k)
    memory.close()
    return any(h.entry.id == anchor.id for h in hits)


def _run_item_flat(
    item: ConsistencyItem,
    cfg: ConsistencyConfig,
    embedder: Embedder,
    rng: random.Random,
) -> bool:
    memory: MemoryStore = InMemoryStore(embedder=embedder)
    anchor = memory.put(item.anchor_text, source="anchor")
    for _ in range(cfg.filler_turns):
        memory.put(rng.choice(_FILLER), source="filler")
    hits = memory.query(item.query, top_k=cfg.top_k)
    memory.close()
    return any(h.entry.id == anchor.id for h in hits)


def run(
    cfg: ConsistencyConfig,
    embedder: Embedder,
    mtm_config: MultiTimescaleConfig | None = None,
) -> dict[str, Any]:
    rng = random.Random(cfg.seed)
    mtm_cfg = mtm_config or MultiTimescaleConfig()
    items = generate(cfg)

    flat: list[bool] = []
    tiered: list[bool] = []
    for it in items:
        flat.append(_run_item_flat(it, cfg, embedder, random.Random(rng.random())))
        tiered.append(_run_item_tiered(it, cfg, embedder, mtm_cfg, random.Random(rng.random())))

    return {
        "task": TASK_NAME,
        "config": asdict(cfg),
        "flat_accuracy": sum(flat) / max(len(flat), 1),
        "tiered_accuracy": sum(tiered) / max(len(tiered), 1),
        "delta": sum(tiered) / max(len(tiered), 1) - sum(flat) / max(len(flat), 1),
        "per_item": [
            {"item_id": it.item_id, "flat": f, "tiered": t}
            for it, f, t in zip(items, flat, tiered, strict=True)
        ],
    }


def compare(
    cfg: ConsistencyConfig, embedder: Embedder, out_path: Path | None = None
) -> dict[str, Any]:
    result = run(cfg, embedder)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = [
    "TASK_NAME",
    "ConsistencyConfig",
    "ConsistencyItem",
    "generate",
    "run",
    "compare",
    "TierConfig",
    "TIER_DURABLE",
]
