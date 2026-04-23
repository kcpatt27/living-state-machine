"""habituation_signal_v1 — does habituation surface signal over repeated filler?

Setup per item:

- Insert one ``signal`` entry into memory (the important, rare fact).
- Insert ``n_filler_ids`` filler entries, each repeated ``filler_hits`` times
  across "observations" so their habituation levels grow.
- Query with text semantically close to the signal.

Metric: mean reciprocal rank (MRR) of the signal entry across items. Higher
is better. We expect habituation-on to beat habituation-off.

This task runs entirely at the retrieval layer; no LLM is involved.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lsm.core.embeddings import Embedder
from lsm.core.habituation import (
    ExpDecayHabituation,
    Habituation,
    HabituationConfig,
    NullHabituation,
)
from lsm.core.memory import InMemoryStore, MemoryStore

SIGNAL_TEMPLATES: list[tuple[str, str]] = [
    ("my neighbour's cat is called {token}", "what is my neighbour's cat called"),
    ("our family car is a {token} wagon", "what kind of car do we have"),
    ("my locker combo ends in {token}", "what's my locker combo ending"),
    ("my bike brand is {token}", "what brand is my bike"),
    ("i keep my keys inside the {token} drawer", "where do i keep my keys"),
]
SIGNAL_TOKENS = ["azuki", "lumen", "myko", "quarto", "pileus", "glade"]

FILLER_PROMPTS: list[str] = [
    "how's the weather today",
    "any news",
    "what time is it",
    "ok cool",
    "got it thanks",
    "interesting, tell me more",
    "alright",
    "sure sure",
    "makes sense",
    "hmm",
]


@dataclass(slots=True)
class HabSignalConfig:
    num_items: int = 20
    n_filler_ids: int = 8
    filler_hits: int = 5
    top_k: int = 5
    seed: int = 0


@dataclass(slots=True)
class HabSignalItem:
    item_id: str
    signal_text: str
    query: str


def generate(cfg: HabSignalConfig) -> list[HabSignalItem]:
    rng = random.Random(cfg.seed)
    items: list[HabSignalItem] = []
    for i in range(cfg.num_items):
        template, question = rng.choice(SIGNAL_TEMPLATES)
        token = rng.choice(SIGNAL_TOKENS) + str(rng.randint(0, 9999))
        items.append(
            HabSignalItem(
                item_id=f"habituation_signal_v1-{cfg.seed}-{i:03d}",
                signal_text=template.format(token=token),
                query=question,
            )
        )
    return items


def _run_item(
    item: HabSignalItem,
    cfg: HabSignalConfig,
    store: MemoryStore,
    habituation: Habituation,
    rng: random.Random,
) -> float:
    for eid in store.put_many(
        [rng.choice(FILLER_PROMPTS) for _ in range(cfg.n_filler_ids)],
        source="habsig_filler",
    ):
        for _ in range(cfg.filler_hits):
            habituation.observe(eid.id)

    signal = store.put(item.signal_text, source="habsig_signal")

    combiner = habituation.combiner if not isinstance(habituation, NullHabituation) else None
    hits = store.query(item.query, top_k=max(cfg.top_k, cfg.n_filler_ids + 1), combiner=combiner)

    for rank, h in enumerate(hits, start=1):
        if h.entry.id == signal.id:
            return 1.0 / rank
    return 0.0


def run(
    cfg: HabSignalConfig,
    embedder: Embedder,
    habituation: Habituation | None = None,
) -> dict[str, Any]:
    """Run the task end-to-end; return per-item and aggregate MRR.

    ``habituation=None`` means the :class:`NullHabituation` ablation (no damping).
    """
    hab = habituation or NullHabituation()
    rng = random.Random(cfg.seed)
    items = generate(cfg)

    rrs: list[float] = []
    for it in items:
        # Fresh store per item so entries don't leak
        store = InMemoryStore(embedder=embedder)
        rr = _run_item(it, cfg, store, hab, rng)
        rrs.append(rr)

    return {
        "task": "habituation_signal_v1",
        "habituation": hab.__class__.__name__,
        "config": asdict(cfg),
        "mrr_mean": sum(rrs) / max(len(rrs), 1),
        "per_item": [
            {"item_id": it.item_id, "reciprocal_rank": rr}
            for it, rr in zip(items, rrs, strict=True)
        ],
    }


def compare_ablation(
    cfg: HabSignalConfig, embedder: Embedder, out_path: Path | None = None
) -> dict[str, Any]:
    """Run with habituation on and off, return both results."""
    on = run(cfg, embedder, ExpDecayHabituation(HabituationConfig()))
    off = run(cfg, embedder, NullHabituation())
    result = {
        "with_habituation": on,
        "without_habituation": off,
        "delta_mrr": on["mrr_mean"] - off["mrr_mean"],
    }
    if out_path is not None:
        import json

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


TASK_NAME = "habituation_signal_v1"
