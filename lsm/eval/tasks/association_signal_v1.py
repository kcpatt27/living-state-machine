"""association_signal_v1 — does Hebbian association surface co-occurring items?

Setup per item:

- Insert a ``target`` entry and several ``distractor`` entries whose texts are
  deliberately close in surface form to the target so their cosine scores
  cluster together (i.e. retrieval on text alone is nearly a coin flip).
- Insert a ``partner`` entry that *co-occurs* repeatedly with the target via
  ``association.co_hit([partner.id, target.id])``. None of the distractors
  co-occur with the partner.
- At query time, pass the partner id in ``context_keys`` and query with a
  topic-generic question. With association on, the combiner should boost the
  target via the (target, partner) edge. With association off, the ordering
  is driven by cosine alone.

Metric: mean reciprocal rank (MRR) of the target. Higher is better.

This task lives at the retrieval layer; no LLM involved.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lsm.core.association import (
    Association,
    AssociationConfig,
    HebbianAssociation,
    NullAssociation,
)
from lsm.core.embeddings import Embedder
from lsm.core.memory import InMemoryStore, MemoryStore

# Each tuple: (target_text, partner_text, distractor_texts, query)
THEMES: list[tuple[str, str, list[str], str]] = [
    (
        "the archive's northern vault holds the {tok} codex",
        "moonstone compass tuned to the archive's north wing",
        [
            "the archive's northern vault is cool and dry",
            "the archive's north wing has tall windows",
            "the archive's vault index lists many codices",
            "the archive's northern reading room is quiet",
        ],
        "what is stored in the archive's northern vault",
    ),
    (
        "the garden's back terrace grows {tok} mint",
        "terracotta watering can left on the garden's back terrace",
        [
            "the garden's back terrace is shaded in the afternoon",
            "the garden's back path is paved with slate",
            "the garden's old greenhouse has many pots",
            "the garden's back fence was repainted last spring",
        ],
        "what grows on the garden's back terrace",
    ),
    (
        "the {tok} service key lives in the desk's top drawer",
        "leather notebook kept beside the desk's top drawer",
        [
            "the desk's top drawer sticks a little",
            "the desk's second drawer has office supplies",
            "the desk has a small reading lamp",
            "the desk's surface is covered in papers",
        ],
        "what is kept in the desk's top drawer",
    ),
    (
        "the spare {tok} key hangs on the kitchen pegboard",
        "blue enamel mug that sits on the kitchen pegboard shelf",
        [
            "the kitchen pegboard holds many utensils",
            "the kitchen counter has fresh herbs",
            "the kitchen has a small clock on the wall",
            "the kitchen window faces the street",
        ],
        "what hangs on the kitchen pegboard",
    ),
]

TOKENS = ["azuki", "lumen", "myko", "quarto", "pileus", "glade", "ember", "koto"]


@dataclass(slots=True)
class AssocSignalConfig:
    num_items: int = 20
    co_hits: int = 12
    top_k: int = 5
    seed: int = 0


@dataclass(slots=True)
class AssocSignalItem:
    item_id: str
    target_text: str
    partner_text: str
    distractor_texts: list[str]
    query: str


def generate(cfg: AssocSignalConfig) -> list[AssocSignalItem]:
    rng = random.Random(cfg.seed)
    items: list[AssocSignalItem] = []
    for i in range(cfg.num_items):
        target_tmpl, partner, distractors, query = rng.choice(THEMES)
        tok = rng.choice(TOKENS) + str(rng.randint(0, 9999))
        items.append(
            AssocSignalItem(
                item_id=f"association_signal_v1-{cfg.seed}-{i:03d}",
                target_text=target_tmpl.format(tok=tok),
                partner_text=partner,
                distractor_texts=list(distractors),
                query=query,
            )
        )
    return items


def _run_item(
    item: AssocSignalItem,
    cfg: AssocSignalConfig,
    store: MemoryStore,
    association: Association,
) -> float:
    target = store.put(item.target_text, source="assoc_target")
    partner = store.put(item.partner_text, source="assoc_partner")
    distractors = [store.put(t, source="assoc_distractor") for t in item.distractor_texts]

    for _ in range(cfg.co_hits):
        association.co_hit([target.id, partner.id])
    # a single co-hit for each distractor+partner just to prove it doesn't
    # dominate the target's edge.
    for d in distractors:
        association.co_hit([d.id, partner.id])

    combiner = (
        association.combiner if not isinstance(association, NullAssociation) else None
    )
    hits = store.query(
        item.query,
        top_k=max(cfg.top_k, len(distractors) + 2),
        context_keys=[partner.id],
        combiner=combiner,
    )
    # exclude the partner itself from the ranking, it's the "context" not a
    # candidate answer.
    ranking = [h for h in hits if h.entry.id != partner.id]
    for rank, h in enumerate(ranking, start=1):
        if h.entry.id == target.id:
            return 1.0 / rank
    return 0.0


def run(
    cfg: AssocSignalConfig,
    embedder: Embedder,
    association: Association | None = None,
) -> dict[str, Any]:
    """Run the task end-to-end; return per-item and aggregate MRR.

    ``association=None`` means :class:`NullAssociation` ablation (no boost).
    """
    assoc = association or NullAssociation()
    items = generate(cfg)

    rrs: list[float] = []
    for it in items:
        store = InMemoryStore(embedder=embedder)
        rr = _run_item(it, cfg, store, assoc)
        rrs.append(rr)

    return {
        "task": TASK_NAME,
        "association": assoc.__class__.__name__,
        "config": asdict(cfg),
        "mrr_mean": sum(rrs) / max(len(rrs), 1),
        "per_item": [
            {"item_id": it.item_id, "reciprocal_rank": rr}
            for it, rr in zip(items, rrs, strict=True)
        ],
    }


def compare_ablation(
    cfg: AssocSignalConfig, embedder: Embedder, out_path: Path | None = None
) -> dict[str, Any]:
    """Run with association on and off, return both results and delta."""
    on = run(cfg, embedder, HebbianAssociation(AssociationConfig()))
    off = run(cfg, embedder, NullAssociation())
    result = {
        "with_association": on,
        "without_association": off,
        "delta_mrr": on["mrr_mean"] - off["mrr_mean"],
    }
    if out_path is not None:
        import json

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


TASK_NAME = "association_signal_v1"
