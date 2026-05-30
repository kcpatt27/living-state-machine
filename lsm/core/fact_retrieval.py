"""F.4 — retrieve structured facts alongside episodic memory."""

from __future__ import annotations

import numpy as np

from lsm.core.embeddings import Embedder
from lsm.core.facts import Fact, FactStore


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_facts_for_query(
    store: FactStore,
    query: str,
    embedder: Embedder,
    *,
    top_k: int = 4,
    min_score: float = 0.12,
) -> list[tuple[Fact, float]]:
    """Rank facts by embedding similarity to ``query``."""
    if not query.strip():
        return []
    facts = store.all_facts()
    if not facts:
        return []
    q_vec = embedder.embed(query)
    scored: list[tuple[Fact, float]] = []
    for fact in facts:
        if fact.category == "assistant.identity":
            continue
        line = fact.display_line()
        score = _cosine(q_vec, embedder.embed(line))
        if score >= min_score:
            scored.append((fact, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, top_k)]


def fact_lines_for_prompt(
    store: FactStore,
    query: str,
    embedder: Embedder,
    *,
    top_k: int = 4,
    min_score: float = 0.12,
) -> list[str]:
    """Lines to inject into the prompt as structured-fact retrieval."""
    hits = score_facts_for_query(store, query, embedder, top_k=top_k, min_score=min_score)
    return [f"[fact] {fact.display_line().lstrip('- ')}" for fact, _ in hits]


__all__ = ["fact_lines_for_prompt", "score_facts_for_query"]
