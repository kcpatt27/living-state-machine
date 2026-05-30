"""Tests for F.4 structured-fact retrieval."""

from __future__ import annotations

from pathlib import Path

from lsm.core.embeddings import HashEmbedder
from lsm.core.fact_retrieval import fact_lines_for_prompt, score_facts_for_query
from lsm.core.facts import CAT_USER_MEDIA, Fact, FactStore


def test_score_facts_ranks_relevant_subject(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="action RPG",
        )
    )
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Elden Ring",
            value="action RPG",
        )
    )
    embedder = HashEmbedder()
    hits = score_facts_for_query(
        store, "tell me about Crimson Desert", embedder, top_k=4, min_score=0.0
    )
    assert len(hits) >= 1
    subjects = {f.subject for f, _ in hits}
    assert subjects <= {"Crimson Desert", "Elden Ring"}


def test_fact_lines_for_prompt_prefix(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="action RPG",
        )
    )
    lines = fact_lines_for_prompt(
        store, "Crimson Desert genre", HashEmbedder(), top_k=2, min_score=0.0
    )
    assert len(lines) >= 1
    assert lines[0].startswith("[fact]")
