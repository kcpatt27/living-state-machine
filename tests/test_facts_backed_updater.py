"""Tests for F.2 FactsBackedUpdater."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from lsm.core.embeddings import HashEmbedder
from lsm.core.facts import CAT_USER_MEDIA, Fact, FactStore
from lsm.core.identity import IdentityProfile
from lsm.core.projection import DeterministicProjection, LLMPolishedProjection
from lsm.core.state import FactsBackedUpdater, State
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _CountingBackend:
    name = "counting"
    model = "counting"

    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        self.generate_calls += 1
        return "polished"

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        self.generate_calls += 1
        yield "polished"

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_facts_backed_updater_changes_with_fact_updates(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    ident = IdentityProfile(name="Elsum", description="Assistant.")
    store.sync_assistant_identity(ident)
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="strategy game",
        )
    )
    upd = FactsBackedUpdater(
        HashEmbedder(dim=32),
        DeterministicProjection(),
        identity=ident,
        fact_store=store,
    )
    s0 = State()
    s1 = await upd.update(s0, "i like Crimson Desert", "ack")
    assert "strategy game" in s1.text.lower()
    assert s1.vector.shape == (32,)
    assert s1.revision == 1

    store.replace_by_subject(
        owner="user",
        category=CAT_USER_MEDIA,
        subject="Crimson Desert",
        new_value="action RPG",
    )
    s2 = await upd.update(s1, "actually not strategy", "ack")
    assert "action rpg" in s2.text.lower()
    assert "strategy game" not in s2.text.lower()
    assert s2.revision == 2


@pytest.mark.asyncio
async def test_facts_backed_updater_carries_notes(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    ident = IdentityProfile(name="Elsum", description="Assistant.")
    upd = FactsBackedUpdater(
        HashEmbedder(dim=16),
        DeterministicProjection(),
        identity=ident,
        fact_store=store,
    )
    prev = State(
        text="x",
        vector=np.ones(16, dtype=np.float32),
        revision=1,
        notes={"rolling": "remember this"},
    )
    out = await upd.update(prev, "hi", "hey")
    assert out.notes.get("rolling") == "remember this"


@pytest.mark.asyncio
async def test_disabled_polish_wrapper_makes_zero_llm_calls(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    ident = IdentityProfile(name="Elsum", description="Assistant.")
    store.sync_assistant_identity(ident)
    backend = _CountingBackend()
    projection = LLMPolishedProjection(
        base=DeterministicProjection(),
        backend=backend,
        enabled=False,
    )
    upd = FactsBackedUpdater(
        HashEmbedder(dim=16),
        projection,
        identity=ident,
        fact_store=store,
    )
    out = await upd.update(State(), "hello", "world")
    assert out.text.startswith("identity/role:")
    assert backend.generate_calls == 0
