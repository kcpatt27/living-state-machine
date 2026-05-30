"""Integration tests for F.4 fact retrieval + correction memory reconcile."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.facts import CAT_USER_MEDIA, Fact, FactStore
from lsm.core.memory import InMemoryStore
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.projection import DeterministicProjection
from lsm.core.state import FactsBackedUpdater, StateStore
from lsm.core.verifier import RuleBasedVerifier
from lsm.core.verifier_log import VerifierLog
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions
from lsm.runtime.prompt_compose import compose


class _OneShotBackend:
    name = "scripted"
    model = "scripted"

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        _ = messages, options
        return self._reply

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        _ = messages, options
        yield self._reply

    async def close(self) -> None:
        return None


def _pipe(tmp_path: Path, memory: InMemoryStore, facts: FactStore) -> Pipeline:
    embedder = HashEmbedder()
    store = StateStore(tmp_path / "home")
    updater = FactsBackedUpdater(
        embedder,
        DeterministicProjection(),
        fact_store=facts,
    )
    cfg = PipelineConfig(
        session_id="t",
        backend_name="scripted",
        model="scripted",
        write_assistant_to_memory=True,
        retrieve_facts_enabled=True,
        correction_memory_reconcile_enabled=True,
    )
    return Pipeline(
        backend=_OneShotBackend("ok"),
        episodic=EpisodicLog(tmp_path / "episodes.sqlite"),
        config=cfg,
        state_store=store,
        state_updater=updater,
        memory=memory,
        fact_store=facts,
        embedder=embedder,
        verifier=RuleBasedVerifier(),
        verifier_log=VerifierLog(tmp_path / "verifier_log.sqlite"),
    )


@pytest.mark.asyncio
async def test_structured_facts_in_prompt_context(tmp_path: Path) -> None:
    facts = FactStore(tmp_path / "home")
    facts.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="action RPG",
        )
    )
    memory = InMemoryStore(HashEmbedder())
    pipe = _pipe(tmp_path, memory, facts)
    ctx = pipe._context("What genre is Crimson Desert?")
    assert ctx.structured_facts
    composed = compose(ctx, [ChatMessage(role="user", content="hi")])
    assert "Structured facts" in composed[0].content
    assert "Crimson Desert" in composed[0].content


@pytest.mark.asyncio
async def test_correction_reconcile_deletes_stale_assistant_memory(tmp_path: Path) -> None:
    facts = FactStore(tmp_path / "home")
    facts.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="action RPG",
        )
    )
    embedder = HashEmbedder()
    memory = InMemoryStore(embedder)
    stale = memory.put(
        "Crimson Desert is a strategy game",
        tags=["assistant", "cli"],
    )
    _ = stale
    assert memory.count() == 1

    pipe = _pipe(tmp_path, memory, facts)
    user_correction = "Actually Crimson Desert is not a strategy game, it's an action RPG."
    pipe._maybe_extract_facts(user_correction)

    assert memory.count() == 0


@pytest.mark.asyncio
async def test_reply_gate_logs_verifier_event(tmp_path: Path) -> None:
    facts = FactStore(tmp_path / "home")
    facts.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="action RPG",
        )
    )
    memory = InMemoryStore(HashEmbedder())
    pipe = _pipe(
        tmp_path,
        memory,
        facts,
    )
    pipe.backend = _OneShotBackend("Crimson Desert is a strategy game")  # type: ignore[assignment]
    await pipe.generate(
        [ChatMessage(role="user", content="What is Crimson Desert?")],
        GenerateOptions(),
    )
    vlog = VerifierLog(tmp_path / "verifier_log.sqlite")
    events = vlog.recent(limit=5)
    vlog.close()
    assert any(e.source == "reply" for e in events)
