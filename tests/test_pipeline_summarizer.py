"""Pipeline + summarizer integration: correction memories reach the updater prompt."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.facts import CAT_USER_MEDIA, Fact, FactStore
from lsm.core.identity import IdentityProfile
from lsm.core.memory import InMemoryStore, MemoryEntry
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.projection import DeterministicProjection
from lsm.core.state import FactsBackedUpdater, LLMSummarizerUpdater
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _CapturingBackend:
    name = "capturing"
    model = "capturing"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.all_messages: list[list[ChatMessage]] = []

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        self.all_messages.append(list(messages))
        return self._reply

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        self.all_messages.append(list(messages))
        yield self._reply

    async def close(self) -> None:
        return None

    def summarizer_bodies(self) -> list[str]:
        bodies: list[str] = []
        for msgs in self.all_messages:
            if len(msgs) >= 2 and msgs[0].role == "system":
                user = next((m for m in msgs if m.role == "user"), None)
                if user is not None:
                    bodies.append(user.content)
        return bodies


@pytest.mark.asyncio
async def test_pipeline_summarizer_sees_correction_memory(tmp_path: Path) -> None:
    memory = InMemoryStore(embedder=HashEmbedder(dim=64))
    memory.put_entry(
        MemoryEntry(
            text="Crimson Desert is an action RPG, not a strategy game.",
            tags=["correction"],
        )
    )
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    backend = _CapturingBackend("acknowledged")
    identity = IdentityProfile(name="Elsum", description="Test assistant.")
    updater = LLMSummarizerUpdater(
        backend,
        HashEmbedder(dim=64),
        identity=identity,
        memory=memory,
    )
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(
            session_id="s",
            backend_name="capturing",
            model="capturing",
            retrieve_top_k=3,
            retrieve_min_score=-1.0,
        ),
        memory=memory,
        state_updater=updater,
    )
    try:
        await pipe.generate(
            [ChatMessage(role="user", content="what genre is Crimson Desert?")],
            GenerateOptions(),
        )
    finally:
        episodic.close()

    bodies = backend.summarizer_bodies()
    assert bodies, "expected at least one summarizer call"
    assert any("Known prior corrections" in b for b in bodies)
    assert any("action RPG" in b for b in bodies)
    assert any("Pinned identity" in b for b in bodies)


@pytest.mark.asyncio
async def test_pipeline_facts_backed_updater_renders_without_summarizer_call(
    tmp_path: Path,
) -> None:
    memory = InMemoryStore(embedder=HashEmbedder(dim=64))
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    backend = _CapturingBackend("acknowledged")
    identity = IdentityProfile(name="Elsum", description="Test assistant.")
    facts = FactStore(tmp_path)
    facts.sync_assistant_identity(identity)
    facts.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="action RPG",
        )
    )
    updater = FactsBackedUpdater(
        HashEmbedder(dim=64),
        DeterministicProjection(),
        identity=identity,
        memory=memory,
        fact_store=facts,
    )
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(
            session_id="s",
            backend_name="capturing",
            model="capturing",
            retrieve_top_k=3,
            retrieve_min_score=-1.0,
        ),
        memory=memory,
        state_updater=updater,
        fact_store=facts,
    )
    try:
        await pipe.generate(
            [ChatMessage(role="user", content="what genre is Crimson Desert?")],
            GenerateOptions(),
        )
    finally:
        episodic.close()

    # No extra LLM call for summarizer prompt; only the main reply call.
    assert len(backend.summarizer_bodies()) == 1
    assert "Prior summary:" not in backend.summarizer_bodies()[0]
    assert "games/media: " in pipe.state.text
