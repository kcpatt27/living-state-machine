"""Integration: pipeline routes writes through the gated writer."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.memory import InMemoryStore
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.writer import NeverWriter, RuleBasedWriter
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _NoopBackend:
    name = "fake"
    model = "fake-model"

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        return "ok"

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        yield "ok"

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_never_writer_suppresses_all_writes(tmp_path: Path) -> None:
    memory = InMemoryStore(embedder=HashEmbedder(dim=64))
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    pipe = Pipeline(
        backend=_NoopBackend(),
        episodic=episodic,
        config=PipelineConfig(session_id="s", backend_name="fake", model="fake-model"),
        memory=memory,
        writer=NeverWriter(),
    )
    await pipe.generate([ChatMessage(role="user", content="my name is Kalen")], GenerateOptions())
    assert memory.count() == 0
    assert pipe.last_write_decisions and all(not d.write for d in pipe.last_write_decisions)
    episodic.close()


@pytest.mark.asyncio
async def test_rules_writer_keeps_signal_drops_filler(tmp_path: Path) -> None:
    memory = InMemoryStore(embedder=HashEmbedder(dim=64))
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    pipe = Pipeline(
        backend=_NoopBackend(),
        episodic=episodic,
        config=PipelineConfig(session_id="s", backend_name="fake", model="fake-model"),
        memory=memory,
        writer=RuleBasedWriter(),
    )
    await pipe.generate([ChatMessage(role="user", content="my name is Kalen")], GenerateOptions())
    assert memory.count() == 1

    await pipe.generate([ChatMessage(role="user", content="ok")], GenerateOptions())
    assert memory.count() == 1

    await pipe.generate(
        [ChatMessage(role="user", content="i prefer green tea over coffee")],
        GenerateOptions(),
    )
    assert memory.count() == 2
    episodic.close()
