"""Tests for the per-turn pipeline at L2."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from lsm.core.episodic import EpisodicLog
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class FakeBackend:
    name = "fake"

    def __init__(self, reply: str = "OK") -> None:
        self.model = "fake-model"
        self._reply = reply
        self.last_messages: list[ChatMessage] = []

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> str:
        self.last_messages = list(messages)
        return self._reply

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        self.last_messages = list(messages)
        for ch in self._reply:
            yield ch

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_generate_logs_user_and_assistant(tmp_path: Path) -> None:
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    backend = FakeBackend(reply="hi there")
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(session_id="s", backend_name="fake", model="fake-model"),
    )
    reply = await pipe.generate(
        [ChatMessage(role="user", content="hello")], GenerateOptions()
    )
    assert reply == "hi there"
    rows = episodic.session("s")
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[0].content == "hello"
    assert rows[1].content == "hi there"
    # composed prompt must prepend a system message
    assert backend.last_messages[0].role == "system"
    episodic.close()


@pytest.mark.asyncio
async def test_stream_accumulates_and_logs(tmp_path: Path) -> None:
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    backend = FakeBackend(reply="hey")
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(session_id="s", backend_name="fake", model="fake-model"),
    )
    chunks: list[str] = []
    async for c in pipe.stream(
        [ChatMessage(role="user", content="yo")], GenerateOptions()
    ):
        chunks.append(c)
    assert "".join(chunks) == "hey"
    rows = episodic.session("s")
    assert rows[-1].content == "hey"
    episodic.close()
