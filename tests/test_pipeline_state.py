"""Tests for pipeline integration with state (L3)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.state import LLMSummarizerUpdater, State, StateStore
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _Backend:
    name = "fake"

    def __init__(self, replies: list[str]) -> None:
        self.model = "fake"
        self._replies = list(replies)
        self.prompts: list[list[ChatMessage]] = []

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> str:
        self.prompts.append(list(messages))
        return self._replies.pop(0)

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        self.prompts.append(list(messages))
        yield self._replies.pop(0)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_pipeline_updates_and_injects_state(tmp_path: Path) -> None:
    # first backend reply is the assistant turn, second is the summarizer output
    backend = _Backend(
        replies=["hi, dark mode is on", "User wants dark mode enabled."]
    )
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    store = StateStore(tmp_path / "state")
    updater = LLMSummarizerUpdater(backend=backend, embedder=HashEmbedder(dim=8))

    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(session_id="s", backend_name="fake", model="fake"),
        state_store=store,
        state_updater=updater,
    )
    # first turn: state is empty, so only the base system prompt is injected
    reply1 = await pipe.generate(
        [ChatMessage(role="user", content="dark mode please")], GenerateOptions()
    )
    assert reply1 == "hi, dark mode is on"
    s1 = store.load()
    assert "dark mode" in s1.text.lower()
    assert s1.vector.size == 8
    assert s1.revision == 1

    # feed a second reply so another full turn works
    backend._replies = ["OK", "User wants dark mode. Asked for status."]
    await pipe.generate(
        [ChatMessage(role="user", content="status?")], GenerateOptions()
    )
    s2 = store.load()
    assert s2.revision == 2
    # second composed prompt should have included the projected state text
    second_turn_system = backend.prompts[2][0].content  # prompt 2 is turn 2's user-facing call
    assert "dark mode" in second_turn_system.lower()


@pytest.mark.asyncio
async def test_pipeline_without_updater_keeps_empty_state(tmp_path: Path) -> None:
    backend = _Backend(replies=["ok"])
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    store = StateStore(tmp_path / "state")
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(session_id="s", backend_name="fake", model="fake"),
        state_store=store,
        state_updater=None,
    )
    await pipe.generate([ChatMessage(role="user", content="hi")], GenerateOptions())
    assert store.load().is_empty()


def test_state_persist_roundtrip_on_pipeline_reload(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    store.save(State(text="preloaded", vector=np.ones(4, dtype=np.float32), revision=9))

    pipe = Pipeline(
        backend=_Backend(replies=[]),
        episodic=EpisodicLog(tmp_path / "e.sqlite"),
        config=PipelineConfig(session_id="s", backend_name="fake", model="fake"),
        state_store=store,
        state_updater=None,
    )
    assert pipe.state.text == "preloaded"
    assert pipe.state.revision == 9
