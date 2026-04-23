"""Tests for hidden state storage and updaters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from lsm.core.embeddings import HashEmbedder
from lsm.core.state import (
    LLMSummarizerUpdater,
    NullStateUpdater,
    PassthroughStateUpdater,
    State,
    StateStore,
    state_to_dict,
)
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _StaticBackend:
    name = "static"

    def __init__(self, reply: str) -> None:
        self.model = "static"
        self._reply = reply

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> str:
        return self._reply

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        yield self._reply

    async def close(self) -> None:
        return None


def test_store_roundtrip(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    assert store.load().is_empty()

    s = State(
        text="user likes oatmilk",
        vector=np.ones(8, dtype=np.float32),
        updated_at=123.0,
        revision=3,
        notes={"source": "test"},
    )
    store.save(s)

    loaded = store.load()
    assert loaded.text == s.text
    assert np.array_equal(loaded.vector, s.vector)
    assert loaded.updated_at == 123.0
    assert loaded.revision == 3
    assert loaded.notes == {"source": "test"}


def test_snapshot_and_restore(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.save(State(text="v1", vector=np.array([1.0], dtype=np.float32), revision=1))
    store.snapshot("v1")
    store.save(State(text="v2", vector=np.array([2.0], dtype=np.float32), revision=2))
    assert store.load().text == "v2"

    store.restore("v1")
    restored = store.load()
    assert restored.text == "v1"
    assert restored.revision == 1
    assert "v1" in store.list_snapshots()


def test_state_to_dict_omits_vector() -> None:
    s = State(text="x", vector=np.zeros(4, dtype=np.float32), revision=2)
    d = state_to_dict(s)
    assert "vector" not in d
    assert d["vector_dim"] == 4
    assert d["text"] == "x"
    assert d["revision"] == 2


@pytest.mark.asyncio
async def test_null_updater_returns_prev_unchanged() -> None:
    prev = State(text="before", vector=np.ones(4, dtype=np.float32), revision=5)
    out = await NullStateUpdater().update(prev, "u", "a")
    assert out is prev


@pytest.mark.asyncio
async def test_passthrough_updater_grows_and_embeds() -> None:
    emb = HashEmbedder(dim=32)
    upd = PassthroughStateUpdater(emb, max_chars=200)
    out = await upd.update(State(), "hello", "hi back")
    assert "hello" in out.text
    assert "hi back" in out.text
    assert out.vector.shape == (32,)
    assert out.revision == 1


@pytest.mark.asyncio
async def test_llm_summarizer_updates_text_and_vector() -> None:
    backend = _StaticBackend("user prefers dark mode; asked about pytorch speed.")
    emb = HashEmbedder(dim=16)
    upd = LLMSummarizerUpdater(backend=backend, embedder=emb)

    prev = State()
    out = await upd.update(prev, "i want dark mode", "sure, here's how...")
    assert "dark mode" in out.text
    assert out.vector.shape == (16,)
    assert out.revision == 1
    assert out.updated_at > 0


@pytest.mark.asyncio
async def test_llm_summarizer_preserves_prev_on_error() -> None:
    class _Broken:
        name = "broken"
        model = "x"

        async def generate(self, messages, options):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        async def stream(self, messages, options):  # type: ignore[no-untyped-def]
            if False:
                yield ""

        async def close(self):  # type: ignore[no-untyped-def]
            return None

    upd = LLMSummarizerUpdater(backend=_Broken(), embedder=HashEmbedder(dim=8))
    prev = State(text="keep me", vector=np.ones(8, dtype=np.float32), revision=1)
    out = await upd.update(prev, "u", "a")
    assert out is prev
