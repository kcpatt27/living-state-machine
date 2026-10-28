"""Tests for hidden state storage and updaters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from lsm.core.embeddings import HashEmbedder
from lsm.core.facts import (
    CAT_USER_PREFERENCES,
    Fact,
    FactStore,
    format_structured_facts_block,
)
from lsm.core.identity import IdentityProfile
from lsm.core.memory import InMemoryStore, MemoryEntry
from lsm.core.state import (
    LLMSummarizerUpdater,
    NullStateUpdater,
    PassthroughStateUpdater,
    State,
    StateStore,
    _looks_like_additive,
    _looks_like_correction,
    _pin_identity_line,
    _sanitize_people_names,
    state_to_dict,
)
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _StaticBackend:
    name = "static"

    def __init__(self, reply: str) -> None:
        self.model = "static"
        self._reply = reply

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
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


class _CapturingBackend:
    """Backend that records the last messages sent to ``generate``."""

    name = "capturing"
    model = "capturing"

    def __init__(self, reply: str = "summary") -> None:
        self._reply = reply
        self.last_messages: list[ChatMessage] = []

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        self.last_messages = list(messages)
        return self._reply

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        self.last_messages = list(messages)
        yield self._reply

    async def close(self) -> None:
        return None


def _user_body(messages: list[ChatMessage]) -> str:
    user = next((m for m in messages if m.role == "user"), None)
    assert user is not None, "summarizer prompt missing user message"
    return user.content


@pytest.mark.asyncio
async def test_summarizer_prompt_labels_assistant_as_unverified() -> None:
    backend = _CapturingBackend("ok")
    upd = LLMSummarizerUpdater(backend=backend, embedder=HashEmbedder(dim=8))
    await upd.update(State(), "i like dark mode", "noted")

    body = _user_body(backend.last_messages)
    assert "New user message (authoritative):" in body
    assert "unverified paraphrase" in body
    assert "i like dark mode" in body
    assert "noted" in body


@pytest.mark.asyncio
async def test_summarizer_prompt_marks_correction_turn() -> None:
    backend = _CapturingBackend("ok")
    upd = LLMSummarizerUpdater(backend=backend, embedder=HashEmbedder(dim=8))
    await upd.update(State(text="prior"), "actually, Crimson Desert is an action RPG", "ok")

    body = _user_body(backend.last_messages)
    assert "User is correcting prior summary" in body


@pytest.mark.asyncio
async def test_summarizer_non_correction_omits_override_line() -> None:
    backend = _CapturingBackend("ok")
    upd = LLMSummarizerUpdater(backend=backend, embedder=HashEmbedder(dim=8))
    await upd.update(State(text="prior"), "hello there", "hi")

    body = _user_body(backend.last_messages)
    assert "User is correcting prior summary" not in body


@pytest.mark.asyncio
async def test_summarizer_assistant_as_facts_keeps_legacy_shape() -> None:
    backend = _CapturingBackend("ok")
    upd = LLMSummarizerUpdater(
        backend=backend, embedder=HashEmbedder(dim=8), assistant_as_facts=True
    )
    await upd.update(State(), "u", "a")

    body = _user_body(backend.last_messages)
    assert "Latest exchange:" in body
    assert "unverified paraphrase" not in body
    assert "New user message (authoritative)" not in body


@pytest.mark.parametrize(
    "text",
    [
        "actually, that's wrong",
        "not a strategy game",
        "i meant action RPG",
        "Correction: it's Crimson Desert",
        "to clarify, the title is different",
        "that's wrong",
        "that isn't right",
    ],
)
def test_correction_regex_matches_expected_phrases(text: str) -> None:
    assert _looks_like_correction(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello there",
        "i love Crimson Desert and strategy games",
        "thanks for the help",
    ],
)
def test_correction_regex_rejects_normal_phrases(text: str) -> None:
    assert _looks_like_correction(text) is False


@pytest.mark.asyncio
async def test_summarizer_pins_identity_slot() -> None:
    profile = IdentityProfile(name="Elsum", description="Pinned assistant.")
    backend = _CapturingBackend("identity/role: Kalen\npeople/names: Elsum\npreferences/likes: tea")
    upd = LLMSummarizerUpdater(backend=backend, embedder=HashEmbedder(dim=8), identity=profile)
    out = await upd.update(State(), "hi", "hello")
    assert out.text.startswith("identity/role: Elsum - Pinned assistant.")
    assert "identity/role: Kalen" not in out.text


@pytest.mark.asyncio
async def test_summarizer_injects_identity_block() -> None:
    profile = IdentityProfile(name="Elsum", description="Helper bot.")
    backend = _CapturingBackend("ok")
    upd = LLMSummarizerUpdater(backend=backend, embedder=HashEmbedder(dim=8), identity=profile)
    await upd.update(State(), "u", "a")
    body = _user_body(backend.last_messages)
    assert "Pinned identity (do NOT alter" in body
    assert "name: Elsum" in body
    assert "description: Helper bot." in body


@pytest.mark.asyncio
async def test_summarizer_identity_none_is_legacy() -> None:
    backend = _CapturingBackend("ok")
    upd = LLMSummarizerUpdater(backend=backend, embedder=HashEmbedder(dim=8), identity=None)
    await upd.update(State(), "u", "a")
    body = _user_body(backend.last_messages)
    assert "Pinned identity" not in body


@pytest.mark.asyncio
async def test_summarizer_includes_correction_memories() -> None:
    memory = InMemoryStore(embedder=HashEmbedder(dim=8))
    memory.put_entry(
        MemoryEntry(
            text="Crimson Desert is an action RPG, not a strategy game.",
            tags=["correction"],
        )
    )
    backend = _CapturingBackend("summary")
    upd = LLMSummarizerUpdater(
        backend=backend,
        embedder=HashEmbedder(dim=8),
        memory=memory,
        correction_top_k=4,
    )
    await upd.update(
        State(text="games/media: Crimson Desert (strategy game)"),
        "ok",
        "noted",
    )
    body = _user_body(backend.last_messages)
    assert "Known prior corrections" in body
    assert "action RPG" in body


@pytest.mark.asyncio
async def test_summarizer_omits_corrections_block_when_none() -> None:
    memory = InMemoryStore(embedder=HashEmbedder(dim=8))
    memory.put("plain fact", tags=["chat"])
    backend = _CapturingBackend("summary")
    upd = LLMSummarizerUpdater(backend=backend, embedder=HashEmbedder(dim=8), memory=memory)
    await upd.update(State(), "u", "a")
    body = _user_body(backend.last_messages)
    assert "Known prior corrections" not in body


def test_pin_identity_line_replaces_wrong_slot() -> None:
    profile = IdentityProfile(name="Elsum", description="AI.")
    text = "identity/role: Kalen\npeople/names: Elsum"
    fixed = _pin_identity_line(text, profile)
    assert fixed.startswith("identity/role: Elsum - AI.")
    assert "identity/role: Kalen" not in fixed


def test_sanitize_people_names_removes_assistant() -> None:
    text = "people/names: Kalen (the user), Elsum"
    fixed = _sanitize_people_names(text, "Elsum")
    assert "Elsum" not in fixed.split("people/names:")[1]


@pytest.mark.asyncio
async def test_summarizer_injects_structured_facts_block(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_PREFERENCES,
            subject="restaurants",
            value="yakiniku",
        )
    )
    backend = _CapturingBackend("summary")
    upd = LLMSummarizerUpdater(
        backend=backend,
        embedder=HashEmbedder(dim=8),
        fact_store=store,
    )
    await upd.update(State(), "u", "a")
    assert "Structured facts" in _user_body(backend.last_messages)
    assert "yakiniku" in _user_body(backend.last_messages)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("i like both yakiniku and korean bbq", True),
        ("hello there", False),
        ('please make it "korean bbq and yakiniku"', True),
    ],
)
def test_looks_like_additive(text: str, expected: bool) -> None:
    assert _looks_like_additive(text) is expected


def test_format_structured_facts_block() -> None:
    f = Fact(owner="user", category=CAT_USER_PREFERENCES, value="tea")
    block = format_structured_facts_block([f])
    assert "Structured facts" in block
    assert "tea" in block
