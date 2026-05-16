"""Tests for the FastAPI server. Uses a fake backend to avoid needing Ollama."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Sequence

import pytest
from fastapi.testclient import TestClient
from lsm.config import get_settings
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class FakeBackend:
    name = "fake"

    def __init__(self, reply: str = "hi from fake") -> None:
        self.model = "fake-model"
        self._reply = reply

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        return self._reply

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        for ch in self._reply:
            yield ch

    async def close(self) -> None:
        return None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from lsm.runtime import pipeline_factory
    from lsm.serve import api as api_mod

    monkeypatch.setattr(pipeline_factory, "build_backend", lambda *a, **k: FakeBackend())

    settings = get_settings()
    app = api_mod.create_app(settings)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


def test_chat_completions_non_stream(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "hi from fake"


def test_chat_completions_stream(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        lines = [line for line in r.iter_lines() if line]

    payload_lines = [line for line in lines if line.startswith("data: ") and "[DONE]" not in line]
    assert payload_lines, lines
    # first chunk carries the role, subsequent chunks carry content deltas
    first = json.loads(payload_lines[0][len("data: ") :])
    assert first["choices"][0]["delta"]["role"] == "assistant"

    content_deltas: list[str] = []
    for line in payload_lines[1:]:
        chunk = json.loads(line[len("data: ") :])
        piece = chunk["choices"][0]["delta"].get("content")
        if piece:
            content_deltas.append(piece)
    assert "".join(content_deltas) == "hi from fake"
    assert lines[-1] == "data: [DONE]"


def test_feedback(client: TestClient) -> None:
    r = client.post(
        "/v1/feedback",
        json={"turn_id": "t1", "session_id": "default", "score": 1.0, "reason": "good"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}
