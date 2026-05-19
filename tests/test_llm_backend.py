"""Regression tests for Ollama backend retry behavior."""

from __future__ import annotations

import httpx
import pytest
from lsm.runtime.llm_backend import (
    ChatMessage,
    GenerateOptions,
    OllamaBackend,
    _is_retryable_http_error,
)


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://localhost:11434/api/chat")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"status {code}", request=req, response=resp)


def test_retryable_http_error_helper() -> None:
    assert _is_retryable_http_error(_status_error(500))
    assert _is_retryable_http_error(_status_error(503))
    assert not _is_retryable_http_error(_status_error(404))
    assert not _is_retryable_http_error(RuntimeError("x"))


@pytest.mark.asyncio
async def test_generate_retries_on_http_500(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OllamaBackend(model="dummy", host="http://localhost:11434", timeout_s=5.0)
    calls = {"n": 0}

    async def fake_post(url: str, *, json: dict[str, object]) -> httpx.Response:
        _ = json
        calls["n"] += 1
        req = httpx.Request("POST", url)
        if calls["n"] == 1:
            return httpx.Response(500, request=req, json={"error": "transient"})
        return httpx.Response(
            200,
            request=req,
            json={"message": {"role": "assistant", "content": "ok"}},
        )

    monkeypatch.setattr(backend._client, "post", fake_post)
    try:
        out = await backend.generate(
            [ChatMessage(role="user", content="hello")],
            GenerateOptions(),
        )
    finally:
        await backend.close()
    assert out == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_generate_does_not_retry_on_http_400(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OllamaBackend(model="dummy", host="http://localhost:11434", timeout_s=5.0)
    calls = {"n": 0}

    async def fake_post(url: str, *, json: dict[str, object]) -> httpx.Response:
        _ = json
        calls["n"] += 1
        req = httpx.Request("POST", url)
        if calls["n"] == 1:
            return httpx.Response(400, request=req, json={"error": "bad request"})
        return httpx.Response(
            200,
            request=req,
            json={"message": {"role": "assistant", "content": "should-not-happen"}},
        )

    monkeypatch.setattr(backend._client, "post", fake_post)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await backend.generate(
                [ChatMessage(role="user", content="hello")],
                GenerateOptions(),
            )
    finally:
        await backend.close()
    assert calls["n"] == 1
