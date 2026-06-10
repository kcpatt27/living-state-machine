"""LLM backend adapters.

Two adapters implement the same :class:`LLMBackend` protocol:

- :class:`OllamaBackend` — calls a local Ollama daemon. This is the daily-use
  path: fast, low-memory, easy to swap models.
- :class:`TransformersBackend` — loads a HuggingFace model locally. This is
  the research path when we need access to logits or hidden states (e.g., for
  L14 when the custom core consumes internal LLM signals). It is lazy-imported
  because `transformers` + `torch` is a heavy optional extra.

The two backends are wire-compatible for generate/stream. Anything that only
works for one (e.g., logits hooks) lives on the concrete class, not in the
protocol.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUS = frozenset({500, 502, 503, 504})


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Return ``True`` when an HTTP error is transient and worth retrying."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    code = exc.response.status_code if exc.response is not None else None
    return code in _RETRYABLE_HTTP_STATUS


@dataclass(slots=True, frozen=True)
class ChatMessage:
    """OpenAI-style chat message."""

    role: str
    content: str
    name: str | None = None

    def to_ollama(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass(slots=True, frozen=True)
class GenerateOptions:
    """Generation knobs we forward to the backend."""

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int | None = 1024
    stop: tuple[str, ...] = ()
    seed: int | None = None

    def to_ollama(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.max_tokens is not None:
            opts["num_predict"] = self.max_tokens
        if self.stop:
            opts["stop"] = list(self.stop)
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface every inference backend must implement."""

    name: str
    model: str

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str: ...

    def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]: ...

    async def close(self) -> None: ...


class OllamaBackend:
    """Adapter for a local Ollama daemon.

    Uses HTTP directly (rather than the `ollama` python client) so we have full
    control over streaming, retries, and timeouts.
    """

    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout_s: float = 300.0,
    ) -> None:
        self.model = model
        self._host = host.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0),
            retry=retry_if_exception_type(
                (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)
            )
            | retry_if_exception(_is_retryable_http_error),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.post(f"{self._host}{path}", json=payload)
                resp.raise_for_status()
                return resp
        raise RuntimeError("unreachable")

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_ollama() for m in messages],
            "stream": False,
            "options": options.to_ollama(),
        }
        resp = await self._post("/api/chat", payload)
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_ollama() for m in messages],
            "stream": True,
            "options": options.to_ollama(),
        }
        yielded_any = False

        def _retryable_stream_error(exc: BaseException) -> bool:
            # Never retry after we have already emitted user-visible tokens:
            # retried streams would duplicate output.
            if yielded_any:
                return False
            if isinstance(exc, httpx.ConnectError | httpx.ReadError | httpx.RemoteProtocolError):
                return True
            return _is_retryable_http_error(exc)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0),
            retry=retry_if_exception(_retryable_stream_error),
            reraise=True,
        ):
            with attempt:
                async with self._client.stream("POST", f"{self._host}/api/chat", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            import json as _json

                            chunk = _json.loads(line)
                        except ValueError:
                            continue
                        piece = chunk.get("message", {}).get("content", "")
                        if piece:
                            yielded_any = True
                            yield piece
                        if chunk.get("done"):
                            return

    async def tags(self) -> list[str]:
        """Return the list of locally available model tags."""
        r = await self._client.get(f"{self._host}/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    async def ensure_model(self, model: str | None = None) -> None:
        """Pull the model if it isn't present locally. No-op if already there."""
        target = model or self.model
        tags = await self.tags()
        if target in tags:
            return
        log.info("pulling ollama model %s ...", target)
        async with self._client.stream(
            "POST", f"{self._host}/api/pull", json={"name": target, "stream": True}
        ) as r:
            r.raise_for_status()
            async for _ in r.aiter_lines():
                pass

    async def unload(self, model: str | None = None) -> None:
        """Ask Ollama to unload the model (free VRAM)."""
        target = model or self.model
        try:
            await self._client.post(
                f"{self._host}/api/generate",
                json={"model": target, "keep_alive": 0},
            )
        except httpx.HTTPError:
            log.warning("failed to unload %s from ollama", target)

    async def close(self) -> None:
        await self._client.aclose()


class TransformersBackend:
    """HuggingFace Transformers backend (lazy-imported).

    This path is only used when we need access to logits or hidden states. It
    is slower and heavier than Ollama for daily use, so is never the default.
    """

    name = "transformers"

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_context: int = 8192,
        adapter_path: str | None = None,
    ) -> None:
        self.model = model
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._max_context = max_context
        self._adapter_path = adapter_path
        self._tokenizer: Any = None
        self._model: Any = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover — optional extra
            raise RuntimeError(
                "TransformersBackend requires the 'research' extra. "
                "Install with: uv sync --extra research"
            ) from e

        kwargs: dict[str, Any] = {"device_map": self._device}
        if self._load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:  # pragma: no cover
                log.warning("bitsandbytes not available; loading full precision")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)
        if self._adapter_path:
            from peft import PeftModel

            log.info("loading LoRA adapter from %s", self._adapter_path)
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)

    def _format(self, messages: Iterable[ChatMessage]) -> str:
        self._lazy_load()
        assert self._tokenizer is not None
        raw = [{"role": m.role, "content": m.content} for m in messages]
        return self._tokenizer.apply_chat_template(raw, tokenize=False, add_generation_prompt=True)

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        return await asyncio.to_thread(self._generate_sync, list(messages), options)

    def _generate_sync(self, messages: list[ChatMessage], options: GenerateOptions) -> str:
        import torch

        prompt = self._format(messages)
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=options.temperature > 0,
                temperature=max(options.temperature, 1e-5),
                top_p=options.top_p,
                top_k=options.top_k,
                max_new_tokens=options.max_tokens or 1024,
            )
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return text

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        # Simplification: we generate fully and yield once. True token streaming
        # via TextIteratorStreamer is added in L14 when we need to inspect
        # hidden states as tokens come out.
        text = await self.generate(messages, options)
        yield text

    async def close(self) -> None:
        self._model = None
        self._tokenizer = None


def build_backend(
    backend: str,
    model: str,
    *,
    ollama_host: str = "http://localhost:11434",
    device: str = "cuda",
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
) -> LLMBackend:
    """Factory used by the server, CLI, and eval runner."""
    if backend == "ollama":
        return OllamaBackend(model=model, host=ollama_host)
    if backend == "transformers":
        return TransformersBackend(
            model=model,
            device=device,
            load_in_4bit=load_in_4bit,
            adapter_path=adapter_path,
        )
    raise ValueError(f"unknown backend: {backend!r}")
