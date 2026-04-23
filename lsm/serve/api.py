"""FastAPI OpenAI-compatible server.

Endpoints:

- ``GET /health`` — liveness probe.
- ``GET /v1/models`` — list available models (single entry: the one we loaded).
- ``POST /v1/chat/completions`` — OpenAI-compatible chat. Streaming supported
  via ``stream=true`` (SSE).
- ``POST /v1/feedback`` — user thumbs / reward payload. L9 begins consuming
  this to modulate state updates; at L2 it is logged only.

The app is built lazily (factory pattern) so tests can inject fakes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from lsm.config import Settings, get_settings
from lsm.core.embeddings import get_default_embedder
from lsm.core.episodic import EpisodicLog, TurnRecord
from lsm.core.memory import MemoryStore, build_store
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.state import LLMSummarizerUpdater, StateStore, state_to_dict
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend, build_backend
from lsm.serve.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageSchema,
    Choice,
    ChoiceChunk,
    DeltaMessage,
    FeedbackRequest,
    FeedbackResponse,
    ModelInfo,
    ModelList,
    Usage,
)

log = logging.getLogger(__name__)


def _episodic_path(settings: Settings) -> Path:
    return settings.memory_dir / "episodes.sqlite"


def _build_pipeline(
    settings: Settings, session_id: str
) -> tuple[Pipeline, LLMBackend, EpisodicLog, StateStore, MemoryStore, "OutcomeLog"]:
    from lsm.core.outcome import HeuristicOutcomeScorer, OutcomeLog

    backend = build_backend(
        settings.backend, model=settings.model, ollama_host=settings.ollama_host
    )
    episodic = EpisodicLog(_episodic_path(settings))
    store = StateStore(settings.home)
    embedder = get_default_embedder(settings.embed_model)
    updater = LLMSummarizerUpdater(backend, embedder)
    memory = build_store(
        "lancedb",
        root=settings.memory_dir / "durable.lance",
        embedder=embedder,
    )
    outcome_log = OutcomeLog(settings.home / "outcomes.sqlite")
    scorer = HeuristicOutcomeScorer()
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(
            session_id=session_id,
            backend_name=settings.backend,
            model=settings.model,
        ),
        state_store=store,
        state_updater=updater,
        memory=memory,
        outcome_scorer=scorer,
        outcome_log=outcome_log,
    )
    return pipe, backend, episodic, store, memory, outcome_log


def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI application factory."""
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pipe, backend, episodic, store, memory, outcome_log = _build_pipeline(
            s, session_id="default"
        )
        app.state.pipeline = pipe
        app.state.backend = backend
        app.state.episodic = episodic
        app.state.state_store = store
        app.state.memory = memory
        app.state.outcome_log = outcome_log
        app.state.settings = s
        log.info("lsm api ready — backend=%s model=%s", s.backend, s.model)
        try:
            yield
        finally:
            memory.close()
            outcome_log.close()
            await backend.close()
            episodic.close()

    app = FastAPI(title="Living State Machine", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": s.model, "backend": s.backend}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelInfo(id=s.model)])

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        req: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
        pipe: Pipeline = app.state.pipeline
        options = GenerateOptions(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop_tuple(),
            seed=req.seed,
        )
        messages = [ChatMessage(role=m.role, content=m.content, name=m.name) for m in req.messages]

        if req.stream:
            return StreamingResponse(
                _sse_stream(pipe, messages, options, model=s.model),
                media_type="text/event-stream",
            )

        text = await pipe.generate(messages, options)
        return ChatCompletionResponse(
            model=s.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessageSchema(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )

    @app.get("/v1/state")
    async def get_state() -> dict[str, object]:
        pipe: Pipeline = app.state.pipeline
        return state_to_dict(pipe.state)

    @app.post("/v1/snapshot/{tag}")
    async def create_snapshot(tag: str) -> dict[str, str]:
        store: StateStore = app.state.state_store
        path = store.snapshot(tag)
        return {"tag": tag, "path": str(path)}

    @app.post("/v1/snapshot/{tag}/restore")
    async def restore_snapshot(tag: str) -> dict[str, str]:
        store: StateStore = app.state.state_store
        store.restore(tag)
        pipe: Pipeline = app.state.pipeline
        pipe._state = store.load()  # type: ignore[attr-defined]
        return {"tag": tag, "status": "restored"}

    @app.get("/v1/snapshots")
    async def list_snapshots() -> dict[str, list[str]]:
        store: StateStore = app.state.state_store
        return {"snapshots": store.list_snapshots()}

    @app.get("/v1/memory/count")
    async def memory_count() -> dict[str, int]:
        mem: MemoryStore = app.state.memory
        return {"count": mem.count()}

    @app.get("/v1/memory/query")
    async def memory_query(q: str, top_k: int = 5) -> dict[str, list[dict[str, object]]]:
        mem: MemoryStore = app.state.memory
        hits = mem.query(q, top_k=top_k)
        return {
            "results": [
                {
                    "id": h.entry.id,
                    "text": h.entry.text,
                    "score": h.score,
                    "components": h.components,
                    "source": h.entry.source,
                    "tags": h.entry.tags,
                }
                for h in hits
            ]
        }

    @app.post("/v1/feedback")
    async def feedback(payload: FeedbackRequest) -> FeedbackResponse:
        ep: EpisodicLog = app.state.episodic
        ep.append(
            TurnRecord(
                session_id=payload.session_id or "default",
                role="feedback",
                content=payload.reason or "",
                meta={
                    "score": payload.score,
                    "target_turn": payload.turn_id,
                    **payload.meta,
                },
            )
        )
        outcome_log = getattr(app.state, "outcome_log", None)
        if outcome_log is not None:
            from lsm.core.outcome import OutcomeRecord

            outcome_log.append(
                OutcomeRecord(
                    turn_id=payload.turn_id or "",
                    session_id=payload.session_id or "default",
                    score=float(payload.score),
                    source="user",
                    reasons=[payload.reason] if payload.reason else [],
                )
            )
        return FeedbackResponse(ok=True)

    return app


async def _sse_stream(
    pipe: Pipeline,
    messages: list[ChatMessage],
    options: GenerateOptions,
    *,
    model: str,
) -> AsyncIterator[bytes]:
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    first = ChatCompletionChunk(
        id=stream_id,
        created=created,
        model=model,
        choices=[
            ChoiceChunk(index=0, delta=DeltaMessage(role="assistant"), finish_reason=None)
        ],
    )
    yield _sse(first.model_dump())

    try:
        async for piece in pipe.stream(messages, options):
            chunk = ChatCompletionChunk(
                id=stream_id,
                created=created,
                model=model,
                choices=[ChoiceChunk(index=0, delta=DeltaMessage(content=piece))],
            )
            yield _sse(chunk.model_dump())
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("stream error")
        raise HTTPException(status_code=500, detail=str(e)) from e

    final = ChatCompletionChunk(
        id=stream_id,
        created=created,
        model=model,
        choices=[ChoiceChunk(index=0, delta=DeltaMessage(), finish_reason="stop")],
    )
    yield _sse(final.model_dump())
    yield b"data: [DONE]\n\n"


def _sse(obj: dict[str, object]) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()
