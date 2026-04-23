"""OpenAI-compatible request/response schemas.

We stick close enough to the OpenAI spec that Open WebUI, Continue.dev, and
similar clients work without adapters. Fields we don't support are ignored
rather than rejected, and we never emit fields not in the spec.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessageSchema(BaseModel):
    """OpenAI-style message."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI /v1/chat/completions request we accept."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessageSchema]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int | None = 1024
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None
    user: str | None = None

    def stop_tuple(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        if isinstance(self.stop, str):
            return (self.stop,)
        return tuple(self.stop)


class Choice(BaseModel):
    index: int = 0
    message: ChatMessageSchema
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChoiceChunk(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChoiceChunk]


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "lsm"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class FeedbackRequest(BaseModel):
    """Simple thumbs-up/down payload, used by L9."""

    model_config = ConfigDict(extra="ignore")

    turn_id: str | None = None
    session_id: str | None = None
    score: float = Field(ge=-1.0, le=1.0)
    reason: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    ok: bool = True
