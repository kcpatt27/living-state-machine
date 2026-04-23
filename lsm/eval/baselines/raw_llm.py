"""Raw-LLM baseline: stuff the full dialogue into one prompt and ask.

This is the natural "no memory, no retrieval" baseline. If our system does
not beat this, we've proven nothing.
"""

from __future__ import annotations

from lsm.eval.tasks.recall_pref_v1 import RecallItem
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend


class RawLLMBaseline:
    name = "raw_llm"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        max_context_turns: int = 0,
        options: GenerateOptions | None = None,
    ) -> None:
        self.backend = backend
        self.max_context_turns = max_context_turns
        self.options = options or GenerateOptions(temperature=0.0, max_tokens=64)

    async def answer(self, item: RecallItem) -> str:
        history: list[ChatMessage] = []
        dialogue = item.dialogue
        if self.max_context_turns > 0:
            dialogue = dialogue[-2 * self.max_context_turns :]
        for role, content in dialogue:
            history.append(ChatMessage(role=role, content=content))
        history.append(ChatMessage(role="user", content=item.question))
        return await self.backend.generate(history, self.options)

    async def close(self) -> None:
        await self.backend.close()
