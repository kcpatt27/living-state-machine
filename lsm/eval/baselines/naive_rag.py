"""Naive RAG baseline.

For each dialogue, embed every prior turn once, then at question time retrieve
the top-k most similar turns and put them into the system prompt. No memory
state, no habituation, no association.
"""

from __future__ import annotations

from lsm.core.embeddings import Embedder
from lsm.core.memory import InMemoryStore
from lsm.eval.tasks.recall_pref_v1 import RecallItem
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend


class NaiveRAGBaseline:
    name = "naive_rag"

    def __init__(
        self,
        backend: LLMBackend,
        embedder: Embedder,
        *,
        top_k: int = 5,
        options: GenerateOptions | None = None,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.top_k = top_k
        self.options = options or GenerateOptions(temperature=0.0, max_tokens=64)

    async def answer(self, item: RecallItem) -> str:
        store = InMemoryStore(self.embedder)
        for role, content in item.dialogue:
            store.put(f"{role}: {content}")
        hits = store.query(item.question, top_k=self.top_k)
        bullets = "\n".join(f"- {h.entry.text}" for h in hits)
        system = (
            "Use the relevant past snippets below to answer the user's "
            "question precisely. If the answer is in the snippets, state it "
            "directly. If not, say you don't know.\n\n"
            f"Relevant snippets:\n{bullets}"
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=item.question),
        ]
        return await self.backend.generate(messages, self.options)
