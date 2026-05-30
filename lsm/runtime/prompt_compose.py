"""Deterministic prompt assembly.

At L2 this is a thin pass-through: the user's message list is forwarded to the
backend with only the base system prompt prepended. Subsequent levels inject:

- L3: a compact textual projection of ``s_t``.
- L4: the top-k retrieved memories.
- L5/L6: habituated / associated context conditioning.
- L8: fast-tier scratchpad.
- L9: outcome critique from the previous turn.

The function signature is stable; only the internals grow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from lsm.runtime.llm_backend import ChatMessage

BASE_SYSTEM_PROMPT = (
    "You are a local personal AI assistant. You have access to a persistent "
    "hidden state and long-term memory that you should use to stay consistent "
    "across turns. Prefer concise, correct, helpful answers. When you are "
    "unsure, say so rather than guess.\n\n"
    "Fact handling:\n"
    "- Treat the user's listed items as separate; do not collapse unrelated "
    "items in a comma list into one category.\n"
    "- Distinguish a title (e.g. a specific game name) from a genre (e.g. "
    "strategy, action RPG); never assert a title is a genre or vice versa.\n"
    "- When summarizing the user's facts back to them, use separate bullets; "
    "if a relationship between two items is unclear, ask one short clarifying "
    "question instead of guessing.\n"
    "- Your own name or persona (whatever the user calls you) is not a fact "
    "about the user.\n"
    "- Do not treat similar categories as synonyms (e.g. yakiniku and Korean "
    "BBQ are distinct). When the user says both or uses and between items, "
    "list each separately."
)


@dataclass(slots=True)
class PromptContext:
    """Everything that will be composed into a final message list.

    All fields default to empty strings so L2 only uses the base system prompt.
    """

    base_system: str = BASE_SYSTEM_PROMPT
    state_projection: str = ""
    structured_facts: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    fast_scratchpad: str = ""
    outcome_critique: str = ""


def compose(ctx: PromptContext, user_messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Return the final message list to send to the LLM backend."""
    system_parts: list[str] = [ctx.base_system]

    if ctx.state_projection:
        system_parts.append("Current internal state (summary):\n" + ctx.state_projection)

    if ctx.structured_facts:
        bullets = "\n".join(f"- {line}" for line in ctx.structured_facts)
        system_parts.append(
            "Structured facts (authoritative; prefer over episodic memories):\n" + bullets
        )

    if ctx.retrieved:
        bullets = "\n".join(f"- {r}" for r in ctx.retrieved)
        system_parts.append("Relevant memories:\n" + bullets)

    if ctx.fast_scratchpad:
        system_parts.append("Scratchpad:\n" + ctx.fast_scratchpad)

    if ctx.outcome_critique:
        system_parts.append("Previous-turn self-critique:\n" + ctx.outcome_critique)

    sys_msg = ChatMessage(role="system", content="\n\n".join(system_parts))
    return [sys_msg, *user_messages]
