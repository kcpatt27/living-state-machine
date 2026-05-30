"""Tests for the deterministic prompt assembly."""

from __future__ import annotations

from lsm.runtime.llm_backend import ChatMessage
from lsm.runtime.prompt_compose import BASE_SYSTEM_PROMPT, PromptContext, compose


def test_base_prompt_contains_list_and_title_genre_rules() -> None:
    # Substring regressions: future edits must not silently drop these rules.
    assert "do not collapse unrelated items" in BASE_SYSTEM_PROMPT
    assert "Distinguish a title" in BASE_SYSTEM_PROMPT
    assert "ask one short clarifying question" in BASE_SYSTEM_PROMPT
    assert "is not a fact about the user" in BASE_SYSTEM_PROMPT


def test_compose_preserves_user_messages_order() -> None:
    user_msgs = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="reply-1"),
        ChatMessage(role="user", content="second"),
    ]
    out = compose(PromptContext(), user_msgs)

    assert out[0].role == "system"
    assert out[0].content == BASE_SYSTEM_PROMPT
    assert [m.content for m in out[1:]] == ["first", "reply-1", "second"]
    assert [m.role for m in out[1:]] == ["user", "assistant", "user"]


def test_compose_injects_state_and_memories() -> None:
    ctx = PromptContext(
        state_projection="reishi likes dark mode",
        retrieved=["IDE theme: Abyss", "favourite game: Crimson Desert"],
    )
    out = compose(ctx, [ChatMessage(role="user", content="hi")])

    sys = out[0].content
    assert "Current internal state" in sys
    assert "reishi likes dark mode" in sys
    assert "Relevant memories" in sys
    assert "IDE theme: Abyss" in sys
    assert "favourite game: Crimson Desert" in sys


def test_compose_injects_structured_facts_before_memories() -> None:
    ctx = PromptContext(
        structured_facts=["[fact] [user.media] Crimson Desert: action RPG"],
        retrieved=["old assistant paraphrase"],
    )
    out = compose(ctx, [ChatMessage(role="user", content="hi")])
    sys = out[0].content
    assert "Structured facts" in sys
    assert "authoritative" in sys
    assert "Crimson Desert" in sys
    assert "Relevant memories" in sys
    assert sys.index("Structured facts") < sys.index("Relevant memories")
