"""Tests for prompt composition."""

from __future__ import annotations

from lsm.runtime.llm_backend import ChatMessage
from lsm.runtime.prompt_compose import BASE_SYSTEM_PROMPT, PromptContext, compose


def test_empty_context_only_base_system() -> None:
    user = ChatMessage(role="user", content="hi")
    out = compose(PromptContext(), [user])
    assert len(out) == 2
    assert out[0].role == "system"
    assert BASE_SYSTEM_PROMPT in out[0].content
    assert out[1] is user


def test_all_sections_rendered() -> None:
    ctx = PromptContext(
        state_projection="user likes oatmilk",
        retrieved=["mem A", "mem B"],
        fast_scratchpad="short-term stuff",
        outcome_critique="last answer was too long",
    )
    user = ChatMessage(role="user", content="hi")
    [sys_msg, _] = compose(ctx, [user])
    text = sys_msg.content
    assert "user likes oatmilk" in text
    assert "mem A" in text
    assert "mem B" in text
    assert "short-term stuff" in text
    assert "last answer was too long" in text
