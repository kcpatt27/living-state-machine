"""Tests for SFT JSONL formatting (no GPU)."""

from __future__ import annotations

from lsm.research.training.sft_format import load_sft_texts, row_to_text


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(parts)


def test_row_to_text_plain() -> None:
    tok = _FakeTokenizer()
    assert row_to_text({"text": "hello"}, tok) == "hello"
    assert row_to_text({"prompt": "ping"}, tok) == "ping"


def test_row_to_text_messages() -> None:
    tok = _FakeTokenizer()
    out = row_to_text(
        {
            "messages": [
                {"role": "user", "content": "I prefer negroni."},
                {"role": "assistant", "content": "Got it."},
            ]
        },
        tok,
    )
    assert "user: I prefer negroni." in out
    assert "assistant: Got it." in out


def test_load_sft_texts_max_rows(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"text": "a"}\n{"text": "b"}\n{"text": "c"}\n',
        encoding="utf-8",
    )
    tok = _FakeTokenizer()
    assert load_sft_texts(path, tok, max_rows=2) == ["a", "b"]
