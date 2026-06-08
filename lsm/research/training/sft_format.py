"""SFT JSONL → plain text for QLoRA (L15).

Supports ``text``, ``prompt``, and chat ``messages`` rows (the format emitted by
:func:`scripts.export_sft_from_lsm_home.export_sft`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def row_to_text(rec: dict[str, Any], tokenizer: Any) -> str:
    """Render one JSONL record to a single training string."""
    if rec.get("text"):
        return str(rec["text"])
    if rec.get("prompt"):
        return str(rec["prompt"])
    messages = rec.get("messages")
    if messages:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return ""


def load_sft_texts(
    path: Path,
    tokenizer: Any,
    *,
    max_rows: int | None = None,
) -> list[str]:
    """Load and format all non-empty rows from a JSONL file."""
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if max_rows is not None and len(texts) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = row_to_text(rec, tokenizer).strip()
            if text:
                texts.append(text)
    return texts


def iter_sft_texts(
    path: Path,
    tokenizer: Any,
    *,
    max_rows: int | None = None,
) -> Iterator[str]:
    """Stream formatted rows (loads via :func:`load_sft_texts`)."""
    yield from load_sft_texts(path, tokenizer, max_rows=max_rows)


__all__ = ["iter_sft_texts", "load_sft_texts", "row_to_text"]
