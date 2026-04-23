"""Scoring primitives shared across tasks."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable


def exact_keyword_match(answer: str, required: Iterable[str]) -> bool:
    """True iff every token in ``required`` appears in ``answer`` (case-insensitive)."""
    hay = answer.lower()
    return all(w.lower() in hay for w in required)


def any_keyword_match(answer: str, options: Iterable[str]) -> bool:
    hay = answer.lower()
    return any(w.lower() in hay for w in options)


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def accuracy(results: list[bool]) -> float:
    return sum(1 for r in results if r) / max(len(results), 1)


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)
