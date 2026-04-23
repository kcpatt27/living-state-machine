"""recall_pref_v1 — preference recall across distractor turns.

Each item is a synthetic dialogue:

1. Turn 0: the user states a preference ("my favourite X is Y").
2. Turns 1..N-2: unrelated distractor exchanges.
3. Turn N-1: the user asks "what is my favourite X?".

A system passes the item iff its answer to the final question contains the
preference value. We vary the preference template and the distractor density.

This task is the first memory-oriented benchmark built into the project. It
is intentionally small and reproducible so ablations are cheap to rerun.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field


PREFERENCES: list[tuple[str, list[str]]] = [
    ("favourite colour", ["cerulean", "chartreuse", "oxblood", "amber", "indigo"]),
    ("favourite drink", ["oatmilk latte", "matcha", "hibiscus tea", "negroni", "ginger kombucha"]),
    (
        "favourite programming language",
        ["rust", "ocaml", "zig", "elixir", "haskell"],
    ),
    ("favourite city", ["kyoto", "lisbon", "tbilisi", "medellín", "tallinn"]),
    ("home town", ["boise", "knoxville", "albany", "tampa", "pueblo"]),
    ("dog's name", ["basil", "kombu", "miso", "tamari", "furikake"]),
]

DISTRACTOR_PROMPTS: list[str] = [
    "what's a short tip for writing cleaner commit messages?",
    "explain BFS in two sentences.",
    "suggest a fun weekend project with a raspberry pi.",
    "what is the fourier transform, informally?",
    "pros and cons of using sqlite vs postgres for a side project?",
    "name three classic sci-fi novels.",
    "what is a reasonable morning stretching routine?",
    "difference between mutex and semaphore in one line.",
    "what's an interesting fact about the atlantic puffin?",
    "explain docker layers briefly.",
    "recommend a walking pace for a 5k warm-up.",
    "what are common causes of cache misses?",
    "how do i make a simple mise en place for pasta night?",
    "what's the story behind lisp parentheses?",
    "quickly: what is a monad, without category theory?",
]

DISTRACTOR_REPLIES: list[str] = [
    "sure, here are some thoughts on that topic.",
    "good question — one angle worth considering is the cost of context switches.",
    "that depends on what you value most: simplicity or throughput.",
    "short version: it trades memory for speed and is usually worth it.",
    "there are several schools of thought on this.",
    "practice matters more than the exact technique here.",
    "this is one of those things that is best tried in a small project first.",
]


@dataclass(slots=True)
class RecallItem:
    """A single benchmark example."""

    item_id: str
    topic: str
    correct: str
    dialogue: list[tuple[str, str]]  # (role, content)
    question: str
    required_keywords: list[str] = field(default_factory=list)

    def expected_contains(self) -> list[str]:
        if self.required_keywords:
            return self.required_keywords
        return [self.correct]


@dataclass(slots=True)
class RecallConfig:
    """Parameters controlling task generation."""

    num_items: int = 20
    distractor_turns: int = 30
    seed: int = 0


def generate(config: RecallConfig) -> list[RecallItem]:
    """Deterministically generate the benchmark set for a given config."""
    rng = random.Random(config.seed)
    items: list[RecallItem] = []
    for i in range(config.num_items):
        topic, options = rng.choice(PREFERENCES)
        correct = rng.choice(options)
        opening = f"just for context: my {topic} is {correct}. please remember that."
        opening_reply = f"got it — i'll remember your {topic} is {correct}."
        dialogue: list[tuple[str, str]] = [
            ("user", opening),
            ("assistant", opening_reply),
        ]
        for _ in range(config.distractor_turns):
            q = rng.choice(DISTRACTOR_PROMPTS)
            a = rng.choice(DISTRACTOR_REPLIES)
            dialogue.append(("user", q))
            dialogue.append(("assistant", a))
        question = f"what is my {topic}? answer in one word if possible."
        items.append(
            RecallItem(
                item_id=f"recall_pref_v1-{config.seed}-{i:03d}",
                topic=topic,
                correct=correct,
                dialogue=dialogue,
                question=question,
                required_keywords=[correct],
            )
        )
    return items


def iter_items(config: RecallConfig) -> Iterator[RecallItem]:
    yield from generate(config)


TASK_NAME = "recall_pref_v1"
