"""recall_pref_v2 — harder preference recall with decoys and repeated filler.

Extends ``recall_pref_v1`` with:

- longer distractor gaps (default 45 turns vs 20),
- decoy preference statements about the **same topic** with wrong values,
- repeated identical filler prompts to stress habituation,
- more items per seed (default 20).

Designed to separate LSM ablations that ``recall_pref_v1`` leaves tied at 1.0.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from lsm.eval.tasks.recall_pref_v1 import (
    DISTRACTOR_PROMPTS,
    DISTRACTOR_REPLIES,
    PREFERENCES,
    RecallConfig,
    RecallItem,
)

TASK_NAME = "recall_pref_v2"


def _decoy_line(rng: random.Random, topic: str, wrong: str) -> str:
    templates = [
        f"random note: my coworker's {topic} is {wrong}, not that it matters.",
        f"btw someone told me their {topic} is {wrong} — weird, right?",
        f"for the record my sister's {topic} is {wrong}.",
        f"unrelated: i read that a celebrity's {topic} is {wrong}.",
    ]
    return rng.choice(templates)


def generate(config: RecallConfig) -> list[RecallItem]:
    """Deterministically generate the v2 benchmark set."""
    rng = random.Random(config.seed)
    decoy_rate = float(getattr(config, "decoy_rate", 0.35))
    repeat_every = int(getattr(config, "repeat_every", 6))
    items: list[RecallItem] = []
    for i in range(config.num_items):
        topic, options = rng.choice(PREFERENCES)
        correct = rng.choice(options)
        wrong_pool = [o for o in options if o != correct]
        opening = f"just for context: my {topic} is {correct}. please remember that."
        opening_reply = f"got it — i'll remember your {topic} is {correct}."
        dialogue: list[tuple[str, str]] = [
            ("user", opening),
            ("assistant", opening_reply),
        ]
        sticky_prompt: str | None = None
        repeat_remaining = 0
        for _turn_idx in range(config.distractor_turns):
            if repeat_remaining > 0 and sticky_prompt is not None:
                q = sticky_prompt
                repeat_remaining -= 1
            elif rng.random() < decoy_rate and wrong_pool:
                q = _decoy_line(rng, topic, rng.choice(wrong_pool))
            else:
                q = rng.choice(DISTRACTOR_PROMPTS)
                if rng.random() < 0.22:
                    sticky_prompt = q
                    repeat_remaining = max(1, repeat_every - 1)
            a = rng.choice(DISTRACTOR_REPLIES)
            dialogue.append(("user", q))
            dialogue.append(("assistant", a))
        question = f"what is my {topic}? answer in one word if possible."
        items.append(
            RecallItem(
                item_id=f"recall_pref_v2-{config.seed}-{i:03d}",
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
