"""Bootstrap SFT JSONL from recall_pref-style dialogues (L15 dataset v0).

Use when ``export_sft_from_lsm_home.py`` yields fewer than 500 rows. Rows use
extractable phrasing (``I prefer …``) and memory-adjacent assistant replies.

Usage::

    uv run python scripts/bootstrap_sft_recall_pref.py
    uv run python scripts/bootstrap_sft_recall_pref.py --count 500 --out data/sft_bootstrap.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from lsm.eval.tasks.recall_pref_v1 import DISTRACTOR_PROMPTS, DISTRACTOR_REPLIES, PREFERENCES

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "sft_bootstrap.jsonl"


def _preference_pair(topic: str, value: str, rng: random.Random) -> dict[str, object]:
    user_line = rng.choice(
        [
            f"I prefer {value}.",
            f"I prefer {value} for my {topic}.",
            f"my {topic} is {value}.",
        ]
    )
    assistant_line = rng.choice(
        [
            f"Got it — I'll remember you prefer {value}.",
            f"Noted: {value} for your {topic}.",
            f"Understood. Your {topic} is {value}.",
        ]
    )
    return {
        "messages": [
            {"role": "user", "content": user_line},
            {"role": "assistant", "content": assistant_line},
        ],
        "source": "bootstrap_recall_pref",
    }


def _recall_pair(topic: str, value: str, rng: random.Random) -> dict[str, object]:
    question = rng.choice(
        [
            f"what is my {topic}?",
            f"what's my {topic}?",
            f"do you remember my {topic}?",
        ]
    )
    assistant_line = rng.choice(
        [
            f"Your {topic} is {value}.",
            f"You prefer {value} for your {topic}.",
            f"You told me {value} — that's your {topic}.",
        ]
    )
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_line},
        ],
        "source": "bootstrap_recall_pref",
    }


def _distractor_pair(rng: random.Random) -> dict[str, object]:
    prompt = rng.choice(DISTRACTOR_PROMPTS)
    reply = rng.choice(DISTRACTOR_REPLIES)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ],
        "source": "bootstrap_recall_pref",
    }


def build_bootstrap_rows(*, count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    topics = list(PREFERENCES)
    while len(rows) < count:
        topic, values = rng.choice(topics)
        value = rng.choice(values)
        rows.append(_preference_pair(topic, value, rng))
        if len(rows) >= count:
            break
        if rng.random() < 0.35:
            rows.append(_distractor_pair(rng))
        if len(rows) >= count:
            break
        rows.append(_recall_pair(topic, value, rng))
    return rows[:count]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = build_bootstrap_rows(count=args.count, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {"out": str(args.out), "rows": len(rows), "seed": args.seed}
    log.info("wrote %d bootstrap rows -> %s", len(rows), args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
