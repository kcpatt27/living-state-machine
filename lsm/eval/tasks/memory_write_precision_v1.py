"""memory_write_precision_v1 — does the writer keep signal and drop filler?

Generates a deterministic labelled corpus of candidate turns, mixing
"signal" turns (durable facts) and "filler" (greetings, affirmations,
off-topic chatter). The writer under test decides write/skip for each;
we grade against the label.

Metrics (per writer):

- precision: of the turns written, what fraction are signal?
- recall:    of the signal turns, what fraction are written?
- f1:        harmonic mean
- fpr:       of the filler turns, what fraction are written? (lower is better)

Comparison table is produced across: ``always`` (baseline), ``rules``,
``learned`` (trained on the same distribution). The ``never`` writer is
also reported for completeness (should score 0 recall).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lsm.core.writer import (
    AlwaysWriter,
    LabeledCandidate,
    LearnedWriter,
    LearnedWriterConfig,
    MemoryWriter,
    NeverWriter,
    RuleBasedWriter,
    WriteCandidate,
    train_learned_writer,
)

TASK_NAME = "memory_write_precision_v1"


# Signal templates — each produces a sentence that should be durably written.
_SIGNAL_TEMPLATES: list[str] = [
    "my name is {name}",
    "my favourite colour is {word}",
    "i prefer {word} tea over coffee",
    "i work at {name} as a software engineer",
    "i live in {name}",
    "call me {name}",
    "remember that our deployment target is {name}",
    "i am allergic to {word}",
    "note that the api key rotates every {num} days",
    "my dog's name is {name}",
    "i love hiking in the {word} mountains",
    "my partner's birthday is the {num}th of march",
    "i hate commuting by {word}",
    "my locker combo ends in {num}",
    "the office wifi password is {word}{num}",
    "i keep my keys in the {word} drawer",
    "our family car is a {word} wagon",
    "my therapist's office is on {name} street",
    "remember the server ip is 10.0.0.{num}",
    "my backup lives on the {word} drive",
]

# Filler — short affirmations and chatter that should not be written.
_FILLER: list[str] = [
    "ok",
    "okay cool",
    "thanks",
    "thank you",
    "got it",
    "sure",
    "yep",
    "nope",
    "hello",
    "hi there",
    "bye",
    "goodbye",
    "hmm",
    "interesting",
    "nice",
    "awesome",
    "makes sense",
    "alright",
    "ty",
    "kk",
    "lol",
    "lmao",
    "yeah sure",
    "oh ok",
    "understood",
    "?",
    "really?",
    "oh really",
    "huh",
    "why?",
]

_NAMES = ["azuki", "lumen", "myko", "quarto", "glade", "ember", "koto", "archive", "lunar", "pileus"]
_WORDS = ["blue", "green", "lavender", "obsidian", "cedar", "willow", "amber", "iron", "silk", "peppermint"]


@dataclass(slots=True)
class WritePrecisionConfig:
    n_signal: int = 30
    n_filler: int = 30
    seed: int = 0


@dataclass(slots=True)
class WritePrecisionItem:
    text: str
    label: int  # 1 = signal (should write), 0 = filler (should skip)


def generate(cfg: WritePrecisionConfig) -> list[WritePrecisionItem]:
    rng = random.Random(cfg.seed)
    items: list[WritePrecisionItem] = []
    for _ in range(cfg.n_signal):
        tmpl = rng.choice(_SIGNAL_TEMPLATES)
        text = tmpl.format(
            name=rng.choice(_NAMES),
            word=rng.choice(_WORDS),
            num=rng.randint(1, 99),
        )
        items.append(WritePrecisionItem(text=text, label=1))
    for _ in range(cfg.n_filler):
        items.append(WritePrecisionItem(text=rng.choice(_FILLER), label=0))
    rng.shuffle(items)
    return items


def _metrics(
    writer: MemoryWriter, items: list[WritePrecisionItem]
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    per_item: list[dict[str, Any]] = []
    for it in items:
        d = writer.decide(WriteCandidate(text=it.text, role="user"))
        wrote = bool(d.write)
        if wrote and it.label == 1:
            tp += 1
        elif wrote and it.label == 0:
            fp += 1
        elif not wrote and it.label == 1:
            fn += 1
        else:
            tn += 1
        per_item.append(
            {"text": it.text, "label": it.label, "wrote": wrote, "reason": d.reason}
        )
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    fpr = fp / max(fp + tn, 1)
    return {
        "writer": writer.name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_item": per_item,
    }


def _train(items: list[WritePrecisionItem]) -> LearnedWriter:
    labelled = [
        LabeledCandidate(
            candidate=WriteCandidate(text=it.text, role="user"), label=it.label
        )
        for it in items
    ]
    model = train_learned_writer(labelled, epochs=500, lr=0.5)
    return LearnedWriter(LearnedWriterConfig(threshold=0.5), model=model)


def run(
    cfg: WritePrecisionConfig,
    writers: list[MemoryWriter] | None = None,
) -> dict[str, Any]:
    train_items = generate(cfg)
    test_items = generate(
        WritePrecisionConfig(
            n_signal=cfg.n_signal, n_filler=cfg.n_filler, seed=cfg.seed + 10_000
        )
    )
    learned = _train(train_items)
    default = writers or [AlwaysWriter(), NeverWriter(), RuleBasedWriter(), learned]
    results = [_metrics(w, test_items) for w in default]
    return {
        "task": TASK_NAME,
        "config": asdict(cfg),
        "results": results,
    }


def compare(
    cfg: WritePrecisionConfig, out_path: Path | None = None
) -> dict[str, Any]:
    result = run(cfg)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
