"""Negative control: shuffle the labels in a committed run and re-score it.

A scorer that is actually reading the model's answer must collapse when the
correct value is swapped between items. If accuracy stays high, the evaluation
is passing answers for the wrong reason.

    python scripts/negative_control.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "v1 LSM": "experiments/20260529-174517-sample_recall_pref_v1_lsm/raw_outputs.jsonl",
    "v2 LSM": "experiments/20260530-072409-sample_recall_pref_v2_lsm/raw_outputs.jsonl",
}


def passed(answer: str, correct: str) -> bool:
    """Same rule as lsm.eval.metrics.any_keyword_match."""
    return correct.lower() in (answer or "").lower()


def score(rows: list[dict], labels: list[str]) -> float:
    if not rows:
        return 0.0
    hits = sum(
        passed(row.get("answer", ""), label) for row, label in zip(rows, labels, strict=False)
    )
    return hits / len(rows)


def main() -> int:
    failures = 0
    for name, relative in RUNS.items():
        path = ROOT / relative
        if not path.exists():
            print(f"MISSING {relative}")
            failures += 1
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        real = [row["correct_value"] for row in rows]
        shuffled = real[:]
        random.Random(0).shuffle(shuffled)

        original = score(rows, real)
        control = score(rows, shuffled)
        ok = control <= original + 0.05
        status = "PASS" if ok else "FAIL"
        print(f"{status} {name}: original {original:.3f} -> shuffled {control:.3f} (n={len(rows)})")
        failures += 0 if ok else 1

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
