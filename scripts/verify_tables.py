"""Recompute Tables 1-2 from the committed run artifacts.

No model, no GPU, no network: reads `experiments/*/metrics.json` and checks the
values printed in the paper. Exit 1 on any mismatch.

    python scripts/verify_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 0.011

# run directory -> (accuracy_mean, accuracy_std) as printed in the paper
EXPECTED = {
    # Table 1, recall_pref_v1
    "20260529-174501-sample_recall_pref_v1_naive_rag": (0.100, 0.100),
    "20260530-081245-sample_recall_pref_v1_raw_llm_matched": (0.000, 0.000),
    "20260529-174440-sample_recall_pref_v1_raw_llm": (1.000, 0.000),
    "20260529-174517-sample_recall_pref_v1_lsm": (1.000, 0.000),
    # Table 1, recall_pref_v2
    "20260530-072347-sample_recall_pref_v2_naive_rag": (0.044, 0.038),
    "20260530-081310-sample_recall_pref_v2_raw_llm_matched": (0.000, 0.000),
    "20260530-072304-sample_recall_pref_v2_raw_llm": (0.911, 0.038),
    "20260530-072409-sample_recall_pref_v2_lsm": (0.089, 0.038),
    # Table 2, ablations on recall_pref_v1
    "20260529-182227-sample_recall_pref_v1_lsm_no_habituation": (0.967, 0.058),
    "20260529-184120-sample_recall_pref_v1_lsm_no_writer": (0.933, 0.115),
    "20260529-190018-sample_recall_pref_v1_lsm_flat_memory": (0.933, 0.058),
    "20260529-180413-sample_recall_pref_v1_lsm_no_assoc": (1.000, 0.000),
    "20260529-191925-sample_recall_pref_v1_lsm_multi_memory": (1.000, 0.000),
}

COMPONENT_FILES = [
    "experiments/memory_write_precision_v1/result.json",
    "experiments/verifier_calibration_v1/result.json",
]


def main() -> int:
    failures = 0
    print("Table 1-2 verification (committed artifacts)\n")
    for run_id, (mean, std) in EXPECTED.items():
        path = ROOT / "experiments" / run_id / "metrics.json"
        if not path.exists():
            print(f"  MISSING {run_id}")
            failures += 1
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        got_mean = float(data["accuracy_mean"])
        got_std = float(data["accuracy_std"])
        ok = abs(got_mean - mean) <= TOLERANCE and abs(got_std - std) <= TOLERANCE
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status} {run_id}\n"
            f"       got {got_mean:.3f} ± {got_std:.3f}  expected {mean:.3f} ± {std:.3f}"
        )
        failures += 0 if ok else 1

    print("\nComponent benches")
    for relative in COMPONENT_FILES:
        path = ROOT / relative
        if path.exists():
            print(f"  PASS {relative}")
        else:
            print(f"  MISSING {relative}")
            failures += 1

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
