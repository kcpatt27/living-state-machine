from __future__ import annotations

import json
from pathlib import Path

from lsm.core.verifier_features import feature_names
from lsm.eval.tasks.verifier_calibration_v1 import (
    VerifierCalibrationConfig,
    compare_ablation,
)
from lsm.verifier_data.schema import VerifierDatasetRow, write_rows


def test_verifier_calibration_runs_rule_and_null(tmp_path: Path) -> None:
    ds = tmp_path / "dataset.jsonl"
    names = feature_names()
    zero = {n: 0.0 for n in names}
    rows = [
        VerifierDatasetRow(
            event_id="e1",
            source="reply",
            candidate="ok",
            ctx_text="",
            ctx_features=zero,
            rule_energy=0.0,
            rule_decision="accept",
            rule_violations=[],
            label_source="manual_gold",
            label="accept",
        ),
        VerifierDatasetRow(
            event_id="e2",
            source="reply",
            candidate="bad",
            ctx_text="",
            ctx_features={**zero, "rule_is_reject": 1.0},
            rule_energy=3.0,
            rule_decision="reject",
            rule_violations=["fact_contradiction"],
            label_source="manual_gold",
            label="reject",
        ),
    ]
    write_rows(ds, rows)
    result = compare_ablation(VerifierCalibrationConfig(dataset_path=ds))
    assert result["task"] == "verifier_calibration_v1"
    assert "rule" in result["variants"]
    assert "null" in result["variants"]


def test_verifier_calibration_linear_variant(tmp_path: Path) -> None:
    ds = tmp_path / "dataset.jsonl"
    names = feature_names()
    zero = {n: 0.0 for n in names}
    rows = [
        VerifierDatasetRow(
            event_id="e1",
            source="reply",
            candidate="ok",
            ctx_text="",
            ctx_features={**zero, "rule_is_accept": 1.0},
            rule_energy=0.0,
            rule_decision="accept",
            rule_violations=[],
            label_source="manual_gold",
            label="accept",
        ),
        VerifierDatasetRow(
            event_id="e2",
            source="reply",
            candidate="bad",
            ctx_text="",
            ctx_features={**zero, "rule_is_reject": 1.0},
            rule_energy=3.0,
            rule_decision="reject",
            rule_violations=["fact_contradiction"],
            label_source="manual_gold",
            label="reject",
        ),
    ]
    write_rows(ds, rows)
    idx_accept = names.index("rule_is_accept")
    idx_reject = names.index("rule_is_reject")
    coef = [[0.0 for _ in names] for _ in range(3)]
    coef[0][idx_accept] = 6.0
    coef[2][idx_reject] = 6.0
    art = {
        "classes": ["accept", "reprompt", "reject"],
        "coef": coef,
        "intercept": [0.0, 0.0, 0.0],
        "feature_names": names,
    }
    model = tmp_path / "linear.json"
    model.write_text(json.dumps(art), encoding="utf-8")
    result = compare_ablation(
        VerifierCalibrationConfig(dataset_path=ds, linear_model_path=model),
    )
    assert "linear" in result["variants"]
    assert "hybrid" in result["variants"]
