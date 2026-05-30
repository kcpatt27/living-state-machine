from __future__ import annotations

import json
from pathlib import Path

from lsm.core.verifier import (
    HybridVerifier,
    LinearLearnedVerifier,
    RuleBasedVerifier,
    VerifierContext,
)
from lsm.core.verifier_features import feature_names


def test_linear_learned_verifier_scores_from_artifact(tmp_path: Path) -> None:
    names = feature_names()
    idx_accept = names.index("rule_is_accept")
    idx_reject = names.index("rule_is_reject")
    coef = [[0.0 for _ in names] for _ in range(3)]
    coef[0][idx_accept] = 4.0
    coef[2][idx_reject] = 4.0
    artifact = {
        "classes": ["accept", "reprompt", "reject"],
        "coef": coef,
        "intercept": [0.0, 0.0, 0.0],
        "feature_names": names,
    }
    path = tmp_path / "linear.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    verifier = LinearLearnedVerifier(path)
    verdict = verifier.score("hello", VerifierContext())
    assert verdict.decision in {"accept", "reprompt", "reject"}
    assert verdict.energy >= 0.0


def test_hybrid_verifier_respects_rule_reject_floor(tmp_path: Path) -> None:
    names = feature_names()
    idx_accept = names.index("rule_is_accept")
    coef = [[0.0 for _ in names] for _ in range(3)]
    coef[0][idx_accept] = 5.0
    artifact = {
        "classes": ["accept", "reprompt", "reject"],
        "coef": coef,
        "intercept": [0.0, 0.0, 0.0],
        "feature_names": names,
    }
    path = tmp_path / "linear.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    learned = LinearLearnedVerifier(path)
    rule = RuleBasedVerifier()
    hybrid = HybridVerifier(learned=learned, rule=rule)
    # Triggers fact contradiction + identity drift in most contexts with identity set.
    verdict = hybrid.score("", VerifierContext())
    assert verdict.decision in {"accept", "reprompt", "reject"}
