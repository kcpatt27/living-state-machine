"""Train a linear learned verifier from exported dataset JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lsm.core.verifier_features import feature_names
from lsm.verifier_data.schema import load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Input verifier dataset JSONL.")
    parser.add_argument(
        "--out-model",
        type=Path,
        default=Path("models/verifier_linear_v1.joblib"),
        help="Output model artifact (.joblib).",
    )
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=Path("models/verifier_linear_v1_metrics.json"),
        help="Optional metrics output JSON.",
    )
    args = parser.parse_args()

    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise RuntimeError(
            "train_verifier_linear.py requires scikit-learn + joblib. "
            "Install with: uv add --dev scikit-learn joblib"
        ) from exc

    rows = load_rows(args.dataset)
    if not rows:
        raise RuntimeError(f"dataset is empty: {args.dataset}")
    names = feature_names()
    x = np.asarray(
        [[float(row.ctx_features.get(name, 0.0)) for name in names] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([row.label for row in rows], dtype=object)

    if len(rows) >= 20:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=0,
            stratify=y if len(set(y.tolist())) > 1 else None,
        )
    else:
        x_train, y_train = x, y
        x_test, y_test = x, y
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)
    train_acc = float(clf.score(x_train, y_train))
    test_acc = float(clf.score(x_test, y_test))

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "classes": [str(c) for c in clf.classes_.tolist()],
        "coef": clf.coef_.tolist(),
        "intercept": np.atleast_1d(clf.intercept_).tolist(),
        "feature_names": names,
        "temperature": 1.0,
    }
    joblib.dump(artifact, args.out_model)

    metrics = {
        "n_rows": len(rows),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "classes": artifact["classes"],
    }
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
