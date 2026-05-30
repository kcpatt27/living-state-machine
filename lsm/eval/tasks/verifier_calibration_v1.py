"""verifier_calibration_v1 — decision calibration for verifier variants."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lsm.core.verifier_features import feature_names
from lsm.verifier_data.schema import VerifierDatasetRow, load_rows

TASK_NAME = "verifier_calibration_v1"
Decision = Literal["accept", "reprompt", "reject"]

_LABELS: tuple[Decision, ...] = ("accept", "reprompt", "reject")


@dataclass(slots=True)
class VerifierCalibrationConfig:
    dataset_path: Path
    linear_model_path: Path | None = None
    encoder_model_path: Path | None = None
    encoder_tokenizer_path: Path | None = None
    reject_floor: float = 2.5


def run(cfg: VerifierCalibrationConfig) -> dict[str, Any]:
    rows = load_rows(cfg.dataset_path)
    if not rows:
        raise RuntimeError(f"dataset is empty: {cfg.dataset_path}")
    variants: dict[str, dict[str, Any]] = {}
    variants["null"] = _evaluate_variant(rows, "null", cfg)
    variants["rule"] = _evaluate_variant(rows, "rule", cfg)
    if cfg.linear_model_path is not None and cfg.linear_model_path.exists():
        variants["linear"] = _evaluate_variant(rows, "linear", cfg)
        variants["hybrid"] = _evaluate_variant(rows, "hybrid", cfg)
    if (
        cfg.encoder_model_path is not None
        and cfg.encoder_model_path.exists()
        and cfg.encoder_tokenizer_path is not None
        and cfg.encoder_tokenizer_path.exists()
    ):
        variants["encoder"] = _evaluate_variant(rows, "encoder", cfg)
        if "hybrid" not in variants:
            variants["hybrid"] = _evaluate_variant(rows, "hybrid", cfg)
    return {
        "task": TASK_NAME,
        "config": _config_to_json(cfg),
        "n_rows": len(rows),
        "variants": variants,
    }


def compare_ablation(
    cfg: VerifierCalibrationConfig, out_path: Path | None = None
) -> dict[str, Any]:
    result = run(cfg)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _evaluate_variant(
    rows: list[VerifierDatasetRow], variant: str, cfg: VerifierCalibrationConfig
) -> dict[str, Any]:
    truth = [row.label for row in rows]
    preds: list[Decision] = []
    p_accepts: list[float] = []
    if variant == "null":
        preds = ["accept"] * len(rows)
        p_accepts = [1.0] * len(rows)
    elif variant == "rule":
        preds = [row.rule_decision for row in rows]
        p_accepts = [math.exp(-max(float(row.rule_energy), 0.0)) for row in rows]
    elif variant == "linear":
        probs = _linear_probs(rows, cfg.linear_model_path)
        preds = [_LABELS[int(np.argmax(p))] for p in probs]
        p_accepts = [float(p[0]) for p in probs]
    elif variant == "encoder":
        probs = _encoder_probs(rows, cfg.encoder_model_path, cfg.encoder_tokenizer_path)
        preds = [_LABELS[int(np.argmax(p))] for p in probs]
        p_accepts = [float(p[0]) for p in probs]
    elif variant == "hybrid":
        p_accepts = []
        for row in rows:
            if float(row.rule_energy) >= float(cfg.reject_floor):
                preds.append(row.rule_decision)
                p_accepts.append(math.exp(-max(float(row.rule_energy), 0.0)))
                continue
            if cfg.linear_model_path is not None and cfg.linear_model_path.exists():
                lp = _linear_probs([row], cfg.linear_model_path)[0]
                preds.append(_LABELS[int(np.argmax(lp))])
                p_accepts.append(float(lp[0]))
            elif (
                cfg.encoder_model_path is not None
                and cfg.encoder_model_path.exists()
                and cfg.encoder_tokenizer_path is not None
                and cfg.encoder_tokenizer_path.exists()
            ):
                ep = _encoder_probs([row], cfg.encoder_model_path, cfg.encoder_tokenizer_path)[0]
                preds.append(_LABELS[int(np.argmax(ep))])
                p_accepts.append(float(ep[0]))
            else:
                preds.append(row.rule_decision)
                p_accepts.append(math.exp(-max(float(row.rule_energy), 0.0)))
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    return _metrics(truth, [str(p) for p in preds], p_accepts)


def _metrics(truth: list[str], pred: list[str], p_accept: list[float]) -> dict[str, Any]:
    n = max(1, len(truth))
    correct = sum(1 for t, p in zip(truth, pred, strict=True) if t == p)
    per: dict[str, dict[str, float]] = {}
    macro_f1 = 0.0
    for label in _LABELS:
        tp = sum(1 for t, p in zip(truth, pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, pred, strict=True) if t == label and p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        per[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(1 for t in truth if t == label)),
        }
        macro_f1 += f1
    macro_f1 /= len(_LABELS)
    y_accept = [1.0 if t == "accept" else 0.0 for t in truth]
    brier = sum((p - y) ** 2 for p, y in zip(p_accept, y_accept, strict=True)) / n
    auc = _roc_auc_binary(y_accept, p_accept)
    return {
        "accuracy": correct / n,
        "macro_f1": macro_f1,
        "brier_accept": brier,
        "roc_auc_accept_vs_not": auc,
        "per_decision": per,
    }


def _roc_auc_binary(y_true: list[float], y_score: list[float]) -> float:
    paired = list(zip(y_true, y_score, strict=True))
    pos = [s for y, s in paired if y == 1.0]
    neg = [s for y, s in paired if y == 0.0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _linear_probs(rows: list[VerifierDatasetRow], model_path: Path | None) -> list[np.ndarray]:
    if model_path is None:
        raise ValueError("linear model path is required")
    art = _load_joblib_or_json(model_path)
    classes = [str(c) for c in art["classes"]]
    feat_names = [str(n) for n in art.get("feature_names", feature_names())]
    coef = np.asarray(art["coef"], dtype=np.float32)
    intercept = np.asarray(art["intercept"], dtype=np.float32)
    out: list[np.ndarray] = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    reorder = [class_to_idx.get(lbl, 0) for lbl in _LABELS]
    for row in rows:
        x = np.asarray(
            [float(row.ctx_features.get(name, 0.0)) for name in feat_names], dtype=np.float32
        )
        logits = (coef @ x) + intercept
        probs = _softmax(logits)
        out.append(np.asarray([probs[i] for i in reorder], dtype=np.float32))
    return out


def _encoder_probs(
    rows: list[VerifierDatasetRow],
    model_path: Path | None,
    tokenizer_path: Path | None,
) -> list[np.ndarray]:
    if model_path is None or tokenizer_path is None:
        raise ValueError("encoder model + tokenizer path required")
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "encoder evaluation requires onnxruntime + transformers installed"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out: list[np.ndarray] = []
    for row in rows:
        enc = tokenizer(
            row.candidate,
            row.ctx_text,
            truncation=True,
            max_length=384,
            padding="max_length",
            return_tensors="np",
        )
        ort_inputs = {
            key: value
            for key, value in enc.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        logits = np.asarray(session.run(None, ort_inputs)[0], dtype=np.float32)[0]
        out.append(_softmax(logits))
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    exp = np.exp(x)
    den = np.sum(exp)
    if den <= 0:
        return np.zeros_like(exp)
    return exp / den


def _config_to_json(cfg: VerifierCalibrationConfig) -> dict[str, Any]:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _load_joblib_or_json(path: Path) -> dict[str, Any]:
    try:
        import joblib

        data = joblib.load(path)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("linear artifact must be a JSON dict-like object")
    return raw
