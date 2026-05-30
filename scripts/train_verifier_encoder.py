"""Train a small encoder verifier and export ONNX artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lsm.verifier_data.schema import load_rows

LABEL_TO_ID = {"accept": 0, "reprompt": 1, "reject": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Input verifier dataset JSONL.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF model checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/verifier_encoder_v1"),
        help="Output directory for trained checkpoint and ONNX files.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--export-onnx", action="store_true", default=False)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "train_verifier_encoder.py requires transformers + datasets + torch. "
            "Install with: uv sync --extra research"
        ) from exc

    rows = load_rows(args.dataset)
    if not rows:
        raise RuntimeError(f"dataset is empty: {args.dataset}")
    texts_a = [r.candidate for r in rows]
    texts_b = [r.ctx_text for r in rows]
    labels = [LABEL_TO_ID.get(r.label, 0) for r in rows]
    ds = Dataset.from_dict({"text_a": texts_a, "text_b": texts_b, "label": labels})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tok(batch: dict[str, list[str]]) -> dict[str, np.ndarray]:
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    ds = ds.map(tok, batched=True)
    if len(rows) >= 50:
        split = ds.train_test_split(test_size=0.1, seed=0)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = ds
        eval_ds = ds

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.out_dir / "checkpoints"),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch" if len(rows) >= 50 else "no",
        logging_steps=20,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        processing_class=tokenizer,
    )
    trainer.train()
    eval_metrics = trainer.evaluate() if len(eval_ds) > 0 else {}

    model_dir = args.out_dir / "hf"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(args.out_dir / "tokenizer")

    metrics = {"n_rows": len(rows), **{k: float(v) for k, v in eval_metrics.items()}}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.export_onnx:
        _export_onnx(model, tokenizer, args.max_length, args.out_dir / "model.onnx")

    print(json.dumps(metrics, indent=2))


def _export_onnx(model, tokenizer, max_length: int, out_path: Path) -> None:
    import torch

    model.eval()
    sample = tokenizer(
        "candidate",
        "context",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_names = ["input_ids", "attention_mask"]
    inputs = [sample["input_ids"], sample["attention_mask"]]
    if "token_type_ids" in sample:
        input_names.append("token_type_ids")
        inputs.append(sample["token_type_ids"])
    dynamic_axes = {name: {0: "batch", 1: "seq"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        tuple(inputs),
        str(out_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )


if __name__ == "__main__":
    main()
