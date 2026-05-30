"""JSONL schema for verifier training/eval rows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

LabelKind = Literal["accept", "reprompt", "reject"]


@dataclass(slots=True)
class VerifierDatasetRow:
    """One labeled verifier training/eval sample."""

    event_id: str
    source: str
    candidate: str
    ctx_text: str
    ctx_features: dict[str, float]
    rule_energy: float
    rule_decision: LabelKind
    rule_violations: list[str]
    label_source: str
    label: LabelKind
    reprompt_used: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_rows(path: Path, rows: list[VerifierDatasetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(row.to_json() + "\n")


def load_rows(path: Path) -> list[VerifierDatasetRow]:
    rows: list[VerifierDatasetRow] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        data = json.loads(raw)
        rows.append(
            VerifierDatasetRow(
                event_id=str(data.get("event_id", "")),
                source=str(data.get("source", "manual")),
                candidate=str(data.get("candidate", "")),
                ctx_text=str(data.get("ctx_text", "")),
                ctx_features={
                    str(k): float(v) for k, v in (data.get("ctx_features", {}) or {}).items()
                },
                rule_energy=float(data.get("rule_energy", 0.0)),
                rule_decision=str(data.get("rule_decision", "accept")),  # type: ignore[arg-type]
                rule_violations=[str(x) for x in data.get("rule_violations", [])],
                label_source=str(data.get("label_source", "unknown")),
                label=str(data.get("label", data.get("rule_decision", "accept"))),  # type: ignore[arg-type]
                reprompt_used=bool(data.get("reprompt_used", False)),
                meta=dict(data.get("meta", {}) or {}),
            )
        )
    return rows
