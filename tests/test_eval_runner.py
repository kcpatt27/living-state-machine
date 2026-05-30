"""End-to-end eval runner test with a stub backend so no Ollama is required."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import yaml
from lsm.eval import runner
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions
from lsm.verifier_data.schema import VerifierDatasetRow, write_rows

_PATTERN = re.compile(r"my [^.\n?]*? is ([^.\n]+)", re.IGNORECASE)


class _ScriptedBackend:
    """Always answers with the correct value from the dialogue history.

    We use this to assert the runner wiring is working without needing a real
    LLM. We scan all messages for the first 'my ... is ...' pattern.
    """

    name = "scripted"

    def __init__(self, model: str = "scripted") -> None:
        self.model = model

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        joined = "\n".join(m.content for m in messages)
        match = _PATTERN.search(joined)
        if match:
            value = match.group(1).strip()
            return f"It's {value}."
        return "I don't know"

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        yield await self.generate(messages, options)

    async def close(self) -> None:
        return None


def test_runner_runs_raw_llm_baseline(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "build_backend", lambda *a, **k: _ScriptedBackend())

    cfg = {
        "name": "smoke",
        "task": {"name": "recall_pref_v1", "num_items": 3, "distractor_turns": 4},
        "system": {"kind": "raw_llm", "backend": "ollama", "model": "scripted"},
        "embedder": {"kind": "hash", "dim": 32},
        "run": {"seeds": [0, 1]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_root = tmp_path / "experiments"
    run_dir = asyncio.run(runner.run_from_config(cfg_path, out_root=out_root, run_id="t1"))
    assert run_dir.exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    # The scripted backend echoes the preference value; accuracy should be 1.
    assert metrics["accuracy_mean"] == 1.0
    assert len(metrics["per_seed"]) == 2
    raw = [json.loads(line) for line in (run_dir / "raw_outputs.jsonl").read_text().splitlines()]
    assert all(r["passed"] for r in raw)


def test_runner_runs_naive_rag_baseline(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "build_backend", lambda *a, **k: _ScriptedBackend())
    cfg = {
        "name": "smoke_rag",
        "task": {"name": "recall_pref_v1", "num_items": 2, "distractor_turns": 3},
        "system": {"kind": "naive_rag", "backend": "ollama", "model": "scripted", "top_k": 5},
        "embedder": {"kind": "hash", "dim": 32},
        "run": {"seeds": [0]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_dir = asyncio.run(runner.run_from_config(cfg_path, out_root=tmp_path / "exp", run_id="rag"))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    # RAG-with-a-helpful-backend should also pass deterministically.
    assert metrics["accuracy_mean"] == 1.0


def test_runner_runs_verifier_calibration_task(tmp_path: Path) -> None:
    ds = tmp_path / "dataset.jsonl"
    rows = [
        VerifierDatasetRow(
            event_id="e1",
            source="reply",
            candidate="ok",
            ctx_text="",
            ctx_features={},
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
            ctx_features={},
            rule_energy=2.5,
            rule_decision="reject",
            rule_violations=["fact_contradiction"],
            label_source="manual_gold",
            label="reject",
        ),
    ]
    write_rows(ds, rows)
    cfg = {
        "name": "verifier_calib_smoke",
        "task": {"name": "verifier_calibration_v1", "dataset_path": str(ds)},
        "run": {"seeds": [0]},
    }
    cfg_path = tmp_path / "cfg_verify.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_dir = asyncio.run(
        runner.run_from_config(cfg_path, out_root=tmp_path / "exp_verify", run_id="verify")
    )
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["task"] == "verifier_calibration_v1"
    assert "rule" in metrics["variants"]
    assert "null" in metrics["variants"]
