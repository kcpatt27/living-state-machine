"""Experiment runner.

Loads a YAML config, builds the task, builds the system, runs one or more
seeds, and writes results to ``experiments/<run_id>/``.

Config skeleton::

    task:
      name: recall_pref_v1
      num_items: 20
      distractor_turns: 30
    system:
      kind: lsm              # one of: raw_llm, naive_rag, lsm
      backend: ollama
      model: qwen2.5:14b-instruct-q4_K_M
      top_k: 4
    ablation:
      state_enabled: true
      memory_enabled: true
    run:
      seeds: [0, 1, 2]
      max_context_turns: 0   # raw_llm only; 0 = all turns

Each run produces::

    experiments/<run_id>/
      config.yaml           (copy of the input)
      metrics.json          (aggregate + per-seed)
      raw_outputs.jsonl     (every (item_id, answer, correct?) row)
      log.txt
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from lsm.config import get_settings
from lsm.core.association import (
    AssociationConfig,
    HebbianAssociation,
    NullAssociation,
)
from lsm.core.embeddings import FastEmbedder, HashEmbedder
from lsm.core.habituation import ExpDecayHabituation, HabituationConfig, NullHabituation
from lsm.core.multi_timescale import (
    MultiTimescaleConfig,
    MultiTimescaleMemory,
    TierConfig,
)
from lsm.core.writer import (
    AlwaysWriter,
    LearnedWriter,
    LearnedWriterConfig,
    MemoryWriter,
    NeverWriter,
    RuleBasedWriter,
    RuleBasedWriterConfig,
)
from lsm.core.memory import InMemoryStore, LanceDBStore, MemoryStore
from lsm.core.state import LLMSummarizerUpdater, StateStore
from lsm.eval.baselines.lsm_system import LSMSystem, LSMSystemConfig
from lsm.eval.baselines.naive_rag import NaiveRAGBaseline
from lsm.eval.baselines.raw_llm import RawLLMBaseline
from lsm.eval.metrics import accuracy, any_keyword_match, mean_std
from lsm.eval.tasks.recall_pref_v1 import (
    RecallConfig,
    RecallItem,
    TASK_NAME as RECALL_PREF_V1,
    generate as generate_recall_pref_v1,
)
from lsm.runtime.llm_backend import build_backend

log = logging.getLogger(__name__)


class System(Protocol):
    name: str

    async def answer(self, item: RecallItem) -> str: ...


TaskLoader = Callable[[dict[str, Any], int], list[RecallItem]]


def _load_task(task_cfg: dict[str, Any], seed: int) -> list[RecallItem]:
    name = task_cfg.get("name")
    if name == RECALL_PREF_V1:
        return generate_recall_pref_v1(
            RecallConfig(
                num_items=int(task_cfg.get("num_items", 20)),
                distractor_turns=int(task_cfg.get("distractor_turns", 30)),
                seed=seed,
            )
        )
    raise ValueError(f"unknown task: {name!r}")


def _build_embedder(embedder_cfg: dict[str, Any] | None) -> FastEmbedder | HashEmbedder:
    cfg = embedder_cfg or {}
    kind = cfg.get("kind", "fastembed")
    if kind == "fastembed":
        return FastEmbedder(model_name=cfg.get("model", "BAAI/bge-small-en-v1.5"))
    if kind == "hash":
        return HashEmbedder(dim=int(cfg.get("dim", 384)))
    raise ValueError(f"unknown embedder kind: {kind!r}")


def _build_writer(cfg: dict[str, Any], enabled: bool) -> MemoryWriter:
    if not enabled:
        return AlwaysWriter()
    kind = str(cfg.get("kind", "rules")).lower()
    if kind == "always":
        return AlwaysWriter()
    if kind == "never":
        return NeverWriter()
    if kind == "rules":
        return RuleBasedWriter(
            RuleBasedWriterConfig(
                min_tokens=int(cfg.get("min_tokens", 4)),
                min_chars=int(cfg.get("min_chars", 12)),
                min_fraction_alpha=float(cfg.get("min_fraction_alpha", 0.5)),
                min_novelty=float(cfg.get("min_novelty", 0.15)),
                assistant_requires_durable_hit=bool(
                    cfg.get("assistant_requires_durable_hit", True)
                ),
            )
        )
    if kind == "learned":
        writer = LearnedWriter(LearnedWriterConfig(threshold=float(cfg.get("threshold", 0.5))))
        model_path = cfg.get("model_path")
        if model_path:
            writer.load(Path(str(model_path)))
        return writer
    raise ValueError(f"unknown writer kind: {kind!r}")


def _tier_from_cfg(cfg: dict[str, Any], defaults: TierConfig) -> TierConfig:
    return TierConfig(
        tau_seconds=float(cfg.get("tau_seconds", defaults.tau_seconds)),
        promote_hits=int(cfg.get("promote_hits", defaults.promote_hits)),
        vitality_floor=float(cfg.get("vitality_floor", defaults.vitality_floor)),
        max_entries=int(cfg.get("max_entries", defaults.max_entries)),
        score_weight=float(cfg.get("score_weight", defaults.score_weight)),
    )


def _build_memory(
    memory_cfg: dict[str, Any] | None, embedder: HashEmbedder | FastEmbedder, run_dir: Path
) -> MemoryStore:
    cfg = memory_cfg or {}
    kind = cfg.get("kind", "memory")
    if kind == "memory":
        return InMemoryStore(embedder=embedder)
    if kind == "lancedb":
        return LanceDBStore(root=run_dir / "lance", embedder=embedder)
    if kind == "multi":
        defaults = MultiTimescaleConfig()
        tiers_cfg = cfg.get("tiers", {}) or {}
        mtm_cfg = MultiTimescaleConfig(
            fast=_tier_from_cfg(tiers_cfg.get("fast", {}) or {}, defaults.fast),
            session=_tier_from_cfg(tiers_cfg.get("session", {}) or {}, defaults.session),
            durable=_tier_from_cfg(tiers_cfg.get("durable", {}) or {}, defaults.durable),
        )
        return MultiTimescaleMemory(embedder=embedder, config=mtm_cfg)
    raise ValueError(f"unknown memory kind: {kind!r}")


def _build_system(
    system_cfg: dict[str, Any],
    ablation_cfg: dict[str, Any],
    embedder: HashEmbedder | FastEmbedder,
    run_dir: Path,
    max_context_turns: int,
) -> System:
    kind = system_cfg["kind"]
    backend = build_backend(
        system_cfg.get("backend", "ollama"),
        model=system_cfg.get("model", get_settings().model),
        ollama_host=system_cfg.get("ollama_host", get_settings().ollama_host),
    )
    if kind == "raw_llm":
        return RawLLMBaseline(backend=backend, max_context_turns=max_context_turns)
    if kind == "naive_rag":
        return NaiveRAGBaseline(backend=backend, embedder=embedder, top_k=int(system_cfg.get("top_k", 5)))
    if kind == "lsm":
        memory = _build_memory(system_cfg.get("memory"), embedder, run_dir)
        state_store = StateStore(run_dir / "state")
        updater = LLMSummarizerUpdater(backend=backend, embedder=embedder)
        hab_cfg = system_cfg.get("habituation", {}) or {}
        habituation = (
            ExpDecayHabituation(
                HabituationConfig(
                    tau_seconds=float(hab_cfg.get("tau_seconds", 3600.0)),
                    eta=float(hab_cfg.get("eta", 1.0)),
                    damping_strength=float(hab_cfg.get("damping_strength", 0.35)),
                )
            )
            if bool(ablation_cfg.get("habituation_enabled", True))
            else NullHabituation()
        )
        writer = _build_writer(
            system_cfg.get("writer", {}) or {},
            bool(ablation_cfg.get("writer_enabled", True)),
        )
        assoc_cfg = system_cfg.get("association", {}) or {}
        association = (
            HebbianAssociation(
                AssociationConfig(
                    eta=float(assoc_cfg.get("eta", 0.25)),
                    lambda_per_second=float(assoc_cfg.get("lambda_per_second", 1e-5)),
                    boost_strength=float(assoc_cfg.get("boost_strength", 0.35)),
                )
            )
            if bool(ablation_cfg.get("association_enabled", True))
            else NullAssociation()
        )
        return LSMSystem(
            backend=backend,
            embedder=embedder,
            memory=memory,
            state_store=state_store,
            state_updater=updater,
            config=LSMSystemConfig(
                top_k=int(system_cfg.get("top_k", 4)),
                min_score=float(system_cfg.get("min_score", 0.15)),
                state_enabled=bool(ablation_cfg.get("state_enabled", True)),
                memory_enabled=bool(ablation_cfg.get("memory_enabled", True)),
                habituation_enabled=bool(ablation_cfg.get("habituation_enabled", True)),
                association_enabled=bool(ablation_cfg.get("association_enabled", True)),
                writer_enabled=bool(ablation_cfg.get("writer_enabled", True)),
                recent_context_window=int(system_cfg.get("recent_context_window", 8)),
            ),
            habituation=habituation,
            association=association,
            writer=writer,
        )
    raise ValueError(f"unknown system kind: {kind!r}")


def _grade(item: RecallItem, answer: str) -> bool:
    return any_keyword_match(answer, item.expected_contains())


@dataclass(slots=True)
class SeedResult:
    seed: int
    accuracy: float
    n: int
    elapsed_s: float
    per_item: list[dict[str, object]] = field(default_factory=list)


async def _run_seed(
    system_cfg: dict[str, Any],
    ablation_cfg: dict[str, Any],
    task_cfg: dict[str, Any],
    run_dir: Path,
    seed: int,
    max_context_turns: int,
    embedder_cfg: dict[str, Any] | None,
) -> SeedResult:
    embedder = _build_embedder(embedder_cfg)
    items = _load_task(task_cfg, seed)
    system = _build_system(
        system_cfg=system_cfg,
        ablation_cfg=ablation_cfg,
        embedder=embedder,
        run_dir=run_dir / f"seed_{seed}",
        max_context_turns=max_context_turns,
    )
    (run_dir / f"seed_{seed}").mkdir(parents=True, exist_ok=True)

    results: list[bool] = []
    per_item: list[dict[str, object]] = []
    t0 = time.time()
    for item in items:
        try:
            answer = await system.answer(item)
        except Exception as e:  # noqa: BLE001
            log.exception("item %s failed: %s", item.item_id, e)
            answer = ""
        correct = _grade(item, answer)
        results.append(correct)
        per_item.append(
            {
                "item_id": item.item_id,
                "topic": item.topic,
                "correct_value": item.correct,
                "answer": answer,
                "passed": correct,
            }
        )
    elapsed = time.time() - t0

    return SeedResult(
        seed=seed,
        accuracy=accuracy(results),
        n=len(results),
        elapsed_s=elapsed,
        per_item=per_item,
    )


async def run_from_config(
    config_path: Path, out_root: Path | None = None, run_id: str | None = None
) -> Path:
    """Run an experiment described by a YAML config and return its output dir."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_root = out_root or (Path.cwd() / "experiments")
    rid = run_id or time.strftime("%Y%m%d-%H%M%S") + f"-{cfg.get('name', config_path.stem)}"
    run_dir = out_root / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    task_cfg = cfg.get("task", {"name": RECALL_PREF_V1})
    system_cfg = cfg.get("system", {"kind": "raw_llm"})
    ablation_cfg = cfg.get("ablation", {})
    run_cfg = cfg.get("run", {})
    embedder_cfg = cfg.get("embedder")
    max_context_turns = int(run_cfg.get("max_context_turns", 0))

    seeds = [int(s) for s in run_cfg.get("seeds", [0, 1, 2])]
    seed_results: list[SeedResult] = []
    raw_path = run_dir / "raw_outputs.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_fp:
        for seed in seeds:
            log.info("seed %d starting ...", seed)
            res = await _run_seed(
                system_cfg=system_cfg,
                ablation_cfg=ablation_cfg,
                task_cfg=task_cfg,
                run_dir=run_dir,
                seed=seed,
                max_context_turns=max_context_turns,
                embedder_cfg=embedder_cfg,
            )
            for row in res.per_item:
                raw_fp.write(json.dumps({"seed": seed, **row}) + "\n")
            seed_results.append(res)
            log.info("seed %d accuracy=%.3f n=%d elapsed=%.1fs", seed, res.accuracy, res.n, res.elapsed_s)

    accs = [r.accuracy for r in seed_results]
    mean, std = mean_std(accs)
    metrics = {
        "task": task_cfg.get("name"),
        "system": system_cfg.get("kind"),
        "model": system_cfg.get("model"),
        "seeds": seeds,
        "accuracy_mean": mean,
        "accuracy_std": std,
        "per_seed": [asdict(r) for r in seed_results],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("done. accuracy mean=%.3f std=%.3f -> %s", mean, std, run_dir)
    return run_dir


def run_from_config_sync(*args: Any, **kwargs: Any) -> Path:
    return asyncio.run(run_from_config(*args, **kwargs))
