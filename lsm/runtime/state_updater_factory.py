"""Build :class:`StateUpdater` instances for eval and production pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lsm.core.embeddings import Embedder
from lsm.core.facts import FactStore
from lsm.core.identity import IdentityProfile
from lsm.core.memory import MemoryStore
from lsm.core.projection import DeterministicProjection, FactProjection
from lsm.core.state import FactsBackedUpdater, LLMSummarizerUpdater, StateUpdater
from lsm.core.verifier import Verifier
from lsm.core.verifier_log import VerifierLog
from lsm.runtime.llm_backend import LLMBackend

DEFAULT_HYBRID_CKPT = Path("runs/small/minimal/ckpts/step_002500.pt")


def build_state_updater(
    kind: str,
    *,
    backend: LLMBackend,
    embedder: Embedder,
    memory: MemoryStore | None = None,
    fact_store: FactStore | None = None,
    identity: IdentityProfile | None = None,
    projection: FactProjection | None = None,
    verifier: Verifier | None = None,
    verifier_reprompt_enabled: bool = True,
    verifier_log: VerifierLog | None = None,
    hybrid_cfg: dict[str, Any] | None = None,
) -> StateUpdater:
    """Return a state updater for ``kind`` (facts_backed / llm_summarizer / hybrid)."""
    normalized = kind.lower()
    if normalized == "llm_summarizer":
        return LLMSummarizerUpdater(
            backend,
            embedder,
            identity=identity,
            memory=memory,
            fact_store=fact_store,
            verifier=verifier,
            verifier_reprompt_enabled=verifier_reprompt_enabled,
            verifier_log=verifier_log,
        )
    if normalized == "hybrid":
        from transformers import GPT2Tokenizer

        from lsm.research.hybrid import HybridConfig, hybrid_from_checkpoint

        cfg = hybrid_cfg or {}
        ckpt_path = Path(cfg.get("ckpt_path", DEFAULT_HYBRID_CKPT))
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = HybridConfig(
            mode=str(cfg.get("mode", "topk_tokens")),
            summary_tokens=int(cfg.get("summary_tokens", 48)),
            top_k_neighbours=int(cfg.get("top_k_neighbours", 4)),
            device=str(cfg.get("device", "cuda")),
        )
        return hybrid_from_checkpoint(
            ckpt_path,
            tokenizer=tokenizer,
            config=config,
            memory=memory,
            embedder=embedder,
        )
    proj = projection if projection is not None else DeterministicProjection()
    return FactsBackedUpdater(
        embedder,
        proj,
        identity=identity,
        memory=memory,
        fact_store=fact_store,
    )


__all__ = ["DEFAULT_HYBRID_CKPT", "build_state_updater"]
