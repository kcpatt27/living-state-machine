"""QLoRA fine-tune wrapper — L15.

A thin, opinionated wrapper over ``peft`` + ``bitsandbytes`` for
4-bit-QLoRA fine-tuning of the wrapped 7B Qwen model on the 5060 Ti.
The Phase 1 baseline (frozen Qwen2.5-14B Q4_K_M via Ollama) is fast to
serve but cannot be nudged on user data; L15 trains adapters on top of
the Qwen2.5-7B-Instruct base via Transformers.

Scope of this file
==================

- :class:`QLoRAConfig` — every tunable in one place.
- :func:`load_base_model` — loads the 4-bit base via
  ``AutoModelForCausalLM`` with ``bitsandbytes`` quantisation.
- :func:`prepare_lora_model` — wraps a base model with LoRA adapters
  targeting the default Qwen linear modules.
- :func:`train_qlora` — TRL-free, plain-PyTorch SFT loop over a
  pre-tokenised dataset. Sample tokeniser built by L11's tokenizer
  pipeline.

This file is dependency-gated: instantiating anything runtime pulls in
``torch``, ``transformers``, ``bitsandbytes``, ``peft``. The skeleton is
readable without those packages so it can live in the repo without
blocking Phase 1.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lsm.research.small_core.minimal_ssm import _require_torch

log = logging.getLogger(__name__)


_DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(slots=True)
class QLoRAConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path: Path = Path("data/sft.jsonl")

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_TARGETS)

    seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 16
    steps: int = 2_000

    lr: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    bnb_quant_type: str = "nf4"
    bnb_compute_dtype: str = "bfloat16"
    double_quant: bool = True

    device: str = "cuda"
    seed: int = 0
    output_dir: Path = Path("runs/qlora_qwen_7b")
    save_every: int = 500


def load_base_model(cfg: QLoRAConfig) -> tuple[Any, Any]:
    torch = _require_torch()
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_compute_dtype),
        bnb_4bit_use_double_quant=cfg.double_quant,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb,
        device_map={"": cfg.device},
    )
    return model, tokenizer


def prepare_lora_model(base_model: Any, cfg: QLoRAConfig) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base_model = prepare_model_for_kbit_training(base_model)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    return model


def jsonl_dataset(path: Path) -> Iterator[str]:
    """Yield plain-text rows from a JSONL file (``text`` key)."""
    import json

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            yield rec["text"] if "text" in rec else rec.get("prompt", "")


def _pack_batches(
    rows: Iterator[str], tokenizer: Any, cfg: QLoRAConfig
) -> Iterator[Any]:
    torch = _require_torch()
    buf: list[int] = []
    for text in rows:
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids + [tokenizer.eos_token_id])
        while len(buf) >= cfg.batch_size * (cfg.seq_len + 1):
            batch = buf[: cfg.batch_size * (cfg.seq_len + 1)]
            buf = buf[cfg.batch_size * (cfg.seq_len + 1) :]
            t = torch.tensor(batch, dtype=torch.long).reshape(
                cfg.batch_size, cfg.seq_len + 1
            )
            yield t


def train_qlora(cfg: QLoRAConfig) -> dict[str, Any]:
    torch = _require_torch()
    torch.manual_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "config.json").write_text(
        str({k: str(v) for k, v in asdict(cfg).items()}), encoding="utf-8"
    )

    base, tokenizer = load_base_model(cfg)
    model = prepare_lora_model(base, cfg)
    model.train()

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    rows = jsonl_dataset(cfg.dataset_path)
    batch_iter = _pack_batches(rows, tokenizer, cfg)

    t0 = time.time()
    running = 0.0
    steps_seen = 0
    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            batch = next(batch_iter)
            batch = batch.to(cfg.device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            logits = model(x).logits
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ) / cfg.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        running += float(loss.detach().item()) * cfg.grad_accum
        steps_seen += 1
        if step % 10 == 0:
            log.info(
                "qlora step=%d loss=%.4f elapsed=%ds",
                step,
                running / max(steps_seen, 1),
                int(time.time() - t0),
            )
            running = 0.0
            steps_seen = 0
        if step % cfg.save_every == 0 or step == cfg.steps:
            model.save_pretrained(cfg.output_dir / f"step_{step:06d}")

    return {"steps": cfg.steps, "output_dir": str(cfg.output_dir)}


__all__ = [
    "QLoRAConfig",
    "load_base_model",
    "prepare_lora_model",
    "train_qlora",
]
