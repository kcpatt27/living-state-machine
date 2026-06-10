"""Transformers backend + LoRA adapter wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from lsm.runtime.llm_backend import TransformersBackend, build_backend

ADAPTER_DIR = Path("runs/runpod_l40s_burst/qlora_qwen_7b/seed_0/step_002000")
has_torch = importlib.util.find_spec("torch") is not None
has_adapter = (ADAPTER_DIR / "adapter_model.safetensors").exists()


def test_build_backend_passes_adapter_path() -> None:
    backend = build_backend(
        "transformers",
        "Qwen/Qwen2.5-7B-Instruct",
        adapter_path="runs/runpod_l40s_burst/qlora_qwen_7b/seed_0/step_002000",
        device="cpu",
    )
    assert isinstance(backend, TransformersBackend)
    assert backend._adapter_path == "runs/runpod_l40s_burst/qlora_qwen_7b/seed_0/step_002000"


@pytest.mark.skipif(not has_torch, reason="torch not installed")
@pytest.mark.skipif(not has_adapter, reason="LoRA adapter weights missing")
def test_lazy_load_applies_peft_adapter() -> None:
    backend = TransformersBackend(
        model="Qwen/Qwen2.5-7B-Instruct",
        device="cpu",
        load_in_4bit=False,
        adapter_path=str(ADAPTER_DIR),
    )
    backend._lazy_load()
    assert backend._model is not None
    # peft wraps the base model
    assert (
        hasattr(backend._model, "peft_config") or backend._model.__class__.__name__ == "PeftModel"
    )
