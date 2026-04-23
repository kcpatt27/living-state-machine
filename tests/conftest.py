"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_lsm_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate per-test state and memory directories."""
    monkeypatch.setenv("LSM_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("LSM_MEMORY_DIR", str(tmp_path / "memory_store"))
    monkeypatch.setenv("LSM_ENCRYPT", "false")
    monkeypatch.setenv("LSM_LOG_LEVEL", "WARNING")

    from lsm.config import reset_settings

    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def clean_env() -> None:
    """Drop any pre-existing LSM_* env vars for strict tests."""
    for k in list(os.environ):
        if k.startswith("LSM_"):
            del os.environ[k]
