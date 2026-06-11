"""Tests for lsm.logging_setup."""

from __future__ import annotations

import logging

from lsm.logging_setup import setup_logging


def test_setup_logging_plain_mode(monkeypatch) -> None:
    monkeypatch.setenv("LSM_LOG_PLAIN", "1")
    root = logging.getLogger()
    if hasattr(root, "_lsm_configured"):
        delattr(root, "_lsm_configured")
    for h in list(root.handlers):
        root.removeHandler(h)

    setup_logging("INFO")

    assert len(root.handlers) == 1
    assert type(root.handlers[0]).__name__ == "StreamHandler"
