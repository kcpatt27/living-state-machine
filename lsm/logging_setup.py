"""Structured logging setup.

We use :class:`rich.logging.RichHandler` for local dev (pretty console output)
and plain JSON-ish records when run under a non-TTY (for log shipping later).
"""

from __future__ import annotations

import logging
import os
import sys

from rich.logging import RichHandler


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    level_name = (level or os.environ.get("LSM_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if getattr(root, "_lsm_configured", False):
        root.setLevel(level_name)
        return

    handler: logging.Handler
    if sys.stderr.isatty():
        handler = RichHandler(
            rich_tracebacks=True,
            markup=False,
            show_time=True,
            show_path=False,
        )
        fmt = "%(name)s | %(message)s"
    else:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"

    logging.basicConfig(level=level_name, handlers=[handler], format=fmt, force=True)
    root._lsm_configured = True  # type: ignore[attr-defined]
