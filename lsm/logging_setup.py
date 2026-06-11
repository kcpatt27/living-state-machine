"""Structured logging setup.

We use :class:`rich.logging.RichHandler` for local dev (pretty console output)
and plain JSON-ish records when run under a non-TTY (for log shipping later).

On Windows PowerShell (dark blue background), Rich's default timestamp color is
nearly invisible — we use a high-contrast theme instead. Set ``LSM_LOG_PLAIN=1``
to force plain text logs.
"""

from __future__ import annotations

import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Readable on dark blue PowerShell + typical dark terminals (sepia timestamps).
_LSM_LOG_THEME = Theme(
    {
        "log.time": "bold #d4a870",
        "logging.level.debug": "cyan",
        "logging.level.info": "bold green",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "repr.number": "bold cyan",
        "repr.url": "underline cyan",
    }
)


def _plain_logs_requested() -> bool:
    flag = os.environ.get("LSM_LOG_PLAIN", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    level_name = (level or os.environ.get("LSM_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if getattr(root, "_lsm_configured", False):
        root.setLevel(level_name)
        return

    handler: logging.Handler
    if sys.stderr.isatty() and not _plain_logs_requested():
        console = Console(theme=_LSM_LOG_THEME, stderr=True, soft_wrap=True)
        handler = RichHandler(
            console=console,
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
