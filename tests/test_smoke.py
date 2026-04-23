"""Basic smoke tests confirming the package imports and settings load."""

from __future__ import annotations


def test_import_lsm() -> None:
    import lsm

    assert lsm.__version__


def test_settings_defaults() -> None:
    from lsm.config import get_settings

    s = get_settings()
    assert s.state_dim == 512
    assert s.port == 8765
    assert s.backend in {"ollama", "transformers"}
