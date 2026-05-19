"""Tests for assistant identity profile persistence."""

from __future__ import annotations

from pathlib import Path

from lsm.core.identity import DEFAULT_DESCRIPTION, DEFAULT_NAME, IdentityProfile, IdentityStore


def test_identity_store_roundtrip(tmp_path: Path) -> None:
    store = IdentityStore(tmp_path)
    profile = IdentityProfile(name="Elsum", description="Test assistant.")
    store.save(profile)
    loaded = store.load()
    assert loaded.name == "Elsum"
    assert loaded.description == "Test assistant."


def test_identity_store_writes_defaults_when_missing(tmp_path: Path) -> None:
    store = IdentityStore(tmp_path)
    assert not store.path.exists()
    loaded = store.load()
    assert loaded.name == DEFAULT_NAME
    assert loaded.description == DEFAULT_DESCRIPTION
    assert store.path.exists()
    again = store.load()
    assert again.name == DEFAULT_NAME


def test_identity_role_line() -> None:
    p = IdentityProfile(name="Elsum", description="Helper.")
    assert p.identity_role_line() == "identity/role: Elsum - Helper."
