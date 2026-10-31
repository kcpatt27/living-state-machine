"""Smoke tests for ``lsm fact prune`` (F.1.1 cleanup escape hatch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lsm.cli.cli import app
from lsm.core.facts import (
    CAT_USER_PREFERENCES,
    CAT_USER_TRAITS,
    Fact,
    FactStore,
)
from lsm.core.identity import IdentityProfile
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LSM_HOME", str(tmp_path))
    return tmp_path


def _seed(home: Path) -> FactStore:
    store = FactStore(home)
    store.sync_assistant_identity(IdentityProfile(name="Elsum", description="Helper."))
    store.upsert(Fact(owner="user", category=CAT_USER_TRAITS, value="Kalen"))
    store.upsert(Fact(owner="user", category=CAT_USER_TRAITS, value="ctually Kalen"))
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_PREFERENCES,
            subject="general",
            value="a long blob preference value",
        )
    )
    return store


def test_prune_by_category(home: Path) -> None:
    _seed(home)
    result = runner.invoke(app, ["fact", "prune", "--category", CAT_USER_TRAITS, "--yes"])
    assert result.exit_code == 0, result.output
    store2 = FactStore(home)
    assert store2.query(owner="user", category=CAT_USER_TRAITS) == []
    # Other categories untouched.
    assert store2.query(owner="user", category=CAT_USER_PREFERENCES)


def test_prune_by_value_contains(home: Path) -> None:
    _seed(home)
    result = runner.invoke(app, ["fact", "prune", "--value-contains", "blob preference", "--yes"])
    assert result.exit_code == 0, result.output
    store2 = FactStore(home)
    assert store2.query(owner="user", category=CAT_USER_PREFERENCES) == []


def test_prune_all_preserves_assistant_identity(home: Path) -> None:
    _seed(home)
    result = runner.invoke(app, ["fact", "prune", "--all", "--yes"])
    assert result.exit_code == 0, result.output
    store2 = FactStore(home)
    remaining = store2.all_facts()
    assert len(remaining) == 1
    assert remaining[0].owner == "assistant"


def test_prune_requires_a_filter(home: Path) -> None:
    _seed(home)
    result = runner.invoke(app, ["fact", "prune"])
    assert result.exit_code == 2


def test_prune_writes_facts_json_atomically(home: Path) -> None:
    _seed(home)
    runner.invoke(app, ["fact", "prune", "--category", CAT_USER_TRAITS, "--yes"])
    data = json.loads((home / "facts.json").read_text(encoding="utf-8"))
    cats = {row["category"] for row in data["facts"]}
    assert CAT_USER_TRAITS not in cats
