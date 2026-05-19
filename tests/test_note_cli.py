"""Smoke tests for ``lsm note`` CLI (F.2 review fixes).

Adds an explicit producer for ``CAT_ASSISTANT_NOTES`` so the slot rendered
by :class:`DeterministicProjection` has a non-test data path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lsm.cli.cli import app
from lsm.core.facts import CAT_ASSISTANT_NOTES, FactStore
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LSM_HOME", str(tmp_path))
    return tmp_path


def test_note_add_writes_fact(home: Path) -> None:
    result = runner.invoke(app, ["note", "add", "keep answers concise"])
    assert result.exit_code == 0, result.output
    store = FactStore(home)
    notes = store.query(owner="assistant", category=CAT_ASSISTANT_NOTES)
    assert len(notes) == 1
    assert notes[0].value == "keep answers concise"
    assert notes[0].subject is None


def test_note_add_with_subject(home: Path) -> None:
    result = runner.invoke(app, ["note", "add", "user prefers terse replies", "--subject", "style"])
    assert result.exit_code == 0, result.output
    notes = FactStore(home).query(owner="assistant", category=CAT_ASSISTANT_NOTES)
    assert len(notes) == 1
    assert notes[0].subject == "style"


def test_note_add_rejects_empty(home: Path) -> None:
    result = runner.invoke(app, ["note", "add", "   "])
    assert result.exit_code == 2


def test_note_list_and_clear(home: Path) -> None:
    runner.invoke(app, ["note", "add", "first note"])
    runner.invoke(app, ["note", "add", "second note"])
    listed = runner.invoke(app, ["note", "list"])
    assert listed.exit_code == 0
    assert "first note" in listed.output
    assert "second note" in listed.output

    cleared = runner.invoke(app, ["note", "clear", "--yes"])
    assert cleared.exit_code == 0
    notes = FactStore(home).query(owner="assistant", category=CAT_ASSISTANT_NOTES)
    assert notes == []
