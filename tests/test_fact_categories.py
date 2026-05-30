"""Tests for F.3 dynamic fact categories."""

from __future__ import annotations

from pathlib import Path

import pytest
from lsm.core.fact_categories import FactCategoryRegistry
from lsm.core.facts import CAT_USER_PREFERENCES, Fact, FactStore


def test_register_and_list_custom_category(tmp_path: Path) -> None:
    reg = FactCategoryRegistry(tmp_path)
    defn = reg.register("food", description="meals and restaurants")
    assert defn.id == "user.food"
    assert len(reg.list_custom()) == 1
    assert reg.is_allowed("user.food")


def test_builtin_category_cannot_be_registered(tmp_path: Path) -> None:
    reg = FactCategoryRegistry(tmp_path)
    with pytest.raises(ValueError, match="built-in"):
        reg.register("preferences")


def test_fact_store_rejects_unknown_category(tmp_path: Path) -> None:
    reg = FactCategoryRegistry(tmp_path)
    store = FactStore(tmp_path, category_registry=reg)
    with pytest.raises(ValueError, match="unknown category"):
        store.upsert(
            Fact(owner="user", category="user.food", value="sushi"),
        )


def test_fact_store_accepts_registered_category(tmp_path: Path) -> None:
    reg = FactCategoryRegistry(tmp_path)
    reg.register("food")
    store = FactStore(tmp_path, category_registry=reg)
    fact = store.upsert(
        Fact(owner="user", category="user.food", subject="lunch", value="sushi"),
    )
    assert fact.category == "user.food"
    assert store.query(category="user.food")[0].value == "sushi"


def test_builtin_preferences_still_allowed(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    fact = store.upsert(
        Fact(owner="user", category=CAT_USER_PREFERENCES, value="dark mode"),
    )
    assert fact.category == CAT_USER_PREFERENCES
