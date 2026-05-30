"""F.3 — user-approved dynamic fact categories.

Core categories are built into :mod:`lsm.core.facts`. Users can register
additional ``user.<slug>`` categories (e.g. ``user.food`` via
``lsm fact category add food``) and store facts there manually or via
``lsm fact set``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from lsm.core.facts import (
    CAT_ASSISTANT_IDENTITY,
    CAT_ASSISTANT_NOTES,
    USER_CATEGORIES,
)

# Built-in categories always allowed (not stored in the registry file).
_BUILTIN_CATEGORIES = frozenset(
    {
        CAT_ASSISTANT_IDENTITY,
        CAT_ASSISTANT_NOTES,
        *USER_CATEGORIES,
    }
)


@dataclass(slots=True)
class FactCategoryDef:
    """A user-registered category."""

    id: str
    label: str
    owner: str = "user"
    description: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at <= 0:
            self.created_at = time.time()


class FactCategoryRegistry:
    """Persisted registry at ``$LSM_HOME/fact_categories.json``."""

    FILENAME = "fact_categories.json"

    def __init__(self, home: Path) -> None:
        self.home = home
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / self.FILENAME
        self._categories: dict[str, FactCategoryDef] = {}
        self.load()

    def load(self) -> FactCategoryRegistry:
        if not self.path.exists():
            self._categories = {}
            self.save()
            return self
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._categories = {}
        for row in data.get("categories", []):
            cat_id = str(row["id"])
            self._categories[cat_id] = FactCategoryDef(
                id=cat_id,
                label=str(row.get("label", cat_id)),
                owner=str(row.get("owner", "user")),
                description=str(row.get("description", "")),
                created_at=float(row.get("created_at", 0.0)),
            )
        return self

    def save(self) -> None:
        payload = {
            "categories": [asdict(c) for c in self._categories.values()],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))

    @staticmethod
    def normalize_slug(slug: str) -> str:
        """Turn ``food`` or ``user.food`` into canonical ``user.food``."""
        s = slug.strip().lower().replace(" ", "_")
        if not s:
            raise ValueError("category slug is empty")
        if s.startswith("user."):
            return s
        if s.startswith("assistant."):
            raise ValueError("only user.* dynamic categories are supported in F.3")
        return f"user.{s}"

    def register(
        self,
        slug: str,
        *,
        description: str = "",
    ) -> FactCategoryDef:
        cat_id = self.normalize_slug(slug)
        if cat_id in _BUILTIN_CATEGORIES:
            raise ValueError(f"{cat_id!r} is a built-in category")
        label = cat_id.split(".", 1)[-1] if "." in cat_id else cat_id
        existing = self._categories.get(cat_id)
        if existing is not None:
            if description:
                existing.description = description
            self.save()
            return existing
        defn = FactCategoryDef(id=cat_id, label=label, description=description)
        self._categories[cat_id] = defn
        self.save()
        return defn

    def remove(self, slug: str) -> bool:
        cat_id = self.normalize_slug(slug)
        if cat_id in _BUILTIN_CATEGORIES:
            raise ValueError(f"cannot remove built-in category {cat_id!r}")
        if cat_id not in self._categories:
            return False
        del self._categories[cat_id]
        self.save()
        return True

    def list_custom(self) -> list[FactCategoryDef]:
        return sorted(self._categories.values(), key=lambda c: c.id)

    def is_allowed(self, category: str) -> bool:
        return category in _BUILTIN_CATEGORIES or category in self._categories

    def all_allowed(self) -> frozenset[str]:
        return _BUILTIN_CATEGORIES | frozenset(self._categories.keys())


__all__ = [
    "FactCategoryDef",
    "FactCategoryRegistry",
]
