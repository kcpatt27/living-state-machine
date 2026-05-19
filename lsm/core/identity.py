"""Assistant identity profile persisted under ``$LSM_HOME``.

The summarizer pins ``identity/role`` from this file so the LLM cannot swap
the assistant's name with the user's.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_NAME = "Elsum"
DEFAULT_DESCRIPTION = (
    "Local Stentor-inspired AI assistant. Persistent state and durable memory; "
    "learns from corrections."
)


@dataclass(slots=True)
class IdentityProfile:
    """Editable assistant persona (name + what it is / can be)."""

    name: str = DEFAULT_NAME
    description: str = DEFAULT_DESCRIPTION

    def identity_role_line(self) -> str:
        """Single ``s_t.text`` bullet for the pinned identity slot."""
        return f"identity/role: {self.name} - {self.description}"


class IdentityStore:
    """Filesystem-backed store for :class:`IdentityProfile`."""

    FILENAME = "identity.json"

    def __init__(self, home: Path) -> None:
        self.home = home
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / self.FILENAME

    def load(self) -> IdentityProfile:
        """Load profile; write defaults if the file is missing."""
        if not self.path.exists():
            profile = IdentityProfile()
            self.save(profile)
            return profile
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return IdentityProfile(
            name=str(data.get("name", DEFAULT_NAME)),
            description=str(data.get("description", DEFAULT_DESCRIPTION)),
        )

    def save(self, profile: IdentityProfile) -> None:
        """Atomically persist the profile."""
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
