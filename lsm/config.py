"""Project configuration.

All runtime configuration flows through :class:`Settings`. Values come from
environment variables (``LSM_*``), an optional ``.env`` file, or explicit
arguments. YAML experiment configs are loaded into :class:`ExperimentConfig`
at run time; they override :class:`Settings` for the duration of a single run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings."""

    model_config = SettingsConfigDict(
        env_prefix="LSM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    home: Path = Field(default=Path("./state"))
    memory_dir: Path = Field(default=Path("./memory_store"))
    models_dir: Path = Field(default=Path("./models"))

    model: str = "qwen2.5:14b-instruct-q4_K_M"
    fast_model: str = "qwen2.5:7b-instruct-q8_0"
    baseline_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_host: str = "http://localhost:11434"
    backend: Literal["ollama", "transformers"] = "ollama"

    host: str = "127.0.0.1"
    port: int = 8765

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: Literal["cpu", "cuda"] = "cpu"

    state_dim: int = 512

    encrypt: bool = True
    encrypt_key: str | None = None

    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create any missing runtime directories."""
        for d in (self.home, self.memory_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the cached process-wide :class:`Settings` instance."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
        _SETTINGS.ensure_dirs()
    return _SETTINGS


def reset_settings() -> None:
    """Drop the cached settings (tests only)."""
    global _SETTINGS
    _SETTINGS = None
