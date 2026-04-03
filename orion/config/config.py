"""Orion configuration management."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomli_w

CONFIG_DIR = Path.home() / ".orion"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULTS = {
    "nim_api_key": "",
    "default_model": "nvidia_nim/meta/llama-3.1-8b-instruct",
    "theme": "dark",
    "cwd": "",
}


@dataclass
class Config:
    """Orion runtime configuration backed by ~/.orion/config.toml."""

    nim_api_key: str = field(default=DEFAULTS["nim_api_key"])
    default_model: str = field(default=DEFAULTS["default_model"])
    theme: str = field(default=DEFAULTS["theme"])
    cwd: str = field(default=DEFAULTS["cwd"])

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> Config:
        """Load config from disk, creating defaults if the file doesn't exist.

        Environment variable ``NVIDIA_NIM_API_KEY`` overrides the stored
        ``nim_api_key`` at runtime (never written back to the file).
        """
        if CONFIG_FILE.exists():
            with CONFIG_FILE.open("rb") as f:
                data = tomllib.load(f)
            cfg = cls(
                nim_api_key=data.get("nim_api_key", DEFAULTS["nim_api_key"]),
                default_model=data.get("default_model", DEFAULTS["default_model"]),
                theme=data.get("theme", DEFAULTS["theme"]),
                cwd=data.get("cwd", DEFAULTS["cwd"]),
            )
        else:
            cfg = cls()
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            cfg.save()

        # Environment override — never persisted
        env_key = os.environ.get("NVIDIA_NIM_API_KEY")
        if env_key:
            cfg.nim_api_key = env_key

        return cfg

    # ── properties ───────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when an API key is available."""
        return bool(self.nim_api_key)

    # ── persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Write the current configuration to ``~/.orion/config.toml``."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_FILE.open("wb") as f:
            tomli_w.dump(asdict(self), f)
