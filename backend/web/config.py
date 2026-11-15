"""Application settings loaded from the environment.

The web layer needs to know where the learner data lives. Every value can be
overridden with a ``TUTOR_*`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the web application."""

    data_dir: Path
    profile_path: Path


def load_settings() -> Settings:
    """Build the settings from the environment, creating the data directory."""
    data_dir = Path(os.environ.get("TUTOR_DATA_DIR", "data")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "concepts").mkdir(parents=True, exist_ok=True)
    (data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    return Settings(
        data_dir=data_dir,
        profile_path=data_dir / "profile.sqlite3",
    )
