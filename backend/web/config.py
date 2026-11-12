"""Application settings loaded from the environment.

The web layer needs to know where the learner data lives and how to reach
the model that generates the pedagogical content. Every value can be
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
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: int
    demo: bool


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    """Build the settings from the environment, creating the data directory."""
    data_dir = Path(os.environ.get("TUTOR_DATA_DIR", "data")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "concepts").mkdir(parents=True, exist_ok=True)
    (data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("TUTOR_LLM_API_KEY", "")
    base_url = os.environ.get("TUTOR_LLM_BASE_URL", "https://api.openai.com/v1")
    demo_setting = os.environ.get("TUTOR_DEMO", "auto").strip().lower()
    if demo_setting == "1":
        demo = True
    elif demo_setting == "0":
        demo = False
    else:
        # Without a key there is no model to call: fall back to the demo.
        demo = not api_key

    return Settings(
        data_dir=data_dir,
        profile_path=data_dir / "profile.sqlite3",
        llm_base_url=base_url.rstrip("/"),
        llm_api_key=api_key,
        llm_model=os.environ.get("TUTOR_LLM_MODEL", "gpt-4o-mini"),
        llm_timeout=int(os.environ.get("TUTOR_LLM_TIMEOUT", "120")),
        demo=demo,
    )
