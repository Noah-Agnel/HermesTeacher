"""Concept registry and global settings.

Each concept is an independent learning path: it owns its documents, graph,
progress and courses, stored in its own SQLite database. This module only
manages the registry that lists those concepts and the few global settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4


_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY CHECK(length(trim(concept_id)) > 0),
    title TEXT NOT NULL,
    objectives TEXT NOT NULL DEFAULT '',
    preferences_json TEXT NOT NULL DEFAULT '{}',
    eta REAL NOT NULL DEFAULT 0.3 CHECK(eta >= 0 AND eta <= 1),
    theta REAL NOT NULL DEFAULT 0.75 CHECK(theta >= 0 AND theta <= 1),
    db_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learner_profile (
    learner_id TEXT PRIMARY KEY CHECK(length(trim(learner_id)) > 0),
    display_name TEXT,
    preferences_json TEXT NOT NULL DEFAULT '{}',
    weaknesses_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);
"""


class ProfileStore:
    """SQLite registry of the learner's concepts and global settings."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _encode_preferences(preferences: dict[str, Any]) -> str:
        try:
            return json.dumps(dict(preferences), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("preferences must contain JSON-compatible values") from error

    def create_concept(
        self,
        *,
        concept_id: str | None = None,
        title: str,
        objectives: str,
        preferences: dict[str, Any],
        eta: float,
        theta: float,
        db_path: str,
    ) -> str:
        concept_id = concept_id or uuid4().hex
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO concepts
                   (concept_id, title, objectives, preferences_json, eta, theta,
                    db_path, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    concept_id,
                    title,
                    objectives,
                    self._encode_preferences(preferences),
                    eta,
                    theta,
                    db_path,
                    now,
                    now,
                ),
            )
        return concept_id

    def get_concept(self, concept_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM concepts WHERE concept_id = ?", (concept_id,)
            ).fetchone()
        if row is None:
            return None
        concept = dict(row)
        concept["preferences"] = json.loads(concept.pop("preferences_json"))
        return concept

    def list_concepts(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM concepts ORDER BY created_at, concept_id"
            ).fetchall()
        concepts = []
        for row in rows:
            concept = dict(row)
            concept["preferences"] = json.loads(concept.pop("preferences_json"))
            concepts.append(concept)
        return concepts

    def update_concept(self, concept_id: str, **fields: Any) -> None:
        allowed = {"title", "objectives", "preferences", "eta", "theta"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown concept fields: {sorted(unknown)!r}")
        if "eta" in fields and not 0 <= fields["eta"] <= 1:
            raise ValueError("eta must be between 0 and 1")
        if "theta" in fields and not 0 <= fields["theta"] <= 1:
            raise ValueError("theta must be between 0 and 1")
        if "preferences" in fields and not isinstance(fields["preferences"], dict):
            raise ValueError("preferences must be a mapping")
        if "eta" in fields and (
            not isinstance(fields["eta"], (int, float)) or not isfinite(fields["eta"])
        ):
            raise ValueError("eta must be finite")

        assignments = []
        parameters: list[Any] = []
        for name, value in fields.items():
            if name == "preferences":
                value = self._encode_preferences(value)
                assignments.append("preferences_json = ?")
            else:
                assignments.append(f"{name} = ?")
            parameters.append(value)
        parameters.append(self._now())
        parameters.append(concept_id)
        with self._connection() as connection:
            cursor = connection.execute(
                f"""UPDATE concepts SET {', '.join(assignments)}, updated_at = ?
                   WHERE concept_id = ?""",
                parameters,
            )
            if cursor.rowcount != 1:
                raise ValueError("concept does not exist")

    def delete_concept(self, concept_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM concepts WHERE concept_id = ?", (concept_id,))

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        return row["value"]

    def set_setting(self, key: str, value: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO settings (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )

    # ------------------------------------------------------------------
    # global learner profile
    # ------------------------------------------------------------------
    def get_learner_profile(self, learner_id: str) -> dict[str, Any] | None:
        """Return the learner's global profile, or None if it does not exist."""
        if not isinstance(learner_id, str) or not learner_id.strip():
            raise ValueError("learner_id must be a non-empty string")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM learner_profile WHERE learner_id = ?", (learner_id,)
            ).fetchone()
        if row is None:
            return None
        profile = dict(row)
        profile["preferences"] = json.loads(profile.pop("preferences_json"))
        profile["weaknesses"] = json.loads(profile.pop("weaknesses_json"))
        return profile

    def save_learner_profile(
        self,
        learner_id: str,
        *,
        display_name: str | None = None,
        preferences: Mapping[str, Any] | None = None,
        weaknesses: list[Any] | None = None,
    ) -> None:
        """Insert or replace the learner's global profile.

        ``preferences`` describes how the learner likes to learn; ``weaknesses``
        is a list of transversal points the agent identified.
        """
        if not isinstance(learner_id, str) or not learner_id.strip():
            raise ValueError("learner_id must be a non-empty string")
        if preferences is not None and not isinstance(preferences, Mapping):
            raise ValueError("preferences must be a mapping")
        if weaknesses is not None and not isinstance(weaknesses, list):
            raise ValueError("weaknesses must be a list")
        preferences_json = self._encode_preferences(dict(preferences or {}))
        try:
            weaknesses_json = json.dumps(weaknesses or [], ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "weaknesses must contain JSON-compatible values"
            ) from error
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO learner_profile
                   (learner_id, display_name, preferences_json, weaknesses_json,
                    updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(learner_id) DO UPDATE SET
                       display_name = excluded.display_name,
                       preferences_json = excluded.preferences_json,
                       weaknesses_json = excluded.weaknesses_json,
                       updated_at = excluded.updated_at""",
                (learner_id, display_name, preferences_json, weaknesses_json, now),
            )
