from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from .domain import AttemptType, progression
from .documents import Document
from .graph import ConceptGraphProposal


_SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    learner_id TEXT PRIMARY KEY CHECK(length(trim(learner_id)) > 0),
    display_name TEXT,
    preferences_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assessments (
    assessment_id TEXT PRIMARY KEY CHECK(length(trim(assessment_id)) > 0),
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('exercise', 'evaluation', 'final_exam')),
    status TEXT NOT NULL CHECK(status IN ('in_progress', 'completed', 'abandoned')),
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    raw_score REAL CHECK(raw_score IS NULL OR (raw_score >= 0 AND raw_score <= 1)),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(assessment_id, learner_id)
);
CREATE TABLE IF NOT EXISTS assessment_questions (
    question_id TEXT PRIMARY KEY CHECK(length(trim(question_id)) > 0),
    assessment_id TEXT NOT NULL REFERENCES assessments(assessment_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK(position >= 0),
    concept_id TEXT,
    prompt TEXT NOT NULL,
    reference_answer TEXT NOT NULL DEFAULT '',
    grading_rubric TEXT NOT NULL DEFAULT '',
    difficulty_coefficient REAL NOT NULL CHECK(
        difficulty_coefficient >= 0 AND difficulty_coefficient <= 1
    ),
    citations_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE(assessment_id, position)
);
CREATE TABLE IF NOT EXISTS responses (
    response_id TEXT PRIMARY KEY CHECK(length(trim(response_id)) > 0),
    question_id TEXT NOT NULL REFERENCES assessment_questions(question_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
    response_text TEXT NOT NULL,
    raw_score REAL NOT NULL CHECK(raw_score >= 0 AND raw_score <= 1),
    assistance_coefficient REAL NOT NULL CHECK(
        assistance_coefficient >= 0 AND assistance_coefficient <= 1
    ),
    feedback TEXT NOT NULL DEFAULT '',
    submitted_at TEXT NOT NULL,
    UNIQUE(question_id, attempt_number)
);
CREATE TABLE IF NOT EXISTS concept_mastery (
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL CHECK(length(trim(concept_id)) > 0),
    concept_label TEXT,
    current_level REAL NOT NULL CHECK(current_level >= 0 AND current_level <= 1),
    target_level REAL NOT NULL CHECK(target_level >= 0 AND target_level <= 1),
    weight REAL NOT NULL CHECK(weight >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(learner_id, concept_id)
);
CREATE TABLE IF NOT EXISTS progress_snapshots (
    snapshot_id TEXT PRIMARY KEY CHECK(length(trim(snapshot_id)) > 0),
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    assessment_id TEXT,
    initial_gap REAL NOT NULL CHECK(initial_gap >= 0),
    current_gap REAL NOT NULL CHECK(current_gap >= 0),
    progress_fraction REAL NOT NULL CHECK(progress_fraction >= 0 AND progress_fraction <= 1),
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY CHECK(length(trim(document_id)) > 0),
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS errors (
    error_id TEXT PRIMARY KEY CHECK(length(trim(error_id)) > 0),
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL CHECK(length(trim(concept_id)) > 0),
    description TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    occurred_at TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE(error_id, learner_id)
);
CREATE TABLE IF NOT EXISTS history_events (
    event_id TEXT PRIMARY KEY CHECK(length(trim(event_id)) > 0),
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    concept_id TEXT,
    kind TEXT NOT NULL CHECK(length(trim(kind)) > 0),
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS concept_graphs (
    graph_id TEXT PRIMARY KEY CHECK(length(trim(graph_id)) > 0),
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('proposed', 'validated', 'corrected')),
    notes TEXT NOT NULL DEFAULT '',
    source_document_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_concepts (
    graph_id TEXT NOT NULL REFERENCES concept_graphs(graph_id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL CHECK(length(trim(concept_id)) > 0),
    label TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    target_level REAL NOT NULL CHECK(target_level >= 0 AND target_level <= 1),
    weight REAL NOT NULL CHECK(weight >= 0),
    evidence_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(graph_id, concept_id)
);
CREATE TABLE IF NOT EXISTS graph_edges (
    graph_id TEXT NOT NULL REFERENCES concept_graphs(graph_id) ON DELETE CASCADE,
    prerequisite_id TEXT NOT NULL CHECK(length(trim(prerequisite_id)) > 0),
    concept_id TEXT NOT NULL CHECK(length(trim(concept_id)) > 0),
    evidence_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(graph_id, prerequisite_id, concept_id)
);
CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY CHECK(length(trim(course_id)) > 0),
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL CHECK(length(trim(concept_id)) > 0),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    saved INTEGER NOT NULL DEFAULT 0 CHECK(saved IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parcours_state (
    learner_id TEXT PRIMARY KEY REFERENCES learners(learner_id) ON DELETE CASCADE,
    graph_id TEXT,
    phase TEXT NOT NULL CHECK(
        phase IN ('not_started', 'initial_evaluation', 'learning', 'final_exam', 'completed')
    ),
    initial_gap REAL,
    current_assessment_id TEXT,
    current_concept_id TEXT,
    objectives TEXT NOT NULL DEFAULT '',
    preferences_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);
"""


class SQLiteStore:
    """SQLite persistence for learner data; schema creation is idempotent."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(_SCHEMA)
            self._migrate(connection)

    @staticmethod
    def _migrate(connection) -> None:
        """Bring an older database up to the current schema.

        The prototype evolves quickly; this only handles the changes that
        affect existing databases. A database created before a change is
        migrated in place when possible, otherwise it must be recreated.
        """
        document_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        if document_columns and "path" not in document_columns:
            connection.execute("ALTER TABLE documents RENAME TO documents_v1")
            connection.execute(
                """CREATE TABLE documents (
                       document_id TEXT PRIMARY KEY CHECK(length(trim(document_id)) > 0),
                       learner_id TEXT NOT NULL REFERENCES learners(learner_id)
                           ON DELETE CASCADE,
                       title TEXT NOT NULL,
                       path TEXT NOT NULL,
                       media_type TEXT NOT NULL,
                       metadata_json TEXT NOT NULL DEFAULT '{}',
                       created_at TEXT NOT NULL
                   )"""
            )
            connection.execute(
                """INSERT INTO documents
                   (document_id, learner_id, title, path, media_type,
                    metadata_json, created_at)
                   SELECT document_id, learner_id, title, source, media_type,
                          metadata_json, created_at
                   FROM documents_v1"""
            )
            connection.execute("DROP TABLE documents_v1")
            connection.execute("DROP TABLE IF EXISTS passages")

        course_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(courses)").fetchall()
        }
        if course_columns and "saved" not in course_columns:
            connection.execute(
                "ALTER TABLE courses ADD COLUMN saved INTEGER NOT NULL DEFAULT 0"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _encode_preferences(preferences: Mapping[str, Any] | None) -> str:
        if preferences is None:
            preferences = {}
        if not isinstance(preferences, Mapping):
            raise ValueError("preferences must be a mapping")
        try:
            return json.dumps(dict(preferences), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("preferences must contain JSON-compatible values") from error

    @staticmethod
    def _encode_json_object(value: Mapping[str, Any] | None, name: str) -> str:
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        try:
            return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain JSON-compatible values") from error

    @staticmethod
    def _encode_citations(citations: list[Mapping[str, Any]] | None) -> str:
        if citations is None:
            citations = []
        if not isinstance(citations, list):
            raise ValueError("citations must be a list")
        encoded = []
        for item in citations:
            if not isinstance(item, Mapping):
                raise ValueError("each citation must be a mapping")
            document_id = item.get("document_id")
            page = item.get("page")
            quote = item.get("quote")
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("citation document_id must be a non-empty string")
            if page is not None:
                if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                    raise ValueError("citation page must be a positive integer or None")
            if quote is not None and not isinstance(quote, str):
                raise ValueError("citation quote must be a string or None")
            encoded.append(
                {"document_id": document_id, "page": page, "quote": quote}
            )
        return json.dumps(encoded, ensure_ascii=False)

    @staticmethod
    def _validate_unit_interval(value: float, name: str) -> None:
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    def create_assessment(
        self,
        assessment_id: str,
        learner_id: str,
        kind: AttemptType | str,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._validate_identifier(assessment_id, "assessment_id")
        self._validate_identifier(learner_id, "learner_id")
        try:
            kind_value = AttemptType(kind).value
        except ValueError as error:
            raise ValueError("kind must be exercise, evaluation, or final_exam") from error
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO assessments
                   (assessment_id, learner_id, kind, status, source_metadata_json, created_at)
                   VALUES (?, ?, ?, 'in_progress', ?, ?)""",
                (
                    assessment_id,
                    learner_id,
                    kind_value,
                    self._encode_json_object(source_metadata, "source_metadata"),
                    now,
                ),
            )

    def add_question(
        self,
        assessment_id: str,
        question_id: str,
        position: int,
        prompt: str,
        reference_answer: str = "",
        grading_rubric: str = "",
        concept_id: str | None = None,
        difficulty_coefficient: float = 1.0,
        citations: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self._validate_identifier(assessment_id, "assessment_id")
        self._validate_identifier(question_id, "question_id")
        self._validate_identifier(prompt, "prompt")
        self._validate_unit_interval(difficulty_coefficient, "difficulty_coefficient")
        if not isinstance(position, int) or position < 0:
            raise ValueError("position must be a non-negative integer")
        if concept_id is not None:
            self._validate_identifier(concept_id, "concept_id")
        citations_json = self._encode_citations(citations)
        with self._connection() as connection:
            assessment = connection.execute(
                "SELECT status FROM assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
            if assessment is None or assessment["status"] != "in_progress":
                raise ValueError("questions can only be added to an in-progress assessment")
            connection.execute(
                """INSERT INTO assessment_questions
                   (question_id, assessment_id, position, concept_id, prompt,
                    reference_answer, grading_rubric, difficulty_coefficient,
                    citations_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    question_id,
                    assessment_id,
                    position,
                    concept_id,
                    prompt,
                    reference_answer,
                    grading_rubric,
                    difficulty_coefficient,
                    citations_json,
                ),
            )

    def record_response(
        self,
        response_id: str,
        question_id: str,
        response_text: str,
        score: float,
        assistance_coefficient: float = 1.0,
        attempt_number: int | None = None,
        feedback: str = "",
    ) -> int:
        self._validate_identifier(response_id, "response_id")
        self._validate_identifier(question_id, "question_id")
        if not isinstance(response_text, str):
            raise ValueError("response_text must be a string")
        if not isinstance(feedback, str):
            raise ValueError("feedback must be a string")
        self._validate_unit_interval(score, "score")
        self._validate_unit_interval(assistance_coefficient, "assistance_coefficient")

        with self._connection() as connection:
            row = connection.execute(
                """SELECT a.status, COALESCE(MAX(r.attempt_number), 0) + 1 AS next_attempt
                   FROM assessment_questions AS q
                   JOIN assessments AS a ON a.assessment_id = q.assessment_id
                   LEFT JOIN responses AS r ON r.question_id = q.question_id
                   WHERE q.question_id = ?
                   GROUP BY q.question_id""",
                (question_id,),
            ).fetchone()
            if row is None:
                raise ValueError("question does not exist")
            if row["status"] != "in_progress":
                raise ValueError("responses can only be added to an in-progress assessment")
            if attempt_number is None:
                attempt_number = row["next_attempt"]
            elif not isinstance(attempt_number, int) or attempt_number < 1:
                raise ValueError("attempt_number must be a positive integer")
            now = self._now()
            connection.execute(
                """INSERT INTO responses
                   (response_id, question_id, attempt_number, response_text, raw_score,
                    assistance_coefficient, feedback, submitted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    response_id,
                    question_id,
                    attempt_number,
                    response_text,
                    score,
                    assistance_coefficient,
                    feedback,
                    now,
                ),
            )
        return attempt_number

    def complete_assessment(self, assessment_id: str, raw_score: float) -> None:
        self._validate_unit_interval(raw_score, "raw_score")
        now = self._now()
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE assessments
                   SET status = 'completed', raw_score = ?, completed_at = ?
                   WHERE assessment_id = ? AND status = 'in_progress'""",
                (raw_score, now, assessment_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("assessment does not exist or is not in progress")

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            if row is None:
                return None
            assessment = dict(row)
            assessment["source_metadata"] = json.loads(
                assessment.pop("source_metadata_json")
            )
            question_rows = connection.execute(
                """SELECT * FROM assessment_questions
                   WHERE assessment_id = ? ORDER BY position""",
                (assessment_id,),
            ).fetchall()
            questions = []
            for question_row in question_rows:
                question = dict(question_row)
                question["citations"] = json.loads(question.pop("citations_json"))
                question["responses"] = [
                    dict(response_row)
                    for response_row in connection.execute(
                        """SELECT * FROM responses WHERE question_id = ?
                           ORDER BY attempt_number""",
                        (question["question_id"],),
                    ).fetchall()
                ]
                questions.append(question)
            assessment["questions"] = questions
        return assessment

    def create_learner(
        self,
        learner_id: str,
        display_name: str | None = None,
        preferences: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(learner_id, str) or not learner_id.strip():
            raise ValueError("learner_id must be a non-empty string")
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO learners
                   (learner_id, display_name, preferences_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    learner_id,
                    display_name,
                    self._encode_preferences(preferences),
                    now,
                    now,
                ),
            )

    def get_learner(self, learner_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM learners WHERE learner_id = ?", (learner_id,)
            ).fetchone()
        if row is None:
            return None
        learner = dict(row)
        learner["preferences"] = json.loads(learner.pop("preferences_json"))
        return learner

    def save_mastery(
        self,
        learner_id: str,
        concept_id: str,
        current_level: float,
        target_level: float = 1.0,
        weight: float = 1.0,
        concept_label: str | None = None,
    ) -> None:
        self._validate_identifier(learner_id, "learner_id")
        self._validate_identifier(concept_id, "concept_id")
        self._validate_unit_interval(current_level, "current_level")
        self._validate_unit_interval(target_level, "target_level")
        if not isinstance(weight, (int, float)) or weight < 0 or not isfinite(weight):
            raise ValueError("weight must be finite and non-negative")
        if concept_label is not None and not isinstance(concept_label, str):
            raise ValueError("concept_label must be a string or None")
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO concept_mastery
                   (learner_id, concept_id, concept_label, current_level, target_level,
                    weight, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(learner_id, concept_id) DO UPDATE SET
                       concept_label = excluded.concept_label,
                       current_level = excluded.current_level,
                       target_level = excluded.target_level,
                       weight = excluded.weight,
                       updated_at = excluded.updated_at""",
                (
                    learner_id,
                    concept_id,
                    concept_label,
                    current_level,
                    target_level,
                    weight,
                    self._now(),
                ),
            )

    def get_mastery(self, learner_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT concept_id, concept_label, current_level, target_level,
                          weight, updated_at
                   FROM concept_mastery WHERE learner_id = ? ORDER BY concept_id""",
                (learner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # errors made by the learner
    # ------------------------------------------------------------------
    def record_error(
        self,
        learner_id: str,
        concept_id: str,
        description: str,
        source: str = "",
    ) -> str:
        """Record an error the learner made, always tied to a concept."""
        self._validate_identifier(learner_id, "learner_id")
        self._validate_identifier(concept_id, "concept_id")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        error_id = uuid4().hex
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO errors
                   (error_id, learner_id, concept_id, description, source,
                    occurred_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    error_id,
                    learner_id,
                    concept_id,
                    description,
                    source,
                    self._now(),
                ),
            )
        return error_id

    def list_errors(
        self,
        learner_id: str,
        concept_id: str | None = None,
        *,
        resolved: bool | None = None,
    ) -> list[dict[str, Any]]:
        """List errors, optionally filtered by concept and resolution state."""
        query = "SELECT * FROM errors WHERE learner_id = ?"
        parameters: list[Any] = [learner_id]
        if concept_id is not None:
            self._validate_identifier(concept_id, "concept_id")
            query += " AND concept_id = ?"
            parameters.append(concept_id)
        if resolved is True:
            query += " AND resolved_at IS NOT NULL"
        elif resolved is False:
            query += " AND resolved_at IS NULL"
        query += " ORDER BY occurred_at, error_id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def resolve_error(self, learner_id: str, error_id: str) -> None:
        """Mark an error as addressed by the learner."""
        self._validate_identifier(learner_id, "learner_id")
        self._validate_identifier(error_id, "error_id")
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE errors SET resolved_at = ?
                   WHERE error_id = ? AND learner_id = ? AND resolved_at IS NULL""",
                (self._now(), error_id, learner_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("error does not exist or is already resolved")

    # ------------------------------------------------------------------
    # history of the learner's work
    # ------------------------------------------------------------------
    def log_event(
        self,
        learner_id: str,
        kind: str,
        concept_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        """Append one event to the learner's work history.

        ``kind`` is a stable label such as ``document_deposited``,
        ``course_read``, ``exercise_submitted`` or ``error_recorded``.
        """
        self._validate_identifier(learner_id, "learner_id")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("kind must be a non-empty string")
        if concept_id is not None:
            self._validate_identifier(concept_id, "concept_id")
        event_id = uuid4().hex
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO history_events
                   (event_id, learner_id, concept_id, kind, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    learner_id,
                    concept_id,
                    kind,
                    self._encode_json_object(payload, "payload"),
                    self._now(),
                ),
            )
        return event_id

    def list_history(
        self, learner_id: str, concept_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the learner's work history, oldest first."""
        query = "SELECT * FROM history_events WHERE learner_id = ?"
        parameters: list[Any] = [learner_id]
        if concept_id is not None:
            self._validate_identifier(concept_id, "concept_id")
            query += " AND concept_id = ?"
            parameters.append(concept_id)
        query += " ORDER BY created_at, event_id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json"))
            events.append(event)
        return events

    def record_progress(
        self,
        learner_id: str,
        initial_gap: float,
        current_gap: float,
        assessment_id: str | None = None,
    ) -> str:
        self._validate_identifier(learner_id, "learner_id")
        if assessment_id is not None:
            self._validate_identifier(assessment_id, "assessment_id")
        progress_fraction = progression(initial_gap, current_gap)
        snapshot_id = uuid4().hex
        with self._connection() as connection:
            if assessment_id is not None:
                assessment = connection.execute(
                    """SELECT 1 FROM assessments
                       WHERE assessment_id = ? AND learner_id = ?""",
                    (assessment_id, learner_id),
                ).fetchone()
                if assessment is None:
                    raise ValueError("assessment must belong to the same learner")
            connection.execute(
                """INSERT INTO progress_snapshots
                   (snapshot_id, learner_id, assessment_id, initial_gap, current_gap,
                    progress_fraction, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    learner_id,
                    assessment_id,
                    initial_gap,
                    current_gap,
                    progress_fraction,
                    self._now(),
                ),
            )
        return snapshot_id

    def get_progress_history(self, learner_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT snapshot_id, assessment_id, initial_gap, current_gap,
                          progress_fraction, recorded_at
                   FROM progress_snapshots WHERE learner_id = ?
                   ORDER BY recorded_at, rowid""",
                (learner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_document(self, learner_id: str, document: Document) -> None:
        """Insert or replace a deposited document's metadata.

        The file itself is already on disk; only its location and metadata are
        recorded here. No passage is created: the agent reads the file.
        """
        self._validate_identifier(learner_id, "learner_id")
        if not isinstance(document, Document):
            raise ValueError("document must be a Document")
        now = self._now()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT learner_id FROM documents WHERE document_id = ?",
                (document.document_id,),
            ).fetchone()
            if existing is not None and existing["learner_id"] != learner_id:
                raise ValueError("document belongs to another learner")
            connection.execute(
                """INSERT INTO documents
                   (document_id, learner_id, title, path, media_type,
                    metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                       learner_id = excluded.learner_id,
                       title = excluded.title,
                       path = excluded.path,
                       media_type = excluded.media_type,
                       metadata_json = excluded.metadata_json""",
                (
                    document.document_id,
                    learner_id,
                    document.title,
                    document.path,
                    document.media_type,
                    self._encode_json_object(dict(document.metadata), "metadata"),
                    now,
                ),
            )

    def get_document(self, learner_id: str, document_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ? AND learner_id = ?",
                (document_id, learner_id),
            ).fetchone()
            if row is None:
                return None
            document = dict(row)
            document["metadata"] = json.loads(document.pop("metadata_json"))
            return document

    def list_documents(self, learner_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT document_id, title, path, media_type, created_at
                   FROM documents WHERE learner_id = ?
                   ORDER BY created_at, document_id""",
                (learner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_graph(
        self,
        learner_id: str,
        proposal: ConceptGraphProposal,
        status: str = "proposed",
    ) -> None:
        """Insert or replace a concept graph proposal and its content."""
        self._validate_identifier(learner_id, "learner_id")
        if not isinstance(proposal, ConceptGraphProposal):
            raise ValueError("proposal must be a ConceptGraphProposal")
        if status not in ("proposed", "validated", "corrected"):
            raise ValueError("status must be proposed, validated, or corrected")
        now = self._now()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT learner_id FROM concept_graphs WHERE graph_id = ?",
                (proposal.graph_id,),
            ).fetchone()
            if existing is not None and existing["learner_id"] != learner_id:
                raise ValueError("graph belongs to another learner")
            connection.execute(
                """INSERT INTO concept_graphs
                   (graph_id, learner_id, status, notes, source_document_ids_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(graph_id) DO UPDATE SET
                       learner_id = excluded.learner_id,
                       status = excluded.status,
                       notes = excluded.notes,
                       source_document_ids_json = excluded.source_document_ids_json,
                       updated_at = excluded.updated_at""",
                (
                    proposal.graph_id,
                    learner_id,
                    status,
                    proposal.notes,
                    json.dumps(list(proposal.source_document_ids), ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM graph_concepts WHERE graph_id = ?", (proposal.graph_id,)
            )
            connection.execute(
                "DELETE FROM graph_edges WHERE graph_id = ?", (proposal.graph_id,)
            )
            for concept in proposal.concepts:
                connection.execute(
                    """INSERT INTO graph_concepts
                       (graph_id, concept_id, label, description, target_level,
                        weight, evidence_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        proposal.graph_id,
                        concept.concept_id,
                        concept.label,
                        concept.description,
                        concept.target_level,
                        concept.weight,
                        json.dumps(
                            [item.to_mapping() for item in concept.evidence],
                            ensure_ascii=False,
                        ),
                    ),
                )
            for edge in proposal.edges:
                connection.execute(
                    """INSERT INTO graph_edges
                       (graph_id, prerequisite_id, concept_id, evidence_json)
                       VALUES (?, ?, ?, ?)""",
                    (
                        proposal.graph_id,
                        edge.prerequisite_id,
                        edge.concept_id,
                        json.dumps(
                            [item.to_mapping() for item in edge.evidence],
                            ensure_ascii=False,
                        ),
                    ),
                )

    def get_graph(self, learner_id: str, graph_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM concept_graphs WHERE graph_id = ? AND learner_id = ?",
                (graph_id, learner_id),
            ).fetchone()
            if row is None:
                return None
            graph = dict(row)
            graph["source_document_ids"] = json.loads(
                graph.pop("source_document_ids_json")
            )
            graph["concepts"] = []
            for concept_row in connection.execute(
                "SELECT * FROM graph_concepts WHERE graph_id = ? ORDER BY concept_id",
                (graph_id,),
            ).fetchall():
                concept = dict(concept_row)
                concept["evidence"] = json.loads(concept.pop("evidence_json"))
                graph["concepts"].append(concept)
            graph["edges"] = []
            for edge_row in connection.execute(
                """SELECT * FROM graph_edges WHERE graph_id = ?
                   ORDER BY concept_id, prerequisite_id""",
                (graph_id,),
            ).fetchall():
                edge = dict(edge_row)
                edge["evidence"] = json.loads(edge.pop("evidence_json"))
                graph["edges"].append(edge)
            return graph

    def list_graphs(self, learner_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT g.graph_id, g.status, g.notes, g.created_at, g.updated_at,
                          COUNT(DISTINCT c.concept_id) AS concept_count,
                          COUNT(DISTINCT e.prerequisite_id || '->' || e.concept_id)
                              AS edge_count
                   FROM concept_graphs AS g
                   LEFT JOIN graph_concepts AS c ON c.graph_id = g.graph_id
                   LEFT JOIN graph_edges AS e ON e.graph_id = g.graph_id
                   WHERE g.learner_id = ?
                   GROUP BY g.graph_id
                   ORDER BY g.updated_at, g.graph_id""",
                (learner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_graph_status(self, learner_id: str, graph_id: str, status: str) -> None:
        if status not in ("proposed", "validated", "corrected"):
            raise ValueError("status must be proposed, validated, or corrected")
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE concept_graphs SET status = ?, updated_at = ?
                   WHERE graph_id = ? AND learner_id = ?""",
                (status, self._now(), graph_id, learner_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("graph does not exist for this learner")

    def save_course(
        self,
        learner_id: str,
        course_id: str,
        concept_id: str,
        title: str,
        content: str = "",
        citations: list[Mapping[str, Any]] | None = None,
        saved: bool = False,
    ) -> None:
        """Insert or replace a generated mini-course.

        ``saved`` marks a course the learner chose to keep; unsaved courses are
        deleted once the learner moves on to the next exercise.
        """
        self._validate_identifier(learner_id, "learner_id")
        self._validate_identifier(course_id, "course_id")
        self._validate_identifier(concept_id, "concept_id")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if not isinstance(saved, bool):
            raise ValueError("saved must be a boolean")
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO courses
                   (course_id, learner_id, concept_id, title, content,
                    citations_json, saved, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(course_id) DO UPDATE SET
                       learner_id = excluded.learner_id,
                       concept_id = excluded.concept_id,
                       title = excluded.title,
                       content = excluded.content,
                       citations_json = excluded.citations_json,
                       saved = excluded.saved""",
                (
                    course_id,
                    learner_id,
                    concept_id,
                    title,
                    content,
                    self._encode_citations(citations),
                    1 if saved else 0,
                    self._now(),
                ),
            )

    def get_course(self, learner_id: str, course_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM courses WHERE course_id = ? AND learner_id = ?",
                (course_id, learner_id),
            ).fetchone()
            if row is None:
                return None
            course = dict(row)
            course["citations"] = json.loads(course.pop("citations_json"))
            course["saved"] = bool(course.get("saved", 0))
            return course

    def list_courses(
        self, learner_id: str, concept_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            query = (
                """SELECT course_id, learner_id, concept_id, title, saved, created_at
                   FROM courses WHERE learner_id = ?"""
            )
            parameters: list[Any] = [learner_id]
            if concept_id is not None:
                query += " AND concept_id = ?"
                parameters.append(concept_id)
            query += " ORDER BY created_at, course_id"
            rows = connection.execute(query, parameters).fetchall()
        courses = []
        for row in rows:
            course = dict(row)
            course["saved"] = bool(course.get("saved", 0))
            courses.append(course)
        return courses

    def set_course_saved(
        self, learner_id: str, course_id: str, saved: bool
    ) -> None:
        """Mark a generated course as kept by the learner, or not."""
        self._validate_identifier(learner_id, "learner_id")
        self._validate_identifier(course_id, "course_id")
        if not isinstance(saved, bool):
            raise ValueError("saved must be a boolean")
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE courses SET saved = ?
                   WHERE course_id = ? AND learner_id = ?""",
                (1 if saved else 0, course_id, learner_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("course does not exist for this learner")

    def delete_course(self, learner_id: str, course_id: str) -> None:
        """Delete a generated course that the learner did not keep."""
        self._validate_identifier(learner_id, "learner_id")
        self._validate_identifier(course_id, "course_id")
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM courses WHERE course_id = ? AND learner_id = ?",
                (course_id, learner_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("course does not exist for this learner")

    def save_parcours_state(
        self,
        learner_id: str,
        phase: str,
        graph_id: str | None = None,
        initial_gap: float | None = None,
        current_assessment_id: str | None = None,
        current_concept_id: str | None = None,
        objectives: str = "",
        preferences: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert or replace the learner's place in the learning path."""
        self._validate_identifier(learner_id, "learner_id")
        if phase not in (
            "not_started",
            "initial_evaluation",
            "learning",
            "final_exam",
            "completed",
        ):
            raise ValueError("unknown parcours phase")
        if not isinstance(objectives, str):
            raise ValueError("objectives must be a string")
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO parcours_state
                   (learner_id, graph_id, phase, initial_gap, current_assessment_id,
                    current_concept_id, objectives, preferences_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(learner_id) DO UPDATE SET
                       graph_id = excluded.graph_id,
                       phase = excluded.phase,
                       initial_gap = excluded.initial_gap,
                       current_assessment_id = excluded.current_assessment_id,
                       current_concept_id = excluded.current_concept_id,
                       objectives = excluded.objectives,
                       preferences_json = excluded.preferences_json,
                       updated_at = excluded.updated_at""",
                (
                    learner_id,
                    graph_id,
                    phase,
                    initial_gap,
                    current_assessment_id,
                    current_concept_id,
                    objectives,
                    self._encode_json_object(preferences, "preferences"),
                    self._now(),
                ),
            )

    def get_parcours_state(self, learner_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM parcours_state WHERE learner_id = ?",
                (learner_id,),
            ).fetchone()
        if row is None:
            return None
        state = dict(row)
        state["preferences"] = json.loads(state.pop("preferences_json"))
        return state
