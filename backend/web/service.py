"""Application service for the web layer.

The service owns the concept registry and opens the right SQLite database for
each concept. It computes the progress summary shown in the interface and
delegates the learning-path decisions to the backend ``Parcours`` engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from ..tutor import documents
from ..tutor.domain import Concept, plan_concepts, progression, weighted_gap
from ..tutor.orchestration import Parcours
from ..tutor.storage import SQLiteStore
from .config import Settings
from .generators import make_generator
from .profile import ProfileStore


LEARNER_ID = "student"
DEFAULT_ETA = 0.3
DEFAULT_THETA = 0.75


class TutorService:
    """Coordinate the concept registry, storage and the learning engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.profile = ProfileStore(settings.profile_path)
        self.generator = make_generator(settings)

    # ------------------------------------------------------------------
    # concept registry
    # ------------------------------------------------------------------
    def _db_path(self, concept_id: str) -> Path:
        return self.settings.data_dir / "concepts" / f"{concept_id}.sqlite3"

    def _store(self, concept_id: str) -> SQLiteStore | None:
        concept = self.profile.get_concept(concept_id)
        if concept is None:
            return None
        return SQLiteStore(concept["db_path"])

    def _parcours(self, concept_id: str) -> Parcours:
        concept = self.profile.get_concept(concept_id)
        if concept is None:
            raise ValueError("concept not found")
        store = SQLiteStore(concept["db_path"])
        return Parcours(
            store,
            LEARNER_ID,
            self.generator,
            eta=concept["eta"],
            prerequisite_threshold=concept["theta"],
        )

    def list_concepts(self) -> list[dict[str, Any]]:
        return [self._decorate(concept) for concept in self.profile.list_concepts()]

    def get_concept(self, concept_id: str) -> dict[str, Any] | None:
        concept = self.profile.get_concept(concept_id)
        if concept is None:
            return None
        return {**concept, **self.summary(concept_id)}

    def _decorate(self, concept: dict[str, Any]) -> dict[str, Any]:
        summary = self.summary(concept["concept_id"])
        return {**concept, **summary}

    def create_concept(
        self,
        *,
        title: str,
        objectives: str,
        preferences: dict[str, Any],
        eta: float,
        theta: float,
        uploads: list[tuple[str, bytes]],
    ) -> str:
        """Create a concept, load its documents and start its learning path."""
        concept_id = uuid4().hex
        db_path = self._db_path(concept_id)
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.data_dir / "concepts").mkdir(parents=True, exist_ok=True)

        store = SQLiteStore(db_path)
        store.create_learner(LEARNER_ID, display_name=title)
        try:
            loaded_documents = [
                self._load_upload(filename, content)
                for filename, content in uploads
            ]
            self.profile.create_concept(
                concept_id=concept_id,
                title=title,
                objectives=objectives,
                preferences=preferences,
                eta=eta,
                theta=theta,
                db_path=str(db_path),
            )
            parcours = Parcours(
                store,
                LEARNER_ID,
                self.generator,
                eta=eta,
                prerequisite_threshold=theta,
            )
            parcours.start(loaded_documents, objectives, preferences)
        except Exception:
            db_path.unlink(missing_ok=True)
            self.profile.delete_concept(concept_id)
            raise
        return concept_id

    def _load_upload(self, filename: str, content: bytes):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md", ".markdown", ".pdf"}:
            raise ValueError(f"format de document non supporté : {suffix or '(aucun)'}")
        upload_dir = self.settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{uuid4().hex}{suffix}"
        path.write_bytes(content)
        document_id = f"{Path(filename).stem}-{uuid4().hex[:8]}"
        return documents.load_file(
            path, document_id=document_id, title=Path(filename).stem
        )

    def delete_concept(self, concept_id: str) -> None:
        concept = self.profile.get_concept(concept_id)
        if concept is None:
            raise ValueError("concept not found")
        Path(concept["db_path"]).unlink(missing_ok=True)
        self.profile.delete_concept(concept_id)

    # ------------------------------------------------------------------
    # progress summary
    # ------------------------------------------------------------------
    @staticmethod
    def _domain_concepts(graph: dict[str, Any]) -> dict[str, Concept]:
        prerequisites = {concept["concept_id"]: set() for concept in graph["concepts"]}
        for edge in graph["edges"]:
            if edge["prerequisite_id"] in prerequisites and edge["concept_id"] in prerequisites:
                prerequisites[edge["concept_id"]].add(edge["prerequisite_id"])
        return {
            concept["concept_id"]: Concept(
                concept["concept_id"],
                frozenset(prerequisites[concept["concept_id"]]),
                target_level=concept["target_level"],
                weight=concept["weight"],
            )
            for concept in graph["concepts"]
        }

    def summary(self, concept_id: str) -> dict[str, Any]:
        """Return the phase, levels, gaps and progress of one concept."""
        concept = self.profile.get_concept(concept_id)
        theta = concept["theta"] if concept else DEFAULT_THETA
        store = self._store(concept_id)
        if store is None:
            return {
                "phase": "not_started",
                "graph": None,
                "mastery": {},
                "levels": {},
                "current_gap": 0.0,
                "initial_gap": None,
                "progress": 0.0,
                "remaining": [],
                "sufficient": False,
                "concept_count": 0,
                "state": None,
            }

        state = store.get_parcours_state(LEARNER_ID) or {"phase": "not_started"}
        graph = (
            store.get_graph(LEARNER_ID, state["graph_id"])
            if state.get("graph_id")
            else None
        )
        mastery = {
            item["concept_id"]: item for item in store.get_mastery(LEARNER_ID)
        }

        levels: dict[str, float] = {}
        targets: dict[str, float] = {}
        weights: dict[str, float] = {}
        if graph:
            for item in graph["concepts"]:
                record = mastery.get(item["concept_id"], {})
                levels[item["concept_id"]] = record.get("current_level", 0.0)
                targets[item["concept_id"]] = item["target_level"]
                weights[item["concept_id"]] = item["weight"]

        current_gap = weighted_gap(levels, targets, weights) if graph else 0.0
        initial_gap = state.get("initial_gap")
        progress = progression(initial_gap, current_gap) if initial_gap is not None else 0.0

        remaining: list[str] = []
        sufficient = False
        if graph:
            remaining = list(
                plan_concepts(
                    self._domain_concepts(graph),
                    levels,
                    prerequisite_threshold=theta,
                )
            )
            sufficient = all(
                levels[item["concept_id"]] >= item["target_level"]
                for item in graph["concepts"]
            )

        return {
            "phase": state.get("phase", "not_started"),
            "graph": graph,
            "mastery": mastery,
            "levels": levels,
            "current_gap": current_gap,
            "initial_gap": initial_gap,
            "progress": progress,
            "remaining": remaining,
            "sufficient": sufficient,
            "concept_count": len(graph["concepts"]) if graph else 0,
            "state": state,
        }

    # ------------------------------------------------------------------
    # current activity
    # ------------------------------------------------------------------
    def get_activity(self, concept_id: str) -> dict[str, Any]:
        """Return what the student should see and do right now."""
        store = self._store(concept_id)
        if store is None:
            raise ValueError("concept not found")
        state = store.get_parcours_state(LEARNER_ID)
        if state is None:
            return {"kind": "not_started"}
        phase = state["phase"]

        if phase == "initial_evaluation":
            assessment = (
                store.get_assessment(state["current_assessment_id"])
                if state["current_assessment_id"]
                else None
            )
            return {
                "kind": "evaluation",
                "questions": assessment["questions"] if assessment else [],
            }

        if phase == "learning":
            if state["current_assessment_id"]:
                assessment = store.get_assessment(state["current_assessment_id"])
                question = (
                    assessment["questions"][0]
                    if assessment and assessment["questions"]
                    else None
                )
                return {
                    "kind": "exercise",
                    "question": question,
                    "concept_id": state["current_concept_id"],
                }
            if state["current_concept_id"]:
                courses = store.list_courses(
                    LEARNER_ID, concept_id=state["current_concept_id"]
                )
                course = (
                    store.get_course(LEARNER_ID, courses[-1]["course_id"])
                    if courses
                    else None
                )
                return {
                    "kind": "lesson",
                    "course": course,
                    "concept_id": state["current_concept_id"],
                }
            return {"kind": "idle"}

        if phase == "final_exam":
            assessment = (
                store.get_assessment(state["current_assessment_id"])
                if state["current_assessment_id"]
                else None
            )
            return {
                "kind": "final_exam",
                "questions": assessment["questions"] if assessment else [],
            }

        if phase == "completed":
            return {"kind": "completed"}

        return {"kind": "not_started"}

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def submit_initial_evaluation(
        self, concept_id: str, responses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._parcours(concept_id).submit_initial_evaluation(responses)

    def next_lesson(self, concept_id: str) -> dict[str, Any]:
        return self._parcours(concept_id).next_lesson()

    def next_exercise(self, concept_id: str) -> dict[str, Any]:
        return self._parcours(concept_id).next_exercise()

    def submit_exercise(self, concept_id: str, response_text: str) -> dict[str, Any]:
        return self._parcours(concept_id).submit_exercise(response_text)

    def start_final_exam(self, concept_id: str) -> dict[str, Any]:
        return self._parcours(concept_id).start_final_exam()

    def submit_final_exam(
        self, concept_id: str, responses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._parcours(concept_id).submit_final_exam(responses)

    def resume_learning(self, concept_id: str) -> dict[str, Any]:
        return self._parcours(concept_id).resume_learning()

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def update_settings(
        self,
        concept_id: str,
        *,
        title: str,
        objectives: str,
        preferences: dict[str, Any],
        eta: float,
        theta: float,
    ) -> None:
        concept = self.profile.get_concept(concept_id)
        if concept is None:
            raise ValueError("concept not found")
        self.profile.update_concept(
            concept_id,
            title=title,
            objectives=objectives,
            preferences=preferences,
            eta=eta,
            theta=theta,
        )
        store = SQLiteStore(concept["db_path"])
        state = store.get_parcours_state(LEARNER_ID)
        if state is not None:
            store.save_parcours_state(
                LEARNER_ID,
                phase=state["phase"],
                graph_id=state["graph_id"],
                initial_gap=state["initial_gap"],
                current_assessment_id=state["current_assessment_id"],
                current_concept_id=state["current_concept_id"],
                objectives=objectives,
                preferences=preferences,
            )
