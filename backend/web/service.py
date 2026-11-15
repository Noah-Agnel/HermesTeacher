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
from .graphview import layout_graph
from .jobs import ReportingGenerator
from .profile import ProfileStore


LEARNER_ID = "student"
DEFAULT_ETA = 0.3
DEFAULT_THETA = 0.75



class EvaluationPending(Exception):
    """The concept and its graph are saved but the initial evaluation failed."""

    def __init__(self, concept_id: str, message: str) -> None:
        super().__init__(message)
        self.concept_id = concept_id

class TutorService:
    """Coordinate the concept registry, storage and the learning engine."""

    def __init__(self, settings: Settings, generator: Any = None) -> None:
        self.settings = settings
        self.profile = ProfileStore(settings.profile_path)
        # The proxy reports each generator call to the running background
        # job (live progress); it delegates everything else unchanged.
        self.generator = ReportingGenerator(
            generator if generator is not None else make_generator()
        )

    def get_profile(self) -> dict[str, Any]:
        """Return the learner's global profile, with defaults when absent."""
        profile = self.profile.get_learner_profile(LEARNER_ID)
        if profile is None:
            return {
                "learner_id": LEARNER_ID,
                "display_name": None,
                "preferences": {},
                "weaknesses": [],
            }
        return profile

    def save_profile(
        self,
        *,
        display_name: str | None = None,
        preferences: dict[str, Any] | None = None,
        weaknesses: list[Any] | None = None,
    ) -> None:
        """Persist the learner's global profile."""
        self.profile.save_learner_profile(
            LEARNER_ID,
            display_name=display_name,
            preferences=preferences,
            weaknesses=weaknesses,
        )

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
        merged = {**concept, **self.summary(concept_id)}
        merged["mini_graph"] = self.mini_graph(merged)
        return merged

    def _decorate(self, concept: dict[str, Any]) -> dict[str, Any]:
        summary = self.summary(concept["concept_id"])
        decorated = {**concept, **summary}
        decorated["mini_graph"] = self.mini_graph(decorated)
        return decorated

    # ------------------------------------------------------------------
    # graph views
    # ------------------------------------------------------------------
    def graph_view(self, concept_id: str) -> dict[str, Any] | None:
        """Return the graph with each concept's level and learning status.

        Status: ``mastered`` (level ≥ target), ``current`` (being worked on),
        ``locked`` (a prerequisite is below the threshold θ) or ``open``.
        """
        concept = self.get_concept(concept_id)
        if concept is None or not concept["graph"]:
            return None
        graph = concept["graph"]
        levels = concept["levels"]
        theta = concept.get("theta", DEFAULT_THETA)
        state = concept.get("state") or {}
        prerequisites: dict[str, list[str]] = {item["concept_id"]: [] for item in graph["concepts"]}
        for edge in graph["edges"]:
            if edge["concept_id"] in prerequisites:
                prerequisites[edge["concept_id"]].append(edge["prerequisite_id"])
        nodes = []
        for item in graph["concepts"]:
            concept_key = item["concept_id"]
            level = levels.get(concept_key, 0.0)
            if level >= item["target_level"]:
                status = "mastered"
            elif concept_key == state.get("current_concept_id"):
                status = "current"
            elif any(levels.get(p, 0.0) < theta for p in prerequisites[concept_key]):
                status = "locked"
            else:
                status = "open"
            nodes.append(
                {
                    "id": concept_key,
                    "label": item["label"],
                    "description": item.get("description") or "",
                    "level": round(level, 3),
                    "target": item["target_level"],
                    "status": status,
                    "prerequisites": prerequisites[concept_key],
                }
            )
        edges = [
            {"source": edge["prerequisite_id"], "target": edge["concept_id"]}
            for edge in graph["edges"]
        ]
        return {"nodes": nodes, "edges": edges, "theta": theta}

    @staticmethod
    def mini_graph(concept: dict[str, Any]) -> dict[str, Any] | None:
        """Small constellation of the graph for tiles and the side rail.

        Positions are normalised to a 100 × 60 box; each node carries a
        mastery bucket (0–4) for its colour.
        """
        graph = concept.get("graph")
        if not graph or not graph.get("concepts"):
            return None
        layout = layout_graph(graph["concepts"], graph["edges"], concept.get("levels", {}))
        xs = [node["x"] for node in layout["nodes"]]
        ys = [node["y"] for node in layout["nodes"]]
        span_x = (max(xs) - min(xs)) or 1
        span_y = (max(ys) - min(ys)) or 1

        def scale(x: float, y: float) -> tuple[float, float]:
            return (
                round(6 + (x - min(xs)) / span_x * 88, 2),
                round(6 + (y - min(ys)) / span_y * 48, 2) if len(set(ys)) > 1 else 30.0,
            )

        position = {}
        nodes = []
        for node in layout["nodes"]:
            x, y = scale(node["x"], node["y"])
            position[node["concept_id"]] = (x, y)
            nodes.append({"x": x, "y": y, "bucket": min(4, int(node["level"] * 4.999)), "label": node["label"]})
        edges = [
            {
                "x1": position[edge["prerequisite_id"]][0],
                "y1": position[edge["prerequisite_id"]][1],
                "x2": position[edge["concept_id"]][0],
                "y2": position[edge["concept_id"]][1],
            }
            for edge in layout["edges"]
        ]
        return {"nodes": nodes, "edges": edges}

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
        loaded_documents: list[Any] = []
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
        except Exception as error:
            state = store.get_parcours_state(LEARNER_ID)
            if state is not None and state.get("graph_id"):
                # The graph is saved: keep the concept so the evaluation can
                # be retried from its page without regenerating the graph.
                raise EvaluationPending(concept_id, str(error)) from error
            db_path.unlink(missing_ok=True)
            self.profile.delete_concept(concept_id)
            self._remove_uploads([str(document.path) for document in loaded_documents])
            raise
        return concept_id

    def prepare_initial_evaluation(self, concept_id: str) -> dict[str, Any]:
        """Retry the initial evaluation of a concept whose graph is saved."""
        return self._parcours(concept_id).prepare_initial_evaluation()

    def _load_upload(self, filename: str, content: bytes):
        upload_dir = self.settings.data_dir / "uploads"
        document_id = f"{Path(filename).stem}-{uuid4().hex[:8]}"
        return documents.store_upload(
            upload_dir,
            document_id=document_id,
            title=Path(filename).stem,
            filename=filename,
            content=content,
        )

    def delete_concept(self, concept_id: str) -> None:
        """Delete a concept: its database, then its deposited files."""
        concept = self.profile.get_concept(concept_id)
        if concept is None:
            raise ValueError("concept not found")
        paths = []
        if Path(concept["db_path"]).exists():
            store = SQLiteStore(concept["db_path"])
            paths = [document["path"] for document in store.list_documents(LEARNER_ID)]
        Path(concept["db_path"]).unlink(missing_ok=True)
        self.profile.delete_concept(concept_id)
        self._remove_uploads(paths)

    def _remove_uploads(self, paths: list[str]) -> None:
        """Delete deposited files (PDF and text companion) inside the upload folder."""
        upload_dir = (self.settings.data_dir / "uploads").resolve()
        for raw in paths:
            pdf = Path(raw)
            if not pdf.is_absolute():
                # stored relative to the working directory (e.g. data/uploads/x.pdf)
                pdf = pdf.resolve()
            for path in (pdf, documents.text_path(pdf)):
                try:
                    if path.resolve().parent == upload_dir:
                        path.unlink(missing_ok=True)
                except OSError:
                    pass

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

        if phase == "not_started" and state.get("graph_id"):
            return {"kind": "evaluation_pending"}

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
        """Grade the exercise; the result also carries the level before."""
        store = self._store(concept_id)
        state = store.get_parcours_state(LEARNER_ID) if store else None
        exercise_concept = state.get("current_concept_id") if state else None
        before = None
        label = exercise_concept
        if store is not None and exercise_concept:
            for item in store.get_mastery(LEARNER_ID):
                if item["concept_id"] == exercise_concept:
                    before = item["current_level"]
                    label = item.get("concept_label") or exercise_concept
        result = self._parcours(concept_id).submit_exercise(response_text)
        return {**result, "level_before": before, "concept_label": label}

    def start_final_exam(self, concept_id: str) -> dict[str, Any]:
        return self._parcours(concept_id).start_final_exam()

    def submit_final_exam(
        self, concept_id: str, responses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._parcours(concept_id).submit_final_exam(responses)

    def resume_learning(self, concept_id: str) -> dict[str, Any]:
        return self._parcours(concept_id).resume_learning()

    def keep_course(self, concept_id: str, course_id: str) -> None:
        """Mark a generated course as kept by the learner."""
        self._parcours(concept_id).keep_course(course_id)

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
