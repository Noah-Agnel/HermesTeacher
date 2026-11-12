"""Learning-path orchestration.

The Parcours engine drives the nominal flow: initial evaluation, learning,
adaptation, final exam and remediation. Deterministic decisions (levels, gaps,
ordering, progress, persistence) stay in the backend. The Generator protocol is
the LLM boundary: the agent reads the learner's documents and generates courses,
exercises and evaluations, and grades free-form responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any
from uuid import uuid4

from .domain import (
    Attempt,
    AttemptType,
    DEFAULT_PREREQUISITE_THRESHOLD,
    plan_concepts,
    progression,
    update_level_from_attempt,
    weighted_gap,
)
from .documents import Document
from .graph import ConceptGraphProposal

DEFAULT_ETA = 0.3


class Generator(ABC):
    """The LLM boundary: reading documents and generating content.

    Implementations return JSON-compatible mappings. The engine validates them
    and turns them into persisted assessments, questions, courses and grades.
    """

    @abstractmethod
    def propose_graph(
        self,
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Propose a concept graph from the learner's document passages."""

    @abstractmethod
    def generate_initial_evaluation(
        self,
        concepts: list[dict[str, Any]],
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Generate the initial evaluation covering the concepts."""

    @abstractmethod
    def generate_course(
        self,
        concept: dict[str, Any],
        passages: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Generate a mini-course adapted to the concept and level."""

    @abstractmethod
    def generate_exercise(
        self,
        concept: dict[str, Any],
        passages: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Generate an exercise for the concept."""

    @abstractmethod
    def generate_final_exam(
        self,
        concepts: list[dict[str, Any]],
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Generate the final exam."""

    @abstractmethod
    def grade_response(
        self, question: dict[str, Any], response_text: str
    ) -> Mapping[str, Any]:
        """Grade a free-form response and estimate the assistance used."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _parse_citations(
    value: Any, known_passages: set[str] | None = None
) -> list[dict[str, Any]]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError("citations must be a list")
    citations = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each citation must be a mapping")
        document_id = item.get("document_id")
        passage_id = item.get("passage_id")
        quote = item.get("quote")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("citation document_id must be a non-empty string")
        if not isinstance(passage_id, str) or not passage_id.strip():
            raise ValueError("citation passage_id must be a non-empty string")
        if quote is not None and not isinstance(quote, str):
            raise ValueError("citation quote must be a string or None")
        if known_passages is not None and passage_id not in known_passages:
            raise ValueError(
                f"citation references unknown passage {passage_id!r}"
            )
        citations.append(
            {"document_id": document_id, "passage_id": passage_id, "quote": quote}
        )
    return citations


def parse_question(
    data: Any, known_passages: set[str] | None = None
) -> dict[str, Any]:
    data = _require_mapping(data, "question")
    difficulty = data.get("difficulty_coefficient", 1.0)
    if not isinstance(difficulty, (int, float)) or not 0 <= difficulty <= 1:
        raise ValueError("difficulty_coefficient must be between 0 and 1")
    concept_id = data.get("concept_id")
    if concept_id is not None:
        concept_id = _require_text(concept_id, "concept_id")
    return {
        "question_id": _require_text(data.get("question_id"), "question_id"),
        "concept_id": concept_id,
        "prompt": _require_text(data.get("prompt"), "prompt"),
        "reference_answer": _require_text(
            data.get("reference_answer", ""), "reference_answer", allow_empty=True
        ),
        "grading_rubric": _require_text(
            data.get("grading_rubric", ""), "grading_rubric", allow_empty=True
        ),
        "difficulty_coefficient": difficulty,
        "citations": _parse_citations(data.get("citations"), known_passages),
    }


def parse_assessment(
    data: Any, known_passages: set[str] | None = None
) -> list[dict[str, Any]]:
    data = _require_mapping(data, "assessment")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("assessment must contain at least one question")
    return [parse_question(item, known_passages) for item in questions]


def parse_course(
    data: Any, known_passages: set[str] | None = None
) -> dict[str, Any]:
    data = _require_mapping(data, "course")
    return {
        "concept_id": _require_text(data.get("concept_id"), "concept_id"),
        "title": _require_text(data.get("title"), "title"),
        "content": _require_text(
            data.get("content", ""), "content", allow_empty=True
        ),
        "citations": _parse_citations(data.get("citations"), known_passages),
    }


def parse_grade(data: Any) -> dict[str, Any]:
    data = _require_mapping(data, "grade")
    score = data.get("score")
    assistance = data.get("assistance_coefficient", 1.0)
    feedback = data.get("feedback", "")
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError("score must be between 0 and 1")
    if not isinstance(assistance, (int, float)) or not 0 <= assistance <= 1:
        raise ValueError("assistance_coefficient must be between 0 and 1")
    if not isinstance(feedback, str):
        raise ValueError("feedback must be a string")
    return {
        "score": score,
        "assistance_coefficient": assistance,
        "feedback": feedback,
    }


class Parcours:
    """Drives one learner through the learning path.

    The engine keeps the deterministic state (levels, gaps, ordering, progress)
    and persists it. The Generator supplies the non-deterministic content.
    """

    def __init__(
        self,
        store,
        learner_id: str,
        generator: Generator,
        *,
        eta: float = DEFAULT_ETA,
        prerequisite_threshold: float = DEFAULT_PREREQUISITE_THRESHOLD,
    ) -> None:
        if not isinstance(eta, (int, float)) or not 0 <= eta <= 1:
            raise ValueError("eta must be between 0 and 1")
        if not 0 <= prerequisite_threshold <= 1:
            raise ValueError("prerequisite_threshold must be between 0 and 1")
        self.store = store
        self.learner_id = learner_id
        self.generator = generator
        self.eta = eta
        self.prerequisite_threshold = prerequisite_threshold

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        state = self.store.get_parcours_state(self.learner_id)
        if state is None:
            return {
                "phase": "not_started",
                "graph_id": None,
                "initial_gap": None,
                "current_assessment_id": None,
                "current_concept_id": None,
                "objectives": "",
                "preferences": {},
            }
        return state

    def _save_state(self, **fields: Any) -> None:
        state = self._state()
        state.update(fields)
        self.store.save_parcours_state(
            self.learner_id,
            phase=state["phase"],
            graph_id=state["graph_id"],
            initial_gap=state["initial_gap"],
            current_assessment_id=state["current_assessment_id"],
            current_concept_id=state["current_concept_id"],
            objectives=state.get("objectives", ""),
            preferences=state.get("preferences"),
        )

    def _proposal(self, graph_id: str) -> ConceptGraphProposal:
        data = self.store.get_graph(self.learner_id, graph_id)
        if data is None:
            raise ValueError(f"graph {graph_id!r} not found")
        return ConceptGraphProposal.from_mapping(
            {
                "graph_id": data["graph_id"],
                "source_document_ids": data["source_document_ids"],
                "notes": data["notes"],
                "concepts": [
                    {key: value for key, value in concept.items() if key != "graph_id"}
                    for concept in data["concepts"]
                ],
                "edges": [
                    {key: value for key, value in edge.items() if key != "graph_id"}
                    for edge in data["edges"]
                ],
            }
        )

    def _passages(self, graph_id: str) -> list[dict[str, Any]]:
        data = self.store.get_graph(self.learner_id, graph_id)
        passages: list[dict[str, Any]] = []
        for document_id in data["source_document_ids"]:
            document = self.store.get_document(self.learner_id, document_id)
            if document is not None:
                passages.extend(document["passages"])
        return passages

    def _known_passages(self, graph_id: str) -> set[str]:
        return {passage["passage_id"] for passage in self._passages(graph_id)}

    def _concept_mapping(self, concept) -> dict[str, Any]:
        return {
            "concept_id": concept.concept_id,
            "label": concept.label,
            "description": concept.description,
            "target_level": concept.target_level,
            "weight": concept.weight,
        }

    def _mastery_levels(self) -> dict[str, float]:
        return {
            item["concept_id"]: item["current_level"]
            for item in self.store.get_mastery(self.learner_id)
        }

    def _current_gap(self, graph_id: str) -> float:
        proposal = self._proposal(graph_id)
        levels = self._mastery_levels()
        current = {
            concept.concept_id: levels.get(concept.concept_id, 0.0)
            for concept in proposal.concepts
        }
        target = {concept.concept_id: concept.target_level for concept in proposal.concepts}
        weights = {concept.concept_id: concept.weight for concept in proposal.concepts}
        return weighted_gap(current, target, weights)

    def _passage_dicts(self, documents: Sequence[Document]) -> list[dict[str, Any]]:
        passages: list[dict[str, Any]] = []
        for document in documents:
            if not isinstance(document, Document):
                raise ValueError("documents must be Document instances")
            for passage in document.passages:
                passages.append(
                    {
                        "passage_id": passage.passage_id,
                        "document_id": passage.document_id,
                        "page": passage.page,
                        "heading": passage.heading,
                        "text": passage.text,
                    }
                )
        return passages

    def _validate_question_concepts(
        self, questions: list[dict[str, Any]], known_concepts: set[str]
    ) -> None:
        for question in questions:
            if question["concept_id"] not in known_concepts:
                raise ValueError(
                    f"question {question['question_id']!r} references unknown "
                    f"concept {question['concept_id']!r}"
                )

    def _store_questions(self, assessment_id: str, questions: list[dict[str, Any]]) -> None:
        for position, question in enumerate(questions):
            self.store.add_question(
                assessment_id,
                question["question_id"],
                position,
                question["prompt"],
                question["reference_answer"],
                question["grading_rubric"],
                question["concept_id"],
                question["difficulty_coefficient"],
                question["citations"],
            )

    # ------------------------------------------------------------------
    # E3-E4: initial evaluation
    # ------------------------------------------------------------------
    def start(
        self,
        documents: Sequence[Document],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Propose the graph, generate the initial evaluation and persist both."""
        state = self._state()
        if state["phase"] != "not_started":
            raise ValueError("the parcours has already started")
        if not isinstance(objectives, str):
            raise ValueError("objectives must be a string")

        passages = self._passage_dicts(documents)
        for document in documents:
            self.store.save_document(self.learner_id, document)
        known = {passage["passage_id"] for passage in passages}

        graph_mapping = self.generator.propose_graph(passages, objectives, preferences)
        proposal = ConceptGraphProposal.from_mapping(graph_mapping)
        report = proposal.validate(known_passages=known)
        if not report.is_valid:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in report.issues
            )
            raise ValueError(f"proposed graph is invalid: {details}")
        self.store.save_graph(self.learner_id, proposal, status="proposed")

        concepts = [self._concept_mapping(concept) for concept in proposal.concepts]
        assessment_data = self.generator.generate_initial_evaluation(
            concepts, passages, objectives, preferences
        )
        questions = parse_assessment(assessment_data, known_passages=known)
        self._validate_question_concepts(
            questions, {concept.concept_id for concept in proposal.concepts}
        )

        assessment_id = uuid4().hex
        self.store.create_assessment(
            assessment_id,
            self.learner_id,
            kind="evaluation",
            source_metadata={"phase": "initial", "graph_id": proposal.graph_id},
        )
        self._store_questions(assessment_id, questions)
        self._save_state(
            phase="initial_evaluation",
            graph_id=proposal.graph_id,
            current_assessment_id=assessment_id,
            current_concept_id=None,
            objectives=objectives,
            preferences=preferences,
        )
        return {
            "graph_id": proposal.graph_id,
            "assessment_id": assessment_id,
            "questions": questions,
        }

    def submit_initial_evaluation(
        self, responses: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Grade the initial evaluation and compute the initial levels and gap."""
        state = self._state()
        if state["phase"] != "initial_evaluation":
            raise ValueError("the parcours is not in the initial evaluation phase")
        assessment_id = state["current_assessment_id"]
        assessment = self.store.get_assessment(assessment_id) if assessment_id else None
        if assessment is None:
            raise ValueError("initial assessment not found")

        by_question = {question["question_id"]: question for question in assessment["questions"]}
        attempts_by_concept: dict[str, list[Attempt]] = {}
        scores: list[float] = []
        for item in responses:
            item = _require_mapping(item, "response")
            question_id = item.get("question_id")
            response_text = item.get("response_text")
            if question_id not in by_question:
                raise ValueError(f"unknown question {question_id!r}")
            if not isinstance(response_text, str):
                raise ValueError("response_text must be a string")
            question = by_question[question_id]
            grade = parse_grade(
                self.generator.grade_response(question, response_text)
            )
            self.store.record_response(
                uuid4().hex,
                question_id,
                response_text,
                grade["score"],
                grade["assistance_coefficient"],
                feedback=grade["feedback"],
            )
            attempts_by_concept.setdefault(question["concept_id"], []).append(
                Attempt(
                    kind=AttemptType.EVALUATION,
                    score=grade["score"],
                    assistance_coefficient=grade["assistance_coefficient"],
                    difficulty_coefficient=question["difficulty_coefficient"],
                )
            )
            scores.append(grade["score"])

        proposal = self._proposal(state["graph_id"])
        levels: dict[str, float] = {}
        for concept in proposal.concepts:
            level = 0.0
            for attempt in attempts_by_concept.get(concept.concept_id, []):
                level = update_level_from_attempt(level, attempt, eta=1.0)
            levels[concept.concept_id] = level
            self.store.save_mastery(
                self.learner_id,
                concept.concept_id,
                current_level=level,
                target_level=concept.target_level,
                weight=concept.weight,
                concept_label=concept.label,
            )

        self.store.complete_assessment(
            assessment_id, raw_score=mean(scores) if scores else 0.0
        )
        target = {concept.concept_id: concept.target_level for concept in proposal.concepts}
        weights = {concept.concept_id: concept.weight for concept in proposal.concepts}
        initial_gap = weighted_gap(levels, target, weights)
        self._save_state(
            phase="learning",
            initial_gap=initial_gap,
            current_assessment_id=None,
            current_concept_id=None,
        )
        return {"levels": levels, "gap": initial_gap}

    # ------------------------------------------------------------------
    # E5-E7: planning, learning and adaptation
    # ------------------------------------------------------------------
    def _select_concept(self) -> tuple[str, Any, float] | None:
        """Return the next concept to work on, or None when none remains."""
        state = self._state()
        if state["phase"] != "learning":
            raise ValueError("the parcours is not in the learning phase")
        proposal = self._proposal(state["graph_id"])
        concepts = proposal.to_domain_concepts()
        levels = self._mastery_levels()
        current_levels = {
            concept.concept_id: levels.get(concept.concept_id, 0.0)
            for concept in proposal.concepts
        }
        order = plan_concepts(
            concepts, current_levels, prerequisite_threshold=self.prerequisite_threshold
        )
        if not order:
            return None
        concept_id = order[0]
        concept = next(item for item in proposal.concepts if item.concept_id == concept_id)
        return concept_id, concept, current_levels[concept_id]

    def _generate_course(
        self, concept_id: str, concept: Any, current_level: float
    ) -> dict[str, Any]:
        state = self._state()
        passages = self._passages(state["graph_id"])
        known = self._known_passages(state["graph_id"])
        course_data = self.generator.generate_course(
            self._concept_mapping(concept),
            passages,
            current_level,
            state.get("preferences"),
        )
        course = parse_course(course_data, known_passages=known)
        if course["concept_id"] != concept_id:
            raise ValueError("the generated course must target the selected concept")
        return course

    def _generate_exercise(
        self, concept_id: str, concept: Any, current_level: float
    ) -> dict[str, Any]:
        state = self._state()
        passages = self._passages(state["graph_id"])
        known = self._known_passages(state["graph_id"])
        exercise_data = self.generator.generate_exercise(
            self._concept_mapping(concept),
            passages,
            current_level,
            state.get("preferences"),
        )
        question = parse_question(exercise_data, known_passages=known)
        if question["concept_id"] != concept_id:
            raise ValueError("the generated exercise must target the selected concept")
        return question

    def next_lesson(self) -> dict[str, Any]:
        """Generate and store the next mini-course, or signal the final exam."""
        selected = self._select_concept()
        if selected is None:
            return {"type": "final_exam"}
        concept_id, concept, current_level = selected
        course = self._generate_course(concept_id, concept, current_level)
        course_id = uuid4().hex
        self.store.save_course(
            self.learner_id,
            course_id,
            concept_id,
            course["title"],
            course["content"],
            course["citations"],
        )
        self._save_state(
            phase="learning", current_concept_id=concept_id, current_assessment_id=None
        )
        return {
            "type": "lesson",
            "concept_id": concept_id,
            "concept_label": concept.label,
            "course": course,
        }

    def next_exercise(self) -> dict[str, Any]:
        """Generate and store an exercise for the current or next concept."""
        state = self._state()
        if state["phase"] != "learning":
            raise ValueError("the parcours is not in the learning phase")
        concept_id = state["current_concept_id"]
        if concept_id is None:
            selected = self._select_concept()
            if selected is None:
                return {"type": "final_exam"}
            concept_id, concept, current_level = selected
        else:
            proposal = self._proposal(state["graph_id"])
            concept = next(
                (item for item in proposal.concepts if item.concept_id == concept_id), None
            )
            if concept is None:
                raise ValueError("the current lesson concept is not in the graph")
            current_level = self._mastery_levels().get(concept_id, 0.0)
        question = self._generate_exercise(concept_id, concept, current_level)
        assessment_id = uuid4().hex
        self.store.create_assessment(
            assessment_id,
            self.learner_id,
            kind="exercise",
            source_metadata={"phase": "learning", "concept_id": concept_id},
        )
        self._store_questions(assessment_id, [question])
        self._save_state(
            phase="learning",
            current_assessment_id=assessment_id,
            current_concept_id=concept_id,
        )
        return {
            "type": "exercise",
            "concept_id": concept_id,
            "concept_label": concept.label,
            "exercise": question,
            "assessment_id": assessment_id,
        }

    def next_activity(self) -> dict[str, Any]:
        """Return the next course and exercise, or signal the final exam."""
        selected = self._select_concept()
        if selected is None:
            return {"type": "final_exam"}

        concept_id, concept, current_level = selected
        course = self._generate_course(concept_id, concept, current_level)
        question = self._generate_exercise(concept_id, concept, current_level)

        course_id = uuid4().hex
        self.store.save_course(
            self.learner_id,
            course_id,
            concept_id,
            course["title"],
            course["content"],
            course["citations"],
        )
        assessment_id = uuid4().hex
        self.store.create_assessment(
            assessment_id,
            self.learner_id,
            kind="exercise",
            source_metadata={
                "phase": "learning",
                "concept_id": concept_id,
                "course_id": course_id,
            },
        )
        self._store_questions(assessment_id, [question])
        self._save_state(
            phase="learning",
            current_assessment_id=assessment_id,
            current_concept_id=concept_id,
        )
        return {
            "type": "activity",
            "concept_id": concept_id,
            "concept_label": concept.label,
            "course": course,
            "exercise": question,
            "assessment_id": assessment_id,
        }

    def submit_exercise(self, response_text: str) -> dict[str, Any]:
        """Grade an exercise, update the level and record the progress."""
        state = self._state()
        if state["phase"] != "learning" or state["current_assessment_id"] is None:
            raise ValueError("no exercise in progress")
        assessment_id = state["current_assessment_id"]
        assessment = self.store.get_assessment(assessment_id)
        if assessment is None or not assessment["questions"]:
            raise ValueError("exercise assessment not found")
        question = assessment["questions"][0]
        if not isinstance(response_text, str):
            raise ValueError("response_text must be a string")

        grade = parse_grade(self.generator.grade_response(question, response_text))
        self.store.record_response(
            uuid4().hex,
            question["question_id"],
            response_text,
            grade["score"],
            grade["assistance_coefficient"],
            feedback=grade["feedback"],
        )

        concept_id = state["current_concept_id"]
        mastery = {
            item["concept_id"]: item for item in self.store.get_mastery(self.learner_id)
        }
        if concept_id not in mastery:
            raise ValueError("the exercise concept has no mastery record")
        record = mastery[concept_id]
        attempt = Attempt(
            kind=AttemptType.EXERCISE,
            score=grade["score"],
            assistance_coefficient=grade["assistance_coefficient"],
            difficulty_coefficient=question["difficulty_coefficient"],
        )
        new_level = update_level_from_attempt(record["current_level"], attempt, eta=self.eta)
        self.store.save_mastery(
            self.learner_id,
            concept_id,
            current_level=new_level,
            target_level=record["target_level"],
            weight=record["weight"],
            concept_label=record["concept_label"],
        )
        self.store.complete_assessment(assessment_id, raw_score=grade["score"])

        current_gap = self._current_gap(state["graph_id"])
        progress = progression(state["initial_gap"], current_gap)
        self.store.record_progress(
            self.learner_id,
            state["initial_gap"],
            current_gap,
            assessment_id=assessment_id,
        )
        self._save_state(
            phase="learning", current_assessment_id=None, current_concept_id=None
        )
        return {
            "score": grade["score"],
            "assistance_coefficient": grade["assistance_coefficient"],
            "feedback": grade["feedback"],
            "level": new_level,
            "gap": current_gap,
            "progress": progress,
        }

    # ------------------------------------------------------------------
    # E8-E9: final exam and remediation
    # ------------------------------------------------------------------
    def start_final_exam(self) -> dict[str, Any]:
        """Generate and store the final exam."""
        state = self._state()
        if state["phase"] != "learning":
            raise ValueError("the parcours is not in the learning phase")
        proposal = self._proposal(state["graph_id"])
        concepts = [self._concept_mapping(concept) for concept in proposal.concepts]
        passages = self._passages(state["graph_id"])
        known = self._known_passages(state["graph_id"])
        exam_data = self.generator.generate_final_exam(
            concepts, passages, state.get("objectives", ""), state.get("preferences")
        )
        questions = parse_assessment(exam_data, known_passages=known)
        self._validate_question_concepts(
            questions, {concept.concept_id for concept in proposal.concepts}
        )

        assessment_id = uuid4().hex
        self.store.create_assessment(
            assessment_id,
            self.learner_id,
            kind="final_exam",
            source_metadata={"phase": "final", "graph_id": proposal.graph_id},
        )
        self._store_questions(assessment_id, questions)
        self._save_state(phase="final_exam", current_assessment_id=assessment_id)
        return {"assessment_id": assessment_id, "questions": questions}

    def submit_final_exam(
        self, responses: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Grade the final exam, re-evaluate the levels and decide sufficiency."""
        state = self._state()
        if state["phase"] != "final_exam":
            raise ValueError("the parcours is not in the final exam phase")
        assessment_id = state["current_assessment_id"]
        assessment = self.store.get_assessment(assessment_id) if assessment_id else None
        if assessment is None:
            raise ValueError("final exam not found")

        by_question = {question["question_id"]: question for question in assessment["questions"]}
        attempts_by_concept: dict[str, list[Attempt]] = {}
        scores: list[float] = []
        for item in responses:
            item = _require_mapping(item, "response")
            question_id = item.get("question_id")
            response_text = item.get("response_text")
            if question_id not in by_question:
                raise ValueError(f"unknown question {question_id!r}")
            if not isinstance(response_text, str):
                raise ValueError("response_text must be a string")
            question = by_question[question_id]
            grade = parse_grade(
                self.generator.grade_response(question, response_text)
            )
            self.store.record_response(
                uuid4().hex,
                question_id,
                response_text,
                grade["score"],
                grade["assistance_coefficient"],
                feedback=grade["feedback"],
            )
            attempts_by_concept.setdefault(question["concept_id"], []).append(
                Attempt(
                    kind=AttemptType.FINAL_EXAM,
                    score=grade["score"],
                    assistance_coefficient=grade["assistance_coefficient"],
                    difficulty_coefficient=question["difficulty_coefficient"],
                )
            )
            scores.append(grade["score"])

        proposal = self._proposal(state["graph_id"])
        mastery = {
            item["concept_id"]: item for item in self.store.get_mastery(self.learner_id)
        }
        levels: dict[str, float] = {}
        for concept in proposal.concepts:
            record = mastery.get(concept.concept_id)
            level = record["current_level"] if record is not None else 0.0
            for attempt in attempts_by_concept.get(concept.concept_id, []):
                level = update_level_from_attempt(level, attempt, eta=self.eta)
            levels[concept.concept_id] = level
            self.store.save_mastery(
                self.learner_id,
                concept.concept_id,
                current_level=level,
                target_level=concept.target_level,
                weight=concept.weight,
                concept_label=concept.label,
            )

        self.store.complete_assessment(
            assessment_id, raw_score=mean(scores) if scores else 0.0
        )
        current_gap = self._current_gap(state["graph_id"])
        progress = progression(state["initial_gap"], current_gap)
        self.store.record_progress(
            self.learner_id,
            state["initial_gap"],
            current_gap,
            assessment_id=assessment_id,
        )
        sufficient = all(
            levels[concept.concept_id] >= concept.target_level
            for concept in proposal.concepts
        )
        self._save_state(
            phase="completed", current_assessment_id=None, current_concept_id=None
        )
        return {
            "levels": levels,
            "gap": current_gap,
            "progress": progress,
            "sufficient": sufficient,
        }

    def remediate(self) -> dict[str, Any]:
        """Return the concepts that still need work after an insufficient exam."""
        state = self._state()
        if state["phase"] != "completed":
            raise ValueError("the parcours is not completed")
        proposal = self._proposal(state["graph_id"])
        concepts = proposal.to_domain_concepts()
        levels = self._mastery_levels()
        current_levels = {
            concept.concept_id: levels.get(concept.concept_id, 0.0)
            for concept in proposal.concepts
        }
        order = plan_concepts(
            concepts, current_levels, prerequisite_threshold=self.prerequisite_threshold
        )
        return {"concepts": list(order)}

    def resume_learning(self) -> dict[str, Any]:
        """Restart the learning phase after an insufficient final exam."""
        state = self._state()
        if state["phase"] != "completed":
            raise ValueError("the parcours is not completed")
        self._save_state(
            phase="learning", current_assessment_id=None, current_concept_id=None
        )
        return {"type": "learning"}
