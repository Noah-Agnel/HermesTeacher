"""Learning-path orchestration.

The Parcours engine drives the nominal flow: initial evaluation, learning,
adaptation, final exam and remediation. Deterministic decisions (levels, gaps,
ordering, progress, persistence) stay in the backend. The Generator protocol is
the LLM boundary: the agent reads the learner's documents and generates courses,
exercises and evaluations, and grades free-form responses.
"""

from __future__ import annotations

import inspect
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

    The backend stores deposited documents as-is, with a text companion
    converted at upload, and hands the agent their records (id, title, path).
    Implementations decide how the agent gets the content (the web generator
    inlines the text); the agent cites documents by id, optionally with a page
    and a quote.
    """

    @abstractmethod
    def propose_graph(
        self,
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Propose a concept graph from the learner's deposited documents."""

    @abstractmethod
    def generate_initial_evaluation(
        self,
        concepts: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
        feedback: str | None = None,
    ) -> Mapping[str, Any]:
        """Generate the initial evaluation covering the concepts.

        ``feedback`` is set when a previous answer had to be corrected: it
        names the rejected questions and the allowed concept ids.
        """

    @abstractmethod
    def generate_course(
        self,
        concept: dict[str, Any],
        documents: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Generate a mini-course adapted to the concept and level."""

    @abstractmethod
    def generate_exercise(
        self,
        concept: dict[str, Any],
        documents: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Generate an exercise for the concept."""

    @abstractmethod
    def generate_final_exam(
        self,
        concepts: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
        feedback: str | None = None,
    ) -> Mapping[str, Any]:
        """Generate the final exam.

        ``feedback`` is set when a previous answer had to be corrected: it
        names the rejected questions and the allowed concept ids.
        """

    @abstractmethod
    def grade_response(
        self, question: dict[str, Any], response_text: str
    ) -> Mapping[str, Any]:
        """Grade a free-form response and estimate the assistance used."""

    def grade_responses(
        self, items: list[dict[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        """Grade several responses at once: question_id → grade.

        ``items`` holds ``{"question": ..., "response_text": ...}``. The
        default grades one by one; implementations may use a single call.
        Missing question ids are graded one by one by the engine.
        """
        return {
            item["question"]["question_id"]: self.grade_response(
                item["question"], item["response_text"]
            )
            for item in items
        }


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
    value: Any, known_documents: set[str] | None = None
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
        page = item.get("page")
        quote = item.get("quote")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("citation document_id must be a non-empty string")
        if page is not None:
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise ValueError("citation page must be a positive integer or None")
        if quote is not None and not isinstance(quote, str):
            raise ValueError("citation quote must be a string or None")
        if known_documents is not None and document_id not in known_documents:
            raise ValueError(
                f"citation references unknown document {document_id!r}"
            )
        citations.append(
            {"document_id": document_id, "page": page, "quote": quote}
        )
    return citations


def parse_question(
    data: Any, known_documents: set[str] | None = None
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
        "citations": _parse_citations(data.get("citations"), known_documents),
    }


def parse_assessment(
    data: Any, known_documents: set[str] | None = None
) -> list[dict[str, Any]]:
    data = _require_mapping(data, "assessment")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("assessment must contain at least one question")
    return [parse_question(item, known_documents) for item in questions]



MAX_ASSESSMENT_ATTEMPTS = 2
MAX_INITIAL_QUESTIONS = 10

# Grade given without calling the model when the learner ticked "je ne sais
# pas" or left the answer empty: there is nothing to judge, so the backend
# records a zero instead of spending a model call on an empty answer.
DONT_KNOW_FEEDBACK = "Pas de réponse (« je ne sais pas »)."


def _dont_know_grade() -> dict[str, Any]:
    return {"score": 0.0, "assistance_coefficient": 1.0, "feedback": DONT_KNOW_FEEDBACK}


def is_blank_response(item: Mapping[str, Any]) -> bool:
    """True when the learner ticked "je ne sais pas" or wrote nothing."""
    text = item.get("response_text")
    return bool(item.get("dont_know")) or not (isinstance(text, str) and text.strip())


def _call_generator(method: Any, *args: Any, feedback: str | None = None) -> Any:
    """Call a generator method, passing ``feedback`` only if it accepts it."""
    if feedback is not None:
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            parameters = []
        if any(
            parameter.name == "feedback" or parameter.kind is parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            return method(*args, feedback=feedback)
    return method(*args)


def screen_assessment(
    data: Any,
    known_concepts: set[str],
    known_documents: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Split an assessment into valid questions and rejected ones.

    A question is rejected when it is malformed, targets a concept outside the
    graph or repeats a question id. Rejected questions are returned with the
    reason, so one faulty question no longer discards the whole assessment.
    """
    try:
        data = _require_mapping(data, "assessment")
    except ValueError as error:
        return [], [{"question_id": "?", "reason": str(error)}]
    items = data.get("questions")
    if not isinstance(items, list) or not items:
        return [], [{"question_id": "?", "reason": "assessment must contain at least one question"}]
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        question_id = item.get("question_id", "?") if isinstance(item, Mapping) else "?"
        try:
            question = parse_question(item, known_documents)
        except ValueError as error:
            rejected.append({"question_id": str(question_id), "reason": str(error)})
            continue
        if question["concept_id"] not in known_concepts:
            rejected.append(
                {
                    "question_id": question["question_id"],
                    "reason": f"unknown concept {question['concept_id']!r}",
                }
            )
            continue
        if question["question_id"] in seen:
            rejected.append(
                {"question_id": question["question_id"], "reason": "duplicate question_id"}
            )
            continue
        seen.add(question["question_id"])
        valid.append(question)
    return valid, rejected


def _correction_feedback(
    rejected: list[dict[str, str]],
    concept_ids: list[str],
    require_coverage: bool = True,
) -> str:
    lines = [f"- {item['question_id']} : {item['reason']}" for item in rejected]
    text = (
        "Ta réponse précédente contenait des questions invalides :\n"
        + "\n".join(lines)
        + "\nUtilise uniquement ces concept_id : "
        + ", ".join(concept_ids)
        + "."
    )
    if require_coverage:
        text += " Couvre chaque concept par au moins une question."
    return text

def parse_course(
    data: Any, known_documents: set[str] | None = None
) -> dict[str, Any]:
    data = _require_mapping(data, "course")
    return {
        "concept_id": _require_text(data.get("concept_id"), "concept_id"),
        "title": _require_text(data.get("title"), "title"),
        "content": _require_text(
            data.get("content", ""), "content", allow_empty=True
        ),
        "citations": _parse_citations(data.get("citations"), known_documents),
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
        generator: Generator | None = None,
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

    def _grade(
        self,
        question: dict[str, Any],
        response_text: str,
        grade: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Return the parsed grade: the agent's proposal, or the generator's.

        The backend never decides a score itself. An explicit proposal is
        validated as-is; without one, the generator (the LLM boundary) grades
        the response.
        """
        if grade is not None:
            return parse_grade(grade)
        if not response_text.strip():
            return _dont_know_grade()
        if self.generator is None:
            raise ValueError("no generator is available to grade the response")
        return parse_grade(self.generator.grade_response(question, response_text))

    def _grade_responses(
        self,
        by_question: dict[str, Any],
        responses: Sequence[Mapping[str, Any]],
        grades: Mapping[str, Mapping[str, Any]] | None,
    ) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
        """Grade every response before anything is persisted.

        Blank answers ("je ne sais pas" or empty) get a zero without a model
        call. Explicit grades are used as-is. The others are graded in one
        batch call when the generator offers ``grade_responses``, otherwise
        one by one. Returns ``(question, response_text, grade)`` in order.
        """
        items: list[tuple[dict[str, Any], str, bool]] = []
        for item in responses:
            item = _require_mapping(item, "response")
            question_id = item.get("question_id")
            response_text = item.get("response_text", "")
            if question_id not in by_question:
                raise ValueError(f"unknown question {question_id!r}")
            if not isinstance(response_text, str):
                raise ValueError("response_text must be a string")
            blank = is_blank_response(item)
            items.append((by_question[question_id], "" if blank else response_text, blank))

        results: dict[str, dict[str, Any]] = {}
        to_model: list[tuple[dict[str, Any], str]] = []
        for question, response_text, blank in items:
            question_id = question["question_id"]
            if grades is not None and question_id in grades:
                results[question_id] = parse_grade(grades[question_id])
            elif blank:
                results[question_id] = _dont_know_grade()
            else:
                to_model.append((question, response_text))

        if to_model:
            if self.generator is None:
                raise ValueError("no generator is available to grade the response")
            batch = getattr(self.generator, "grade_responses", None)
            proposed: Mapping[str, Any] = {}
            if callable(batch) and len(to_model) > 1:
                proposed = batch(
                    [
                        {"question": question, "response_text": response_text}
                        for question, response_text in to_model
                    ]
                )
            for question, response_text in to_model:
                question_id = question["question_id"]
                grade = None
                if question_id in proposed:
                    try:
                        grade = parse_grade(proposed[question_id])
                    except ValueError:
                        grade = None  # malformed batch grade: grade it alone
                if grade is not None:
                    results[question_id] = grade
                else:
                    results[question_id] = parse_grade(
                        self.generator.grade_response(question, response_text)
                    )
        return [(question, text, results[question["question_id"]]) for question, text, _ in items]

    @staticmethod
    def _check_grades(
        by_question: dict[str, Any],
        grades: Mapping[str, Mapping[str, Any]] | None,
        responses: Sequence[Mapping[str, Any]],
    ) -> None:
        """Validate an explicit grade mapping against the answered questions.

        An explicit mapping must cover every submitted response and may not
        reference a question outside the assessment.
        """
        if grades is None:
            return
        if not isinstance(grades, Mapping):
            raise ValueError("grades must be a mapping of question_id to grade")
        answered = [
            item.get("question_id")
            for item in responses
            if isinstance(item, Mapping)
            and isinstance(item.get("question_id"), str)
            and not is_blank_response(item)
        ]
        unknown = set(grades) - set(by_question)
        if unknown:
            raise ValueError(f"grades reference unknown questions: {sorted(unknown)}")
        missing = set(answered) - set(grades)
        if missing:
            raise ValueError(f"missing grades for questions: {sorted(missing)}")

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

    def _documents(self, graph_id: str) -> list[dict[str, Any]]:
        """Return the deposited documents linked to a graph, as agent inputs."""
        data = self.store.get_graph(self.learner_id, graph_id)
        documents: list[dict[str, Any]] = []
        for document_id in data["source_document_ids"]:
            document = self.store.get_document(self.learner_id, document_id)
            if document is not None:
                documents.append(
                    {
                        "document_id": document["document_id"],
                        "title": document["title"],
                        "path": document["path"],
                        "media_type": document["media_type"],
                    }
                )
        return documents

    def _known_documents(self, graph_id: str) -> set[str]:
        return {document["document_id"] for document in self._documents(graph_id)}

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

    def _document_dicts(self, documents: Sequence[Document]) -> list[dict[str, Any]]:
        """Return the deposited documents as agent inputs (id, title, path)."""
        result: list[dict[str, Any]] = []
        for document in documents:
            if not isinstance(document, Document):
                raise ValueError("documents must be Document instances")
            result.append(
                {
                    "document_id": document.document_id,
                    "title": document.title,
                    "path": document.path,
                    "media_type": document.media_type,
                }
            )
        return result

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
        if state["phase"] != "not_started" or state.get("graph_id"):
            raise ValueError("the parcours has already started")
        if not isinstance(objectives, str):
            raise ValueError("objectives must be a string")

        document_dicts = self._document_dicts(documents)
        for document in documents:
            self.store.save_document(self.learner_id, document)
            self.store.log_event(
                self.learner_id,
                "document_deposited",
                payload={
                    "document_id": document.document_id,
                    "title": document.title,
                },
            )
        known = {document["document_id"] for document in document_dicts}

        graph_mapping = self.generator.propose_graph(document_dicts, objectives, preferences)
        proposal = ConceptGraphProposal.from_mapping(graph_mapping)
        report = proposal.validate(known_documents=known)
        if not report.is_valid:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in report.issues
            )
            raise ValueError(f"proposed graph is invalid: {details}")
        self.store.save_graph(self.learner_id, proposal, status="proposed")
        # The graph is persisted before the evaluation. If the evaluation
        # fails, the parcours stays "not_started" with a graph_id (graph
        # ready, evaluation pending, the same convention as agent_tools) and
        # prepare_initial_evaluation() can be retried without a new graph.
        self._save_state(
            phase="not_started",
            graph_id=proposal.graph_id,
            current_assessment_id=None,
            current_concept_id=None,
            objectives=objectives,
            preferences=preferences,
        )
        return self.prepare_initial_evaluation()

    def prepare_initial_evaluation(self) -> dict[str, Any]:
        """Generate and store the initial evaluation of the persisted graph.

        Called by start(); call it again to retry after a failure. Requires
        a persisted graph and no evaluation yet (phase "not_started" with a
        graph_id).
        """
        state = self._state()
        if state["phase"] != "not_started" or not state.get("graph_id"):
            raise ValueError("the concept graph is not ready for the initial evaluation")
        graph_id = state["graph_id"]
        proposal = self._proposal(graph_id)
        concepts = [self._concept_mapping(concept) for concept in proposal.concepts]
        questions = self._generate_assessment(
            self.generator.generate_initial_evaluation,
            concepts,
            self._documents(graph_id),
            state.get("objectives", ""),
            state.get("preferences"),
            known_documents=self._known_documents(graph_id),
            kind="initial_evaluation",
            require_coverage=False,
            max_questions=MAX_INITIAL_QUESTIONS,
        )

        assessment_id = uuid4().hex
        self.store.create_assessment(
            assessment_id,
            self.learner_id,
            kind="evaluation",
            source_metadata={"phase": "initial", "graph_id": graph_id},
        )
        self._store_questions(assessment_id, questions)
        self._save_state(
            phase="initial_evaluation",
            current_assessment_id=assessment_id,
            current_concept_id=None,
        )
        return {
            "graph_id": graph_id,
            "assessment_id": assessment_id,
            "questions": questions,
        }

    def _generate_assessment(
        self,
        method: Any,
        concepts: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None,
        *,
        known_documents: set[str],
        kind: str,
        require_coverage: bool = True,
        max_questions: int | None = None,
    ) -> list[dict[str, Any]]:
        """Ask the generator for an assessment, keeping only valid questions.

        Faulty questions (unknown concept, malformed, duplicated) are dropped.
        The answer is accepted when nothing was dropped, or when the remaining
        questions still cover every concept (``require_coverage``, final exam)
        or when at most a third of the questions were dropped (initial
        evaluation, which only targets the general notions). Otherwise the
        generator is asked once more with the list of errors; the better of
        the two answers is kept. It fails only when no valid question remains.
        ``max_questions`` caps the number of kept questions.
        """
        concept_ids = [concept["concept_id"] for concept in concepts]
        known_concepts = set(concept_ids)
        candidates: list[tuple[list[dict[str, Any]], list[dict[str, str]]]] = []
        feedback: str | None = None
        for _ in range(MAX_ASSESSMENT_ATTEMPTS):
            data = _call_generator(
                method, concepts, documents, objectives, preferences, feedback=feedback
            )
            valid, rejected = screen_assessment(data, known_concepts, known_documents)
            candidates.append((valid, rejected))
            if require_coverage:
                covered = {question["concept_id"] for question in valid}
                acceptable = bool(valid) and covered >= known_concepts
            else:
                # at most a third of the questions dropped
                acceptable = bool(valid) and 3 * len(rejected) <= len(valid) + len(rejected)
            if not rejected or acceptable:
                break
            feedback = _correction_feedback(rejected, concept_ids, require_coverage)

        def quality(candidate):
            valid, _ = candidate
            return (len({question["concept_id"] for question in valid}), len(valid))

        # max() keeps the first best; prefer the latest answer on ties.
        valid, rejected = max(reversed(candidates), key=quality)
        if not valid:
            details = "; ".join(
                f"{item['question_id']}: {item['reason']}" for item in rejected
            )
            raise ValueError(f"no valid question in the generated {kind}: {details}")
        if max_questions is not None and len(valid) > max_questions:
            rejected = rejected + [
                {"question_id": question["question_id"], "reason": "too many questions"}
                for question in valid[max_questions:]
            ]
            valid = valid[:max_questions]
        if rejected:
            self.store.log_event(
                self.learner_id,
                "questions_discarded",
                payload={"assessment": kind, "rejected": rejected},
            )
        return valid

    def submit_initial_evaluation(
        self,
        responses: Sequence[Mapping[str, Any]],
        grades: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Grade the initial evaluation and compute the initial levels and gap.

        ``grades`` maps a question id to the agent's proposal (score,
        assistance_coefficient, feedback). When it is given it must cover
        every submitted response, and the generator is not called.
        """
        state = self._state()
        if state["phase"] != "initial_evaluation":
            raise ValueError("the parcours is not in the initial evaluation phase")
        assessment_id = state["current_assessment_id"]
        assessment = self.store.get_assessment(assessment_id) if assessment_id else None
        if assessment is None:
            raise ValueError("initial assessment not found")

        by_question = {question["question_id"]: question for question in assessment["questions"]}
        self._check_grades(by_question, grades, responses)
        attempts_by_concept: dict[str, list[Attempt]] = {}
        scores: list[float] = []
        for question, response_text, grade in self._grade_responses(
            by_question, responses, grades
        ):
            self.store.record_response(
                uuid4().hex,
                question["question_id"],
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
        self.store.log_event(
            self.learner_id,
            "evaluation_submitted",
            payload={
                "assessment_id": assessment_id,
                "raw_score": mean(scores) if scores else 0.0,
            },
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
        documents = self._documents(state["graph_id"])
        known = self._known_documents(state["graph_id"])
        course_data = self.generator.generate_course(
            self._concept_mapping(concept),
            documents,
            current_level,
            state.get("preferences"),
        )
        course = parse_course(course_data, known_documents=known)
        if course["concept_id"] != concept_id:
            raise ValueError("the generated course must target the selected concept")
        return course

    def _generate_exercise(
        self, concept_id: str, concept: Any, current_level: float
    ) -> dict[str, Any]:
        state = self._state()
        documents = self._documents(state["graph_id"])
        known = self._known_documents(state["graph_id"])
        exercise_data = self.generator.generate_exercise(
            self._concept_mapping(concept),
            documents,
            current_level,
            state.get("preferences"),
        )
        question = parse_question(exercise_data, known_documents=known)
        if question["concept_id"] != concept_id:
            raise ValueError("the generated exercise must target the selected concept")
        return question

    def _discard_unsaved_courses(self, concept_id: str) -> list[str]:
        """Delete the generated courses the learner did not keep.

        A course is considered read once the learner moves on to the next
        exercise, unless it was explicitly saved.
        """
        discarded: list[str] = []
        for course in self.store.list_courses(self.learner_id, concept_id=concept_id):
            if not course.get("saved", False):
                self.store.delete_course(self.learner_id, course["course_id"])
                discarded.append(course["course_id"])
                self.store.log_event(
                    self.learner_id,
                    "course_read",
                    concept_id=concept_id,
                    payload={"course_id": course["course_id"], "title": course["title"]},
                )
        return discarded

    def keep_course(self, course_id: str) -> None:
        """Mark a generated course as kept by the learner."""
        state = self._state()
        if state["phase"] != "learning":
            raise ValueError("the parcours is not in the learning phase")
        self.store.set_course_saved(self.learner_id, course_id, True)
        self.store.log_event(
            self.learner_id,
            "course_saved",
            concept_id=state.get("current_concept_id"),
            payload={"course_id": course_id},
        )

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
        self.store.log_event(
            self.learner_id,
            "course_generated",
            concept_id=concept_id,
            payload={"course_id": course_id, "title": course["title"]},
        )
        self._save_state(
            phase="learning", current_concept_id=concept_id, current_assessment_id=None
        )
        return {
            "type": "lesson",
            "concept_id": concept_id,
            "concept_label": concept.label,
            "course": course,
            "course_id": course_id,
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
        self._discard_unsaved_courses(concept_id)
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
        self.store.log_event(
            self.learner_id,
            "course_generated",
            concept_id=concept_id,
            payload={"course_id": course_id, "title": course["title"]},
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

    def submit_exercise(
        self,
        response_text: str,
        grade: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Grade an exercise, update the level and record the progress.

        ``grade`` is the agent's proposal (score, assistance_coefficient,
        feedback). When it is omitted, the generator grades the response.
        """
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

        parsed = self._grade(question, response_text, grade)
        self.store.record_response(
            uuid4().hex,
            question["question_id"],
            response_text,
            parsed["score"],
            parsed["assistance_coefficient"],
            feedback=parsed["feedback"],
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
            score=parsed["score"],
            assistance_coefficient=parsed["assistance_coefficient"],
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
        self.store.complete_assessment(assessment_id, raw_score=parsed["score"])
        self.store.log_event(
            self.learner_id,
            "exercise_submitted",
            concept_id=concept_id,
            payload={
                "question_id": question["question_id"],
                "score": parsed["score"],
                "assistance_coefficient": parsed["assistance_coefficient"],
            },
        )

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
            "score": parsed["score"],
            "assistance_coefficient": parsed["assistance_coefficient"],
            "feedback": parsed["feedback"],
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
        documents = self._documents(state["graph_id"])
        known = self._known_documents(state["graph_id"])
        questions = self._generate_assessment(
            self.generator.generate_final_exam,
            concepts,
            documents,
            state.get("objectives", ""),
            state.get("preferences"),
            known_documents=known,
            kind="final_exam",
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
        self,
        responses: Sequence[Mapping[str, Any]],
        grades: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Grade the final exam, re-evaluate the levels and decide sufficiency.

        ``grades`` maps a question id to the agent's proposal (score,
        assistance_coefficient, feedback). When it is given it must cover
        every submitted response, and the generator is not called.
        """
        state = self._state()
        if state["phase"] != "final_exam":
            raise ValueError("the parcours is not in the final exam phase")
        assessment_id = state["current_assessment_id"]
        assessment = self.store.get_assessment(assessment_id) if assessment_id else None
        if assessment is None:
            raise ValueError("final exam not found")

        by_question = {question["question_id"]: question for question in assessment["questions"]}
        self._check_grades(by_question, grades, responses)
        attempts_by_concept: dict[str, list[Attempt]] = {}
        scores: list[float] = []
        for question, response_text, grade in self._grade_responses(
            by_question, responses, grades
        ):
            self.store.record_response(
                uuid4().hex,
                question["question_id"],
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
        self.store.log_event(
            self.learner_id,
            "final_exam_submitted",
            payload={
                "assessment_id": assessment_id,
                "raw_score": mean(scores) if scores else 0.0,
            },
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
