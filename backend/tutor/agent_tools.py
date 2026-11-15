"""Command-line tools for the Hermes tutor agent.

Hermes calls these tools with ``python -m backend.tutor.agent_tools <command>``.
Each command prints exactly one JSON object on stdout:

- ``{"ok": true, "data": ...}`` on success;
- ``{"ok": false, "error": "..."}`` on failure.

Nothing else is printed, so the agent can parse the output directly. The agent
reads the learner's state (documents, graph, mastery, history, errors, profile)
and proposes structured content; the backend validates every proposal and
persists it. Reading, extracting and judging stay on the agent's side.

Run from the repository root:

    python -m backend.tutor.agent_tools concepts
    python -m backend.tutor.agent_tools graph <concept_id>
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from backend.tutor.domain import Concept, plan_concepts
from backend.tutor.graph import ConceptGraphProposal
from backend.tutor.orchestration import (
    Parcours,
    parse_assessment,
    parse_course,
    parse_question,
)
from backend.tutor.storage import SQLiteStore
from backend.web.profile import ProfileStore


LEARNER_ID = "student"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    """Resolve the data directory, relative to the repository when not absolute."""
    raw = Path(os.environ.get("TUTOR_DATA_DIR", "data")).expanduser()
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw


def _profile_path() -> Path:
    return _data_dir() / "profile.sqlite3"


def _concept(concept_id: str) -> dict[str, Any]:
    """Return the registry row of one concept."""
    concept = ProfileStore(_profile_path()).get_concept(concept_id)
    if concept is None:
        raise ValueError(f"concept inconnu : {concept_id!r}")
    return concept


def _concept_store(concept_id: str) -> SQLiteStore:
    """Open the SQLite database of one concept, scoped to the learner."""
    return SQLiteStore(_concept(concept_id)["db_path"])


def _phase(concept: dict[str, Any]) -> str:
    state = SQLiteStore(concept["db_path"]).get_parcours_state(LEARNER_ID)
    return state["phase"] if state else "not_started"


def _list_concepts() -> list[dict[str, Any]]:
    profile = ProfileStore(_profile_path())
    return [
        {
            "concept_id": concept["concept_id"],
            "title": concept["title"],
            "objectives": concept["objectives"],
            "phase": _phase(concept),
        }
        for concept in profile.list_concepts()
    ]


def _documents(concept_id: str) -> list[dict[str, Any]]:
    return _concept_store(concept_id).list_documents(LEARNER_ID)


def _graph(concept_id: str) -> dict[str, Any]:
    store = _concept_store(concept_id)
    state = store.get_parcours_state(LEARNER_ID) or {}
    graph_id = state.get("graph_id")
    graph = store.get_graph(LEARNER_ID, graph_id) if graph_id else None
    return {
        "concept_id": concept_id,
        "phase": state.get("phase", "not_started"),
        "graph": graph,
    }


def _mastery(concept_id: str) -> list[dict[str, Any]]:
    return _concept_store(concept_id).get_mastery(LEARNER_ID)


def _history(concept_id: str) -> list[dict[str, Any]]:
    return _concept_store(concept_id).list_history(LEARNER_ID)


def _errors(concept_id: str) -> list[dict[str, Any]]:
    return _concept_store(concept_id).list_errors(LEARNER_ID)


def _activity(concept_id: str) -> dict[str, Any]:
    """Return what the learner must answer now, with the question ids.

    The agent needs the question ids and the rubric to grade a response; the
    backend keeps the ordering and the state.
    """
    store = _concept_store(concept_id)
    state = store.get_parcours_state(LEARNER_ID)
    if state is None:
        return {"phase": "not_started", "kind": "not_started", "questions": []}
    assessment_id = state.get("current_assessment_id")
    assessment = store.get_assessment(assessment_id) if assessment_id else None
    return {
        "phase": state["phase"],
        "kind": state["phase"],
        "concept_id": state.get("current_concept_id"),
        "questions": assessment["questions"] if assessment else [],
    }


def _profile() -> dict[str, Any]:
    profile = ProfileStore(_profile_path()).get_learner_profile(LEARNER_ID)
    if profile is None:
        return {
            "learner_id": LEARNER_ID,
            "display_name": None,
            "preferences": {},
            "weaknesses": [],
        }
    return profile


def _save_profile(path: str) -> dict[str, Any]:
    """Validate and store the global profile written by the agent.

    The agent writes a JSON file with optional ``display_name``,
    ``preferences`` and ``weaknesses``; the backend merges them into the
    existing profile and persists the result. The profile is the source of
    truth for the learner's preferences and transversal weaknesses, read by
    the agent with the ``profile`` command.
    """
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")
    display_name = payload.get("display_name")
    if display_name is not None and not isinstance(display_name, str):
        raise ValueError("display_name must be a string or null")
    preferences = payload.get("preferences")
    if preferences is not None and not isinstance(preferences, Mapping):
        raise ValueError("preferences must be a mapping")
    weaknesses = payload.get("weaknesses")
    if weaknesses is not None and not isinstance(weaknesses, list):
        raise ValueError("weaknesses must be a list")

    profile = ProfileStore(_profile_path())
    existing = profile.get_learner_profile(LEARNER_ID) or {}
    merged = {
        "display_name": (
            display_name if display_name is not None
            else existing.get("display_name")
        ),
        "preferences": dict(preferences) if preferences is not None
        else existing.get("preferences", {}),
        "weaknesses": list(weaknesses) if weaknesses is not None
        else existing.get("weaknesses", []),
    }
    profile.save_learner_profile(LEARNER_ID, **merged)
    return merged


def _save_graph(concept_id: str, path: str) -> dict[str, Any]:
    """Validate and store a graph proposal written by the agent.

    The agent writes the proposal as JSON in ``data/agent/`` and calls this
    command. The backend validates it (DAG, known documents, known concepts);
    when it is invalid the command fails and returns the issues so the agent
    can correct its proposal.
    """
    store = _concept_store(concept_id)
    known = {document["document_id"] for document in store.list_documents(LEARNER_ID)}
    proposal = ConceptGraphProposal.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    report = proposal.validate(known_documents=known)
    if not report.is_valid:
        issues = [
            {
                "code": issue.code,
                "message": issue.message,
                "concept_id": issue.concept_id,
                "prerequisite_id": issue.prerequisite_id,
            }
            for issue in report.issues
        ]
        raise ValueError(
            "graphe invalide : " + json.dumps(issues, ensure_ascii=False)
        )
    store.save_graph(LEARNER_ID, proposal, status="proposed")
    state = store.get_parcours_state(LEARNER_ID) or {}
    store.save_parcours_state(
        LEARNER_ID,
        # Proposing a graph does not start the evaluation: without a state the
        # parcours stays not_started until the evaluation is stored.
        phase=state.get("phase", "not_started"),
        graph_id=proposal.graph_id,
        initial_gap=state.get("initial_gap"),
        current_assessment_id=state.get("current_assessment_id"),
        current_concept_id=state.get("current_concept_id"),
        objectives=state.get("objectives", ""),
        preferences=state.get("preferences"),
    )
    return {
        "graph_id": proposal.graph_id,
        "concepts": len(proposal.concepts),
        "edges": len(proposal.edges),
    }


# ------------------------------------------------------------------
# content proposed by the agent
# ------------------------------------------------------------------
def _read_json(path: str) -> Any:
    """Read the JSON file written by the agent, with a readable error."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"fichier illisible : {path!r} ({error})") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON invalide dans {path!r} : {error}") from error


def _load_graph(concept_id: str) -> tuple[SQLiteStore, dict[str, Any], dict[str, Any]]:
    """Open the concept store and return its state and its stored graph."""
    store = _concept_store(concept_id)
    state = store.get_parcours_state(LEARNER_ID) or {}
    graph_id = state.get("graph_id")
    graph = store.get_graph(LEARNER_ID, graph_id) if graph_id else None
    if graph is None:
        raise ValueError(
            f"aucun graphe enregistré pour {concept_id!r} : "
            "appelez d'abord save_graph"
        )
    return store, state, graph


def _known_concepts(graph: dict[str, Any]) -> set[str]:
    return {concept["concept_id"] for concept in graph["concepts"]}


def _known_documents(store: SQLiteStore, graph: dict[str, Any]) -> set[str]:
    """Return the deposited documents the graph was built from."""
    known: set[str] = set()
    for document_id in graph["source_document_ids"]:
        if store.get_document(LEARNER_ID, document_id) is not None:
            known.add(document_id)
    return known


def _domain_concepts(graph: dict[str, Any]) -> dict[str, Concept]:
    prerequisites = {concept["concept_id"]: set() for concept in graph["concepts"]}
    for edge in graph["edges"]:
        if edge["prerequisite_id"] in prerequisites and edge["concept_id"] in prerequisites:
            prerequisites[edge["concept_id"]].add(edge["prerequisite_id"])
    return {
        concept["concept_id"]: Concept(
            concept["concept_id"],
            prerequisites=frozenset(prerequisites[concept["concept_id"]]),
            target_level=concept["target_level"],
            weight=concept["weight"],
        )
        for concept in graph["concepts"]
    }


def _allowed_concepts(
    store: SQLiteStore, graph: dict[str, Any], theta: float, state: dict[str, Any]
) -> set[str]:
    """Return the concepts the agent may write content for.

    Ordering is the backend's decision: the next concept in the plan, plus the
    concept already in progress when there is one.
    """
    levels = {concept_id: 0.0 for concept_id in _known_concepts(graph)}
    for item in store.get_mastery(LEARNER_ID):
        if item["concept_id"] in levels:
            levels[item["concept_id"]] = item["current_level"]
    order = plan_concepts(_domain_concepts(graph), levels, prerequisite_threshold=theta)
    allowed = set(order[:1])
    current = state.get("current_concept_id")
    if current in levels:
        allowed.add(current)
    return allowed


def _check_concept(
    store: SQLiteStore,
    graph: dict[str, Any],
    theta: float,
    state: dict[str, Any],
    concept_id: str,
) -> None:
    """Reject a concept outside the graph or not planned for now."""
    if concept_id not in _known_concepts(graph):
        raise ValueError(f"concept {concept_id!r} hors du graphe")
    allowed = _allowed_concepts(store, graph, theta, state)
    if concept_id not in allowed:
        expected = ", ".join(sorted(allowed)) or "aucun : tous les concepts sont à leur cible"
        raise ValueError(
            f"concept {concept_id!r} pas prévu maintenant ; le backend attend : {expected}"
        )


def _check_phase(state: dict[str, Any], expected: str) -> None:
    phase = state.get("phase", "not_started")
    if phase != expected:
        raise ValueError(f"phase attendue {expected!r}, parcours en {phase!r}")


def _save_state(store: SQLiteStore, **fields: Any) -> None:
    state = store.get_parcours_state(LEARNER_ID) or {}
    state.update(fields)
    store.save_parcours_state(
        LEARNER_ID,
        phase=state.get("phase", "not_started"),
        graph_id=state.get("graph_id"),
        initial_gap=state.get("initial_gap"),
        current_assessment_id=state.get("current_assessment_id"),
        current_concept_id=state.get("current_concept_id"),
        objectives=state.get("objectives", ""),
        preferences=state.get("preferences"),
    )


def _discard_unsaved_courses(store: SQLiteStore, concept_id: str) -> list[str]:
    """Delete the courses of a concept the learner did not keep.

    Same rule as ``Parcours.next_exercise``: a course is read once the learner
    moves on to the next exercise, unless it was explicitly saved.
    """
    discarded: list[str] = []
    for course in store.list_courses(LEARNER_ID, concept_id=concept_id):
        if not course.get("saved", False):
            store.delete_course(LEARNER_ID, course["course_id"])
            store.log_event(
                LEARNER_ID,
                "course_read",
                concept_id=concept_id,
                payload={"course_id": course["course_id"], "title": course["title"]},
            )
            discarded.append(course["course_id"])
    return discarded


def _store_assessment(
    store: SQLiteStore,
    graph: dict[str, Any],
    *,
    kind: str,
    phase: str,
    metadata: dict[str, Any],
    data: Any,
) -> dict[str, Any]:
    """Validate and persist an assessment proposed by the agent."""
    questions = parse_assessment(data, known_documents=_known_documents(store, graph))
    known = _known_concepts(graph)
    for question in questions:
        if question["concept_id"] not in known:
            raise ValueError(
                f"question {question['question_id']!r} vise le concept inconnu "
                f"{question['concept_id']!r}"
            )
    assessment_id = uuid4().hex
    store.create_assessment(
        assessment_id, LEARNER_ID, kind=kind, source_metadata=metadata
    )
    for position, question in enumerate(questions):
        store.add_question(
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
    store.log_event(
        LEARNER_ID,
        f"{kind}_generated",
        payload={"assessment_id": assessment_id, "questions": len(questions)},
    )
    _save_state(store, phase=phase, current_assessment_id=assessment_id, current_concept_id=None)
    return {
        "assessment_id": assessment_id,
        "questions": len(questions),
        "question_ids": [question["question_id"] for question in questions],
    }


def _save_course(concept_id: str, path: str) -> dict[str, Any]:
    """Validate and store a mini-course written by the agent."""
    store, state, graph = _load_graph(concept_id)
    _check_phase(state, "learning")
    course = parse_course(_read_json(path), known_documents=_known_documents(store, graph))
    _check_concept(store, graph, _concept(concept_id)["theta"], state, course["concept_id"])
    course_id = uuid4().hex
    store.save_course(
        LEARNER_ID,
        course_id,
        course["concept_id"],
        course["title"],
        course["content"],
        course["citations"],
    )
    store.log_event(
        LEARNER_ID,
        "course_generated",
        concept_id=course["concept_id"],
        payload={"course_id": course_id, "title": course["title"]},
    )
    _save_state(
        store,
        phase="learning",
        current_concept_id=course["concept_id"],
        current_assessment_id=None,
    )
    return {
        "course_id": course_id,
        "concept_id": course["concept_id"],
        "title": course["title"],
    }


def _save_exercise(concept_id: str, path: str) -> dict[str, Any]:
    """Validate and store an exercise written by the agent."""
    store, state, graph = _load_graph(concept_id)
    _check_phase(state, "learning")
    question = parse_question(
        _read_json(path), known_documents=_known_documents(store, graph)
    )
    _check_concept(store, graph, _concept(concept_id)["theta"], state, question["concept_id"])
    _discard_unsaved_courses(store, question["concept_id"])
    assessment_id = uuid4().hex
    store.create_assessment(
        assessment_id,
        LEARNER_ID,
        kind="exercise",
        source_metadata={"phase": "learning", "concept_id": question["concept_id"]},
    )
    store.add_question(
        assessment_id,
        question["question_id"],
        0,
        question["prompt"],
        question["reference_answer"],
        question["grading_rubric"],
        question["concept_id"],
        question["difficulty_coefficient"],
        question["citations"],
    )
    store.log_event(
        LEARNER_ID,
        "exercise_generated",
        concept_id=question["concept_id"],
        payload={"assessment_id": assessment_id, "question_id": question["question_id"]},
    )
    _save_state(
        store,
        phase="learning",
        current_assessment_id=assessment_id,
        current_concept_id=question["concept_id"],
    )
    return {
        "assessment_id": assessment_id,
        "question_id": question["question_id"],
        "concept_id": question["concept_id"],
    }


def _save_evaluation(concept_id: str, path: str) -> dict[str, Any]:
    """Validate and store the initial evaluation written by the agent."""
    store, state, graph = _load_graph(concept_id)
    _check_phase(state, "not_started")
    return _store_assessment(
        store,
        graph,
        kind="evaluation",
        phase="initial_evaluation",
        metadata={"phase": "initial", "graph_id": graph["graph_id"]},
        data=_read_json(path),
    )


def _save_final_exam(concept_id: str, path: str) -> dict[str, Any]:
    """Validate and store the final exam written by the agent."""
    store, state, graph = _load_graph(concept_id)
    _check_phase(state, "learning")
    return _store_assessment(
        store,
        graph,
        kind="final_exam",
        phase="final_exam",
        metadata={"phase": "final", "graph_id": graph["graph_id"]},
        data=_read_json(path),
    )


def _submit_response(concept_id: str, path: str) -> dict[str, Any]:
    """Record the grades the agent proposes for the assessment in progress.

    The agent writes a JSON proposal under ``data/agent/`` and calls this
    command. The backend validates the grades, records the responses, updates
    the levels and returns the new state. The agent never decides a level.
    """
    store = _concept_store(concept_id)
    state = store.get_parcours_state(LEARNER_ID)
    if state is None:
        raise ValueError("parcours non démarré")
    phase = state["phase"]
    data = _read_json(path)
    if not isinstance(data, Mapping):
        raise ValueError("payload must be an object")
    items = data.get("responses")
    if not isinstance(items, list) or not items:
        raise ValueError("responses must be a non-empty list")
    responses: list[dict[str, Any]] = []
    grades: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("each response must be an object")
        question_id = item.get("question_id")
        response_text = item.get("response_text")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question_id must be a non-empty string")
        if not isinstance(response_text, str):
            raise ValueError("response_text must be a string")
        responses.append(
            {"question_id": question_id, "response_text": response_text}
        )
        grades[question_id] = {
            "score": item.get("score"),
            "assistance_coefficient": item.get("assistance_coefficient", 1.0),
            "feedback": item.get("feedback", ""),
        }

    concept = _concept(concept_id)
    parcours = Parcours(
        store,
        LEARNER_ID,
        eta=concept["eta"],
        prerequisite_threshold=concept["theta"],
    )
    if phase == "initial_evaluation":
        return parcours.submit_initial_evaluation(responses, grades)
    if phase == "learning":
        assessment_id = state.get("current_assessment_id")
        assessment = store.get_assessment(assessment_id) if assessment_id else None
        questions = assessment["questions"] if assessment else []
        if len(questions) != 1 or len(responses) != 1:
            raise ValueError("un exercice comporte une seule question")
        if responses[0]["question_id"] != questions[0]["question_id"]:
            raise ValueError(f"question attendue : {questions[0]['question_id']!r}")
        return parcours.submit_exercise(
            responses[0]["response_text"], grades[responses[0]["question_id"]]
        )
    if phase == "final_exam":
        return parcours.submit_final_exam(responses, grades)
    raise ValueError(f"phase {phase!r} : aucune évaluation à corriger")


COMMANDS = {
    "concepts": _list_concepts,
    "documents": _documents,
    "graph": _graph,
    "mastery": _mastery,
    "history": _history,
    "errors": _errors,
    "activity": _activity,
    "profile": _profile,
    "save_profile": _save_profile,
    "save_graph": _save_graph,
    "save_course": _save_course,
    "save_exercise": _save_exercise,
    "save_evaluation": _save_evaluation,
    "save_final_exam": _save_final_exam,
    "submit_response": _submit_response,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backend.tutor.agent_tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("concepts")
    subparsers.add_parser("profile")
    subparsers.add_parser("save_profile").add_argument("path")
    for name in ("documents", "graph", "mastery", "history", "errors", "activity"):
        command = subparsers.add_parser(name)
        command.add_argument("concept_id")
    for name in (
        "save_graph",
        "save_course",
        "save_exercise",
        "save_evaluation",
        "save_final_exam",
        "submit_response",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("concept_id")
        command.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = COMMANDS[args.command]
    try:
        if hasattr(args, "concept_id") and hasattr(args, "path"):
            data = command(args.concept_id, args.path)
        elif hasattr(args, "concept_id"):
            data = command(args.concept_id)
        elif hasattr(args, "path"):
            data = command(args.path)
        else:
            data = command()
    except Exception as error:  # the agent only sees the message
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "data": data}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
