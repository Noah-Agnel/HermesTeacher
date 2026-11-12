"""LLM adapter for the orchestration engine.

The backend does not decide course content, exercises or grades. This module
turns each non-deterministic task into a prompt and parses the model's JSON
response. The caller supplies the ``complete`` callable, so any model backend
(local agent, API, test double) can be plugged in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Callable

from .orchestration import Generator

Completer = Callable[[str], str]

_GRAPH_SCHEMA = """{
  "graph_id": "string",
  "source_document_ids": ["string"],
  "notes": "string",
  "concepts": [
    {
      "concept_id": "string",
      "label": "string",
      "description": "string",
      "target_level": 1.0,
      "weight": 1.0,
      "evidence": [{"document_id": "string", "passage_id": "string", "quote": "string"}]
    }
  ],
  "edges": [
    {"prerequisite_id": "string", "concept_id": "string", "evidence": []}
  ]
}"""

_ASSESSMENT_SCHEMA = """{
  "questions": [
    {
      "question_id": "string",
      "concept_id": "string",
      "prompt": "string",
      "reference_answer": "string",
      "grading_rubric": "string",
      "difficulty_coefficient": 1.0,
      "citations": [{"document_id": "string", "passage_id": "string", "quote": "string"}]
    }
  ]
}"""

_COURSE_SCHEMA = """{
  "concept_id": "string",
  "title": "string",
  "content": "string",
  "citations": [{"document_id": "string", "passage_id": "string", "quote": "string"}]
}"""

_EXERCISE_SCHEMA = """{
  "question_id": "string",
  "concept_id": "string",
  "prompt": "string",
  "reference_answer": "string",
  "grading_rubric": "string",
  "difficulty_coefficient": 1.0,
  "citations": [{"document_id": "string", "passage_id": "string", "quote": "string"}]
}"""

_GRADE_SCHEMA = """{
  "score": 0.0,
  "assistance_coefficient": 1.0,
  "feedback": "string"
}"""


def extract_json(text: Any) -> Any:
    """Return the first JSON value found in a model response.

    Handles markdown code fences and prose around the JSON. Raises ValueError
    when no valid JSON value is present.
    """
    if not isinstance(text, str):
        raise ValueError("the model must return text")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("no valid JSON found in the model response")


def _format_passages(passages: list[dict[str, Any]]) -> str:
    lines = []
    for passage in passages:
        location = passage.get("passage_id", "?")
        if passage.get("page") is not None:
            location += f" (page {passage['page']})"
        if passage.get("heading"):
            location += f" — « {passage['heading']} »"
        lines.append(f"[{location}] {passage.get('text', '')}")
    return "\n".join(lines) if lines else "(aucun passage)"


def _format_concepts(concepts: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {concept.get('concept_id')} — {concept.get('label')} "
        f"(cible {concept.get('target_level')}, poids {concept.get('weight')})"
        for concept in concepts
    )


def _format_preferences(preferences: Mapping[str, Any] | None) -> str:
    if not preferences:
        return "(non déclarées)"
    return ", ".join(f"{key}={value}" for key, value in preferences.items())


class PromptLLMGenerator(Generator):
    """Generator that delegates every non-deterministic task to an LLM."""

    def __init__(self, complete: Completer) -> None:
        if not callable(complete):
            raise ValueError("complete must be callable")
        self.complete = complete

    def _complete(self, prompt: str) -> Any:
        return extract_json(self.complete(prompt))

    def propose_graph(
        self,
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es un tuteur. À partir des passages ci-dessous, propose un graphe "
            "de concepts et leurs relations de prérequis.\n\n"
            f"OBJECTIFS : {objectives}\n"
            f"PRÉFÉRENCES : {_format_preferences(preferences)}\n\n"
            f"PASSAGES :\n{_format_passages(passages)}\n\n"
            "Le graphe doit être acyclique. Cite les passages qui justifient "
            "chaque concept et chaque relation.\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_GRAPH_SCHEMA}"
        )
        return self._complete(prompt)

    def generate_initial_evaluation(
        self,
        concepts: list[dict[str, Any]],
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es un tuteur. Génère une évaluation initiale couvrant les "
            "concepts ci-dessous, en t'appuyant sur les passages fournis.\n\n"
            f"CONCEPTS :\n{_format_concepts(concepts)}\n\n"
            f"OBJECTIFS : {objectives}\n"
            f"PRÉFÉRENCES : {_format_preferences(preferences)}\n\n"
            f"PASSAGES :\n{_format_passages(passages)}\n\n"
            "Chaque question doit citer les passages utilisés.\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_ASSESSMENT_SCHEMA}"
        )
        return self._complete(prompt)

    def generate_course(
        self,
        concept: dict[str, Any],
        passages: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es un tuteur. Écris un mini-cours structuré et adapté au niveau "
            "de l'apprenant pour le concept ci-dessous.\n\n"
            f"CONCEPT : {concept.get('concept_id')} — {concept.get('label')}\n"
            f"NIVEAU ACTUEL : {current_level}\n"
            f"PRÉFÉRENCES : {_format_preferences(preferences)}\n\n"
            f"PASSAGES :\n{_format_passages(passages)}\n\n"
            "Le cours doit s'appuyer sur les passages et citer les extraits "
            "utilisés. Un cours lu ne prouve pas la maîtrise : il prépare "
            "seulement l'exercice.\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_COURSE_SCHEMA}"
        )
        return self._complete(prompt)

    def generate_exercise(
        self,
        concept: dict[str, Any],
        passages: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es un tuteur. Génère un exercice pour le concept ci-dessous, "
            "adapté au niveau de l'apprenant.\n\n"
            f"CONCEPT : {concept.get('concept_id')} — {concept.get('label')}\n"
            f"NIVEAU ACTUEL : {current_level}\n"
            f"PRÉFÉRENCES : {_format_preferences(preferences)}\n\n"
            f"PASSAGES :\n{_format_passages(passages)}\n\n"
            "L'exercice doit citer les passages utilisés et préciser le barème "
            "ainsi que le coefficient de difficulté ressenti.\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_EXERCISE_SCHEMA}"
        )
        return self._complete(prompt)

    def generate_final_exam(
        self,
        concepts: list[dict[str, Any]],
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es un tuteur. Génère un examen final couvrant tous les concepts "
            "ci-dessous, au niveau demandé par les objectifs.\n\n"
            f"CONCEPTS :\n{_format_concepts(concepts)}\n\n"
            f"OBJECTIFS : {objectives}\n"
            f"PRÉFÉRENCES : {_format_preferences(preferences)}\n\n"
            f"PASSAGES :\n{_format_passages(passages)}\n\n"
            "Chaque question doit citer les passages utilisés.\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_ASSESSMENT_SCHEMA}"
        )
        return self._complete(prompt)

    def grade_response(
        self, question: dict[str, Any], response_text: str
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es un tuteur. Note la réponse de l'apprenant par rapport à la "
            "réponse attendue et au barème.\n\n"
            f"QUESTION : {question.get('prompt')}\n"
            f"RÉPONSE ATTENDUE : {question.get('reference_answer')}\n"
            f"BARÈME : {question.get('grading_rubric')}\n"
            f"RÉPONSE DE L'APPRENANT : {response_text}\n\n"
            "score est dans [0, 1]. assistance_coefficient est dans [0, 1] : "
            "1 signifie sans aide, 0 signifie aide totale.\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_GRADE_SCHEMA}"
        )
        return self._complete(prompt)
