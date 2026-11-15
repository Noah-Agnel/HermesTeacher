"""Deterministic generator used by the web tests.

The production generator is the agent (or an OpenAI-compatible model). This
stand-in lets the web tests exercise the interface and the deterministic
engine without reading documents or calling a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from backend.tutor.graph import ProposedConcept, ProposedEdge
from backend.tutor.orchestration import Generator


class ScriptedGenerator(Generator):
    """A fixed graph, one question per concept, and a word-overlap grade."""

    def __init__(self) -> None:
        self._exercise_counter = 0

    def propose_graph(
        self,
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        concepts = [
            ProposedConcept("produit-scalaire", "Produit scalaire"),
            ProposedConcept("applications", "Applications"),
        ]
        return {
            "graph_id": f"scripted-{uuid4().hex[:8]}",
            "source_document_ids": [document["document_id"] for document in documents],
            "notes": "Graphe scripté pour les tests.",
            "concepts": [concept.to_mapping() for concept in concepts],
            "edges": [
                ProposedEdge("produit-scalaire", "applications").to_mapping()
            ],
        }

    def generate_initial_evaluation(
        self,
        concepts: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return {
            "questions": [
                {
                    "question_id": f"init-{concept['concept_id']}",
                    "concept_id": concept["concept_id"],
                    "prompt": f"Question — {concept['label']}.",
                    "reference_answer": f"Réponse — {concept['label']}.",
                    "grading_rubric": "Barème.",
                    "difficulty_coefficient": 1.0,
                    "citations": [],
                }
                for concept in concepts
            ]
        }

    def generate_course(
        self,
        concept: dict[str, Any],
        documents: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return {
            "concept_id": concept["concept_id"],
            "title": f"Mini-cours : {concept['label']}",
            "content": f"Contenu du cours pour {concept['label']}.",
            "citations": [],
        }

    def generate_exercise(
        self,
        concept: dict[str, Any],
        documents: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self._exercise_counter += 1
        return {
            "question_id": f"ex-{concept['concept_id']}-{self._exercise_counter}",
            "concept_id": concept["concept_id"],
            "prompt": f"Exercice — {concept['label']}.",
            "reference_answer": f"Réponse — {concept['label']}.",
            "grading_rubric": "Barème.",
            "difficulty_coefficient": 1.0,
            "citations": [],
        }

    def generate_final_exam(
        self,
        concepts: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return {
            "questions": [
                {
                    "question_id": f"final-{concept['concept_id']}",
                    "concept_id": concept["concept_id"],
                    "prompt": f"Examen — {concept['label']}.",
                    "reference_answer": f"Réponse — {concept['label']}.",
                    "grading_rubric": "Barème.",
                    "difficulty_coefficient": 1.0,
                    "citations": [],
                }
                for concept in concepts
            ]
        }

    def grade_response(
        self, question: dict[str, Any], response_text: str
    ) -> Mapping[str, Any]:
        reference = set((question.get("reference_answer") or "").lower().split())
        response = set(response_text.lower().split())
        overlap = len(reference & response) / len(reference) if reference else 0.5
        return {
            "score": round(min(1.0, 0.2 + 0.8 * overlap), 2),
            "assistance_coefficient": 1.0,
            "feedback": "Correction scriptée.",
        }
