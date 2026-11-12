"""Model boundary for the web layer.

Two implementations are provided:

- ``DemoGenerator`` is a deterministic stand-in that extracts a concept graph
  from the deposited documents and generates simple courses, exercises and
  grades. It lets the interface be exercised without a model or an API key.
- ``OpenAICompleter`` calls any OpenAI-compatible chat-completions endpoint
  and is plugged into the backend's ``PromptLLMGenerator``.

The backend keeps every deterministic decision; these generators only supply
the non-deterministic content.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any
import urllib.error
import urllib.request
from uuid import uuid4

from ..tutor.llm import PromptLLMGenerator
from ..tutor.orchestration import Generator


_WORD_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ0-9']+")
_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9À-ÿ]+")


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD_PATTERN.findall(text or "")}


def _slug(text: str) -> str:
    slug = _SLUG_PATTERN.sub("-", text.lower()).strip("-")
    return slug[:40] or "concept"


def _clean_label(text: str) -> str:
    return text.strip().lstrip("#").strip()


class DemoGenerator(Generator):
    """Deterministic generator used when no model is configured.

    It reads the passages, proposes a small prerequisite chain of concepts,
    and produces one question per concept for the initial evaluation, the
    final exam and the generated courses and exercises.
    """

    def __init__(self) -> None:
        self._exercise_counter = 0

    @staticmethod
    def _evidence(passages: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not passages:
            return []
        passage = passages[0]
        return [
            {
                "document_id": passage["document_id"],
                "passage_id": passage["passage_id"],
            }
        ]

    def _extract_concepts(
        self, passages: list[dict[str, Any]], objectives: str
    ) -> list[tuple[str, list[dict[str, str]]]]:
        extracted: list[tuple[str, list[dict[str, str]]]] = []
        seen: set[str] = set()
        for passage in passages:
            label = _clean_label(passage.get("heading") or "")
            if not label:
                text = passage.get("text", "").strip()
                label = _clean_label(text.split("\n")[0]) if text else ""
            if len(label) < 3 or label in seen:
                continue
            seen.add(label)
            extracted.append(
                (
                    label,
                    [
                        {
                            "document_id": passage["document_id"],
                            "passage_id": passage["passage_id"],
                        }
                    ],
                )
            )
            if len(extracted) >= 8:
                break
        if not extracted:
            label = _clean_label(objectives) or "Concept à maîtriser"
            extracted.append((label[:80], self._evidence(passages)))
        return extracted

    def propose_graph(
        self,
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        extracted = self._extract_concepts(passages, objectives)
        concepts: list[dict[str, Any]] = []
        concept_ids: list[str] = []
        for label, evidence in extracted:
            base = _slug(label)
            concept_id = base
            suffix = 2
            while concept_id in concept_ids:
                concept_id = f"{base}-{suffix}"
                suffix += 1
            concept_ids.append(concept_id)
            concepts.append(
                {
                    "concept_id": concept_id,
                    "label": label,
                    "description": f"Concept extrait des documents : {label}.",
                    "target_level": 1.0,
                    "weight": 1.0,
                    "evidence": evidence,
                }
            )
        edges = [
            {"prerequisite_id": previous, "concept_id": current, "evidence": []}
            for previous, current in zip(concept_ids, concept_ids[1:])
        ]
        return {
            "graph_id": f"demo-graph-{uuid4().hex[:8]}",
            "source_document_ids": list(
                dict.fromkeys(passage["document_id"] for passage in passages)
            ),
            "notes": "Graphe généré par le démonstrateur.",
            "concepts": concepts,
            "edges": edges,
        }

    def generate_initial_evaluation(
        self,
        concepts: list[dict[str, Any]],
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return {
            "questions": [
                {
                    "question_id": f"init-{concept['concept_id']}",
                    "concept_id": concept["concept_id"],
                    "prompt": (
                        f"Évaluation initiale — {concept['label']}. "
                        "Définissez et expliquez ce concept."
                    ),
                    "reference_answer": f"Réponse attendue pour {concept['label']}.",
                    "grading_rubric": (
                        "1 pour une définition correcte, 0.5 pour une définition "
                        "partielle, 0 sinon."
                    ),
                    "difficulty_coefficient": 0.8,
                    "citations": self._evidence(passages),
                }
                for concept in concepts
            ]
        }

    def generate_course(
        self,
        concept: dict[str, Any],
        passages: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return {
            "concept_id": concept["concept_id"],
            "title": f"Mini-cours : {concept['label']}",
            "content": (
                f"Ce mini-cours couvre « {concept['label']} ».\n\n"
                f"Niveau actuel estimé : {current_level:.2f}.\n\n"
                "Définition, propriétés et exemples. Lisez attentivement : "
                "l'exercice qui suit vérifiera votre compréhension. "
                "Un cours lu ne prouve pas la maîtrise."
            ),
            "citations": self._evidence(passages),
        }

    def generate_exercise(
        self,
        concept: dict[str, Any],
        passages: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self._exercise_counter += 1
        return {
            "question_id": (
                f"ex-{concept['concept_id']}-{self._exercise_counter}"
            ),
            "concept_id": concept["concept_id"],
            "prompt": (
                f"Exercice — {concept['label']}. "
                "Appliquez la notion à un cas concret."
            ),
            "reference_answer": f"Réponse type pour {concept['label']}.",
            "grading_rubric": (
                "1 pour une réponse complète et correcte, 0.5 pour une réponse "
                "partielle, 0 sinon."
            ),
            "difficulty_coefficient": 1.0,
            "citations": self._evidence(passages),
        }

    def generate_final_exam(
        self,
        concepts: list[dict[str, Any]],
        passages: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return {
            "questions": [
                {
                    "question_id": f"final-{concept['concept_id']}",
                    "concept_id": concept["concept_id"],
                    "prompt": (
                        f"Examen final — {concept['label']}. "
                        "Question de synthèse."
                    ),
                    "reference_answer": f"Réponse attendue pour {concept['label']}.",
                    "grading_rubric": "Barème de l'examen final.",
                    "difficulty_coefficient": 1.0,
                    "citations": self._evidence(passages),
                }
                for concept in concepts
            ]
        }

    def grade_response(
        self, question: dict[str, Any], response_text: str
    ) -> Mapping[str, Any]:
        reference = _words(question.get("reference_answer", ""))
        response = _words(response_text)
        overlap = len(reference & response) / len(reference) if reference else 0.5
        score = min(1.0, 0.2 + 0.8 * overlap)
        return {
            "score": round(score, 2),
            "assistance_coefficient": 1.0,
            "feedback": (
                "Correction du démonstrateur : score basé sur le recouvrement "
                "avec la réponse attendue."
            ),
        }


class OpenAICompleter:
    """Call an OpenAI-compatible chat-completions endpoint with stdlib HTTP."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"the model request failed ({error.code}): {body}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"the model request failed: {error.reason}") from error
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"unexpected model response: {data!r}") from error


def make_generator(settings) -> Generator:
    """Return the generator configured by the settings."""
    if settings.demo:
        return DemoGenerator()
    return PromptLLMGenerator(
        OpenAICompleter(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            settings.llm_timeout,
        )
    )
