"""Model boundary for the web layer.

The Hermes agent is the only engine. The web application calls it as a
subprocess (``hermes -z``, which prints only the final response); the agent
reads the learner's profile and documents with its own tools and returns
structured JSON. The backend validates and persists every proposal: the agent
never decides a level, a gap or an ordering.

The profile is not injected into the prompts. The agent reads it itself with
the ``profile`` command, so the SQL stays the single source of truth.

The documents, on the other hand, are injected: their text (converted once at
upload, see ``backend.tutor.documents``) is placed directly in the prompt, so
the agent does not spend dozens of tool calls reading a PDF piece by piece.
Above ``TUTOR_INLINE_CHARS`` characters (default 200 000) the prompt lists the
text files instead and the agent reads them with ``read_file``.
"""

from __future__ import annotations

import json
import re
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..tutor import documents as document_store
from ..tutor.orchestration import MAX_INITIAL_QUESTIONS, Generator


REPO_ROOT = Path(__file__).resolve().parents[2]

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
      "evidence": [{"document_id": "string", "page": 1, "quote": "string"}]
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
      "citations": [{"document_id": "string", "page": 1, "quote": "string"}]
    }
  ]
}"""

_COURSE_SCHEMA = """{
  "concept_id": "string",
  "title": "string",
  "content": "string",
  "citations": [{"document_id": "string", "page": 1, "quote": "string"}]
}"""

_EXERCISE_SCHEMA = """{
  "question_id": "string",
  "concept_id": "string",
  "prompt": "string",
  "reference_answer": "string",
  "grading_rubric": "string",
  "difficulty_coefficient": 1.0,
  "citations": [{"document_id": "string", "page": 1, "quote": "string"}]
}"""

_GRADE_SCHEMA = """{
  "score": 0.0,
  "assistance_coefficient": 1.0,
  "feedback": "string"
}"""


# LaTeX commands whose first letter forms a valid JSON escape (\\b \\f \\n
# \\r \\t): written with a single backslash inside a JSON string they would
# silently become control characters ("\\frac" → form feed + "rac").
_LATEX_ESCAPE_TRAPS = re.compile(
    r"(?<!\\)\\(?=(?:"
    r"b(?:eta|ar|inom|igcup|igcap|igg?[lr]?|oldsymbol|f|ullet|ot|ox|ackslash|egin|mod)"
    r"|f(?:rac|orall|lat|box|rown)"
    r"|n(?:abla|eq|e|u|ot|otin|olimits|eg|i|exists|leq|geq|ewline)"
    r"|r(?:ho|ight|ightarrow|angle|m|ceil|floor|vert|Vert|brace|brack)"
    r"|t(?:heta|imes|ext|extbf|extit|extrm|au|o|op|frac|ilde|riangle|ri|ext[a-z]*)"
    r")(?![a-zA-Z]))"
)
_INVALID_ESCAPE = re.compile(r'(?<!\\)\\(?=[^"\\/bfnrtu])')


def _repair_latex(text: str) -> str:
    """Double the lone backslashes of LaTeX commands inside a JSON text."""
    return _LATEX_ESCAPE_TRAPS.sub(r"\\\\", text)


def _decode_first(text: str) -> Any:
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


def extract_json(text: Any) -> Any:
    """Return the first JSON value found in a model response.

    Handles markdown code fences and prose around the JSON, and LaTeX written
    with single backslashes (\\frac, \\sum ...) inside JSON strings. Raises
    ValueError when no valid JSON value is present.
    """
    if not isinstance(text, str):
        raise ValueError("the model must return text")
    repaired = _repair_latex(text)
    try:
        return _decode_first(repaired)
    except ValueError:
        # e.g. "\\sum" with one backslash: an invalid JSON escape
        return _decode_first(_INVALID_ESCAPE.sub(r"\\\\", repaired))


def _format_documents(documents: list[dict[str, Any]]) -> str:
    """Render the documents for a prompt: their full text when it fits.

    Each document's converted text (with ``=== Page N ===`` markers) is
    inlined between ``<document>`` tags. When the text is missing or the
    total is above the inline budget, the document is listed with the path of
    its text file (or of the PDF) for the agent to read itself.
    """
    if not documents:
        return "(aucun document)"
    budget = int(os.environ.get("TUTOR_INLINE_CHARS", "200000"))
    texts = [
        document_store.read_text(document["path"]) if document.get("path") else None
        for document in documents
    ]
    inline = sum(len(text or "") for text in texts) <= budget
    sections = []
    for document, text in zip(documents, texts):
        header = f'document_id="{document.get("document_id", "?")}"'
        if document.get("title"):
            header += f' title="{document["title"]}"'
        if text is not None and inline:
            sections.append(f"<document {header}>\n{text.strip()}\n</document>")
        else:
            path = document.get("path", "")
            if text is not None:
                path = str(document_store.text_path(path))
            sections.append(f"<document {header} path=\"{path}\" />")
    return "\n\n".join(sections)


def _documents_instruction(documents: list[dict[str, Any]]) -> str:
    """Tell the agent how to access the documents rendered above."""
    return (
        "Le contenu des documents est fourni ci-dessus (une balise <document> "
        "par fichier, avec des marqueurs === Page N === pour citer la page). "
        "Ne relis pas les fichiers. Exception : si une balise <document> est "
        "vide et n'indique qu'un chemin, lis ce fichier avec `read_file`."
    )


def _format_concepts(concepts: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {concept.get('concept_id')} — {concept.get('label')} "
        f"(cible {concept.get('target_level')}, poids {concept.get('weight')})"
        for concept in concepts
    )


def _format_preferences(preferences: Mapping[str, Any] | None) -> str:
    shown = {k: v for k, v in (preferences or {}).items() if k != "daily_minutes"}
    if not shown:
        return "(non déclarées)"
    return ", ".join(f"{key}={value}" for key, value in shown.items())


_FORMAT = 'Mise en forme des textes (champs content, prompt, reference_answer, feedback) : Markdown (titres ##, listes, **gras**, citations > pour les définitions et théorèmes) et formules en LaTeX entre $...$ en ligne ou $$...$$ en bloc (par ex. $\\mathbb{P}(X_n = x)$, $\\mathcal{F}$). Dans le JSON, double chaque barre oblique inverse : écris \\\\frac pour obtenir \\frac.'


_NO_TOOLS = (
    "En dehors des étapes ci-dessus, n'utilise aucun outil : ne lis, ne cherche et n'écris aucun "
    "fichier, n'ouvre aucun skill, n'appelle aucun outil `save_*`. Ta réponse "
    "finale doit contenir le JSON lui-même (pas un chemin de fichier) ; le "
    "backend le validera et le persistera."
)


def _allowed_concepts(
    concepts: list[dict[str, Any]], require_coverage: bool = True
) -> str:
    ids = ", ".join(str(concept.get("concept_id")) for concept in concepts)
    text = (
        f"concept_id AUTORISÉS (aucun autre) : {ids}. Chaque question doit "
        "utiliser exactement l'un de ces identifiants"
    )
    if require_coverage:
        text += ", et chaque concept doit être couvert par au moins une question"
    return text + "."


def _feedback_block(feedback: str | None) -> str:
    return f"CORRECTION DEMANDÉE :\n{feedback}\n\n" if feedback else ""


class HermesGenerator(Generator):
    """Generator that delegates every task to the Hermes agent.

    The agent runs as a subprocess from the repository root, reads the
    learner's profile and documents with its own tools, and answers with the
    JSON the backend expects. The command can be overridden for tests.
    """

    def __init__(
        self,
        command: list[str] | None = None,
        timeout: int | None = None,
        cwd: str | None = None,
    ) -> None:
        self.command = command or ["hermes", "-z"]
        # Free models are slow: default 30 min, override with TUTOR_HERMES_TIMEOUT (seconds).
        self.timeout = timeout or int(os.environ.get("TUTOR_HERMES_TIMEOUT", "1800"))
        self.cwd = cwd or str(REPO_ROOT)

    def _complete(self, prompt: str) -> Any:
        """Run the agent and return the JSON of its answer.

        When the answer holds no JSON (the model answered in prose, or Hermes
        printed an error such as an overloaded model), the agent is run once
        more. The final error quotes the beginning of the answer.
        """
        output = self._run(prompt)
        try:
            return extract_json(output)
        except ValueError:
            pass
        output = self._run(prompt)
        try:
            return extract_json(output)
        except ValueError:
            excerpt = " ".join(output.split())[:300] or "(réponse vide)"
            raise ValueError(
                f"no valid JSON found in the model response (twice). Hermes answered: {excerpt}"
            ) from None

    def _run(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                [*self.command, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"the Hermes agent did not answer within {self.timeout} s "
                "(set TUTOR_HERMES_TIMEOUT to allow more time)"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"the Hermes agent failed ({result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    def propose_graph(
        self,
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es le tuteur de HermesTeacher. On te demande de proposer un "
            "graphe de prérequis.\n\n"
            f"OBJECTIFS : {objectives}\n"
            f"PRÉFÉRENCES DU CONCEPT : {_format_preferences(preferences)}\n\n"
            f"DOCUMENTS :\n{_format_documents(documents)}\n\n"
            f"{_documents_instruction(documents)}\n\n"
            "Étapes :\n"
            "1. Lis le profil global de l'élève : "
            "`python -m backend.tutor.agent_tools profile`.\n"
            "2. Propose un graphe de prérequis acyclique, avec des citations "
            "(document_id, page, extrait) tirées du contenu ci-dessus.\n\n"
            f"{_NO_TOOLS}\n\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_GRAPH_SCHEMA}"
        )
        return self._complete(prompt)

    def generate_initial_evaluation(
        self,
        concepts: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
        feedback: str | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es le tuteur de HermesTeacher. On te demande de générer une "
            "évaluation initiale.\n\n"
            f"CONCEPTS :\n{_format_concepts(concepts)}\n\n"
            f"OBJECTIFS : {objectives}\n"
            f"PRÉFÉRENCES DU CONCEPT : {_format_preferences(preferences)}\n\n"
            f"DOCUMENTS :\n{_format_documents(documents)}\n\n"
            f"{_documents_instruction(documents)}\n\n"
            "Étapes :\n"
            "1. Lis le profil global : "
            "`python -m backend.tutor.agent_tools profile`.\n"
            "2. Génère une évaluation initiale courte qui situe le niveau "
            "de l'élève sur les notions générales du cours.\n\n"
            "Contraintes de l'évaluation initiale :\n"
            f"- {MAX_INITIAL_QUESTIONS} questions au maximum ;\n"
            "- une seule question par item : pas de sous-questions "
            "(pas de a), b), c)), pas de question à plusieurs parties ;\n"
            "- uniquement les aspects généraux et essentiels du cours "
            "(définitions clés, idées principales, résultats centraux), pas "
            "de détails techniques ni de démonstrations ;\n"
            "- difficulté modérée : une réponse courte (une à trois phrases ou "
            "un petit calcul) doit suffire ;\n"
            "- il n'est pas nécessaire de couvrir tous les concepts : rattache "
            "chaque question au concept_id le plus pertinent, en privilégiant "
            "les concepts de base du graphe.\n\n"
            f"{_allowed_concepts(concepts, require_coverage=False)}\n\n"
            f"{_FORMAT}\n\n"
            f"{_feedback_block(feedback)}"
            f"{_NO_TOOLS}\n\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_ASSESSMENT_SCHEMA}"
        )
        return self._complete(prompt)

    def generate_course(
        self,
        concept: dict[str, Any],
        documents: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es le tuteur de HermesTeacher. On te demande d'écrire un "
            "mini-cours.\n\n"
            f"CONCEPT : {concept.get('concept_id')} — {concept.get('label')}\n"
            f"NIVEAU ACTUEL : {current_level}\n"
            f"PRÉFÉRENCES DU CONCEPT : {_format_preferences(preferences)}\n\n"
            f"DOCUMENTS :\n{_format_documents(documents)}\n\n"
            f"{_documents_instruction(documents)}\n\n"
            "Étapes :\n"
            "1. Lis le profil global : "
            "`python -m backend.tutor.agent_tools profile`.\n"
            "2. Écris un mini-cours structuré et adapté au niveau, ancré sur "
            "les documents : une courte introduction, des sections ## avec "
            "définitions, exemples et points de vigilance, puis un résumé. "
            "Reste sur ce concept : les concepts suivants du graphe peuvent "
            "être cités comme « à venir », sans être développés. Ne parle pas "
            "de ces consignes à l'élève.\n\n"
            f"{_FORMAT}\n\n"
            f"Le champ concept_id doit valoir exactement : {concept.get('concept_id')}\n\n"
            f"{_NO_TOOLS}\n\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_COURSE_SCHEMA}"
        )
        return self._complete(prompt)

    def generate_exercise(
        self,
        concept: dict[str, Any],
        documents: list[dict[str, Any]],
        current_level: float,
        preferences: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es le tuteur de HermesTeacher. On te demande de générer un "
            "exercice.\n\n"
            f"CONCEPT : {concept.get('concept_id')} — {concept.get('label')}\n"
            f"NIVEAU ACTUEL : {current_level}\n"
            f"PRÉFÉRENCES DU CONCEPT : {_format_preferences(preferences)}\n\n"
            f"DOCUMENTS :\n{_format_documents(documents)}\n\n"
            f"{_documents_instruction(documents)}\n\n"
            "Étapes :\n"
            "1. Lis le profil global : "
            "`python -m backend.tutor.agent_tools profile`.\n"
            "2. Génère un exercice adapté au niveau, avec le barème et le "
            "coefficient de difficulté ressenti. S'il comporte plusieurs "
            "questions, numérote-les dans l'énoncé.\n\n"
            f"{_FORMAT}\n\n"
            f"Le champ concept_id doit valoir exactement : {concept.get('concept_id')}\n\n"
            f"{_NO_TOOLS}\n\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_EXERCISE_SCHEMA}"
        )
        return self._complete(prompt)

    def generate_final_exam(
        self,
        concepts: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        objectives: str,
        preferences: Mapping[str, Any] | None = None,
        feedback: str | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es le tuteur de HermesTeacher. On te demande de générer un "
            "examen final.\n\n"
            f"CONCEPTS :\n{_format_concepts(concepts)}\n\n"
            f"OBJECTIFS : {objectives}\n"
            f"PRÉFÉRENCES DU CONCEPT : {_format_preferences(preferences)}\n\n"
            f"DOCUMENTS :\n{_format_documents(documents)}\n\n"
            f"{_documents_instruction(documents)}\n\n"
            "Étapes :\n"
            "1. Lis le profil global : "
            "`python -m backend.tutor.agent_tools profile`.\n"
            "2. Génère un examen final couvrant tous les concepts.\n\n"
            f"{_FORMAT}\n\n"
            f"{_allowed_concepts(concepts)}\n\n"
            f"{_feedback_block(feedback)}"
            f"{_NO_TOOLS}\n\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_ASSESSMENT_SCHEMA}"
        )
        return self._complete(prompt)

    def grade_response(
        self, question: dict[str, Any], response_text: str
    ) -> Mapping[str, Any]:
        prompt = (
            "Tu es le tuteur de HermesTeacher. On te demande de noter une "
            "réponse.\n\n"
            f"QUESTION : {question.get('prompt')}\n"
            f"RÉPONSE ATTENDUE : {question.get('reference_answer')}\n"
            f"BARÈME : {question.get('grading_rubric')}\n"
            f"RÉPONSE DE L'APPRENANT : {response_text}\n\n"
            "score est dans [0, 1]. assistance_coefficient est dans [0, 1] : "
            "1 signifie sans aide, 0 signifie aide totale.\n"
            "N'utilise aucun outil (pas même la commande profile) et "
            "n'écris aucun fichier.\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_GRADE_SCHEMA}"
        )
        return self._complete(prompt)

    def grade_responses(
        self, items: list[dict[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        """Grade all the answers of an assessment in a single agent call."""
        prompt = (
            "Tu es le tuteur de HermesTeacher. On te demande de noter "
            f"{len(items)} réponses d'un élève, chacune indépendamment.\n\n"
            f"{_format_grading_items(items)}\n\n"
            "Pour chaque item : score est dans [0, 1] ; assistance_coefficient "
            "est dans [0, 1] (1 = sans aide, 0 = aide totale) ; feedback est "
            "un retour court en français (formules en LaTeX entre $...$, "
            "barres obliques inverses doublées dans le JSON). Donne exactement "
            "une note par question_id ci-dessus.\n"
            "N'utilise aucun outil (pas même la commande profile) et "
            "n'écris aucun fichier.\n"
            "Réponds uniquement avec un objet JSON de la forme :\n"
            f"{_GRADES_SCHEMA}"
        )
        data = self._complete(prompt)
        grades = data.get("grades") if isinstance(data, Mapping) else None
        if not isinstance(grades, list):
            return {}
        return {
            str(grade["question_id"]): grade
            for grade in grades
            if isinstance(grade, Mapping) and grade.get("question_id") is not None
        }


def _format_grading_items(items: list[dict[str, Any]]) -> str:
    blocks = []
    for item in items:
        question = item["question"]
        blocks.append(
            f'<item question_id="{question.get("question_id")}">\n'
            f"QUESTION : {question.get('prompt')}\n"
            f"RÉPONSE ATTENDUE : {question.get('reference_answer')}\n"
            f"BARÈME : {question.get('grading_rubric')}\n"
            f"RÉPONSE DE L'APPRENANT : {item['response_text']}\n"
            "</item>"
        )
    return "\n\n".join(blocks)


_GRADES_SCHEMA = """{
  "grades": [
    {"question_id": "string", "score": 0.0, "assistance_coefficient": 1.0, "feedback": "string"}
  ]
}"""


def make_generator() -> Generator:
    """Return the generator used by the web application: the Hermes agent."""
    return HermesGenerator()
