import json
import re
import tempfile
import unittest
from pathlib import Path
import urllib.request

try:
    from backend.web.app import create_app
    from backend.web.config import Settings
    from backend.web.generators import DemoGenerator
    from backend.web.graphview import layout_graph
    from backend.web.profile import ProfileStore
    from backend.tutor.graph import ConceptGraphProposal
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - dependencies missing
    create_app = None


COURSE = b"""# Produit scalaire

Le produit scalaire est une forme bilineaire symetrique.

# Applications

On l'utilise pour calculer des angles et des longueurs.
"""


def make_settings(root: Path) -> Settings:
    return Settings(
        data_dir=root,
        profile_path=root / "profile.sqlite3",
        llm_base_url="https://example.invalid/v1",
        llm_api_key="",
        llm_model="demo",
        llm_timeout=5,
        demo=True,
    )


class DemoGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = DemoGenerator()

    def test_proposes_a_valid_graph_from_passages(self):
        passages = [
            {
                "passage_id": "doc-1:0000",
                "document_id": "doc-1",
                "heading": "Produit scalaire",
                "text": "Contenu.",
            },
            {
                "passage_id": "doc-1:0001",
                "document_id": "doc-1",
                "heading": "Applications",
                "text": "Contenu.",
            },
        ]

        mapping = self.generator.propose_graph(passages, "objectifs")
        proposal = ConceptGraphProposal.from_mapping(mapping)
        report = proposal.validate(
            known_passages={"doc-1:0000", "doc-1:0001"}
        )

        self.assertTrue(report.is_valid)
        self.assertEqual(len(proposal.concepts), 2)
        self.assertEqual(len(proposal.edges), 1)

    def test_generates_questions_courses_and_grades(self):
        concepts = [{"concept_id": "a", "label": "Concept A"}]
        passages = [
            {
                "passage_id": "doc-1:0000",
                "document_id": "doc-1",
                "text": "Contenu.",
            }
        ]

        evaluation = self.generator.generate_initial_evaluation(
            concepts, passages, "objectifs"
        )
        self.assertEqual(len(evaluation["questions"]), 1)

        course = self.generator.generate_course(concepts[0], passages, 0.4)
        self.assertEqual(course["concept_id"], "a")

        exercise = self.generator.generate_exercise(concepts[0], passages, 0.4)
        self.assertEqual(exercise["concept_id"], "a")

        exam = self.generator.generate_final_exam(concepts, passages, "objectifs")
        self.assertEqual(len(exam["questions"]), 1)

        grade = self.generator.grade_response(
            {"reference_answer": "reponse attendue"}, "la reponse attendue"
        )
        self.assertAlmostEqual(grade["score"], 1.0)


class OpenAICompleterTests(unittest.TestCase):
    def test_calls_chat_completions_and_parses_content(self):
        from backend.web.generators import OpenAICompleter

        captured: dict = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices": [{"message": {"content": "hello"}}]}'

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["data"] = request.data
            captured["headers"] = {
                key.lower(): value for key, value in request.header_items()
            }
            captured["timeout"] = timeout
            return FakeResponse()

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            completer = OpenAICompleter(
                "https://api.example/v1", "secret", "model-x", timeout=7
            )
            result = completer("prompt text")
        finally:
            urllib.request.urlopen = original

        self.assertEqual(result, "hello")
        self.assertEqual(captured["url"], "https://api.example/v1/chat/completions")
        payload = json.loads(captured["data"])
        self.assertEqual(payload["model"], "model-x")
        self.assertEqual(payload["messages"][0]["content"], "prompt text")
        self.assertEqual(captured["headers"].get("authorization"), "Bearer secret")
        self.assertEqual(captured["timeout"], 7)


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProfileStore(
            Path(self.temporary_directory.name) / "profile.sqlite3"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_create_get_update_and_delete(self):
        concept_id = self.store.create_concept(
            title="Concept",
            objectives="objectifs",
            preferences={"format": "texte"},
            eta=0.3,
            theta=0.75,
            db_path="/tmp/concept.sqlite3",
        )

        concept = self.store.get_concept(concept_id)
        self.assertEqual(concept["title"], "Concept")
        self.assertEqual(concept["preferences"], {"format": "texte"})

        self.store.update_concept(
            concept_id, title="Nouveau", preferences={"format": "exemples"}
        )
        self.assertEqual(self.store.get_concept(concept_id)["title"], "Nouveau")
        self.assertEqual(
            self.store.get_concept(concept_id)["preferences"], {"format": "exemples"}
        )

        listed = self.store.list_concepts()
        self.assertEqual([item["concept_id"] for item in listed], [concept_id])

        self.store.delete_concept(concept_id)
        self.assertIsNone(self.store.get_concept(concept_id))

    def test_update_rejects_unknown_fields(self):
        concept_id = self.store.create_concept(
            title="Concept",
            objectives="",
            preferences={},
            eta=0.3,
            theta=0.75,
            db_path="/tmp/concept.sqlite3",
        )
        with self.assertRaises(ValueError):
            self.store.update_concept(concept_id, unknown=1)


class GraphViewTests(unittest.TestCase):
    def test_layout_places_prerequisites_to_the_left(self):
        concepts = [
            {"concept_id": "a", "label": "A"},
            {"concept_id": "b", "label": "B"},
            {"concept_id": "c", "label": "C"},
        ]
        edges = [
            {"prerequisite_id": "a", "concept_id": "b"},
            {"prerequisite_id": "b", "concept_id": "c"},
        ]

        layout = layout_graph(concepts, edges, {"a": 0.2, "b": 0.6, "c": 1.0})

        positions = {node["concept_id"]: node for node in layout["nodes"]}
        self.assertLess(positions["a"]["x"], positions["b"]["x"])
        self.assertLess(positions["b"]["x"], positions["c"]["x"])
        self.assertEqual(positions["c"]["percent"], 100)
        self.assertEqual(len(layout["edges"]), 2)


@unittest.skipIf(create_app is None, "web dependencies are not installed")
class WebFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.settings = make_settings(Path(self.temporary_directory.name))
        self.app = create_app(self.settings)
        self.client = TestClient(self.app, follow_redirects=False)
        self.service = self.app.state.service

    def tearDown(self):
        self.temporary_directory.cleanup()

    def create_concept(self):
        response = self.client.post(
            "/concepts/new",
            data={
                "title": "Produit scalaire",
                "objectives": "Maîtriser le produit scalaire",
                "format": "texte",
                "daily_minutes": "20",
                "eta": "0.3",
                "theta": "0.75",
            },
            files={"files": ("cours.md", COURSE, "text/markdown")},
        )
        self.assertEqual(response.status_code, 303, response.text)
        return response.headers["location"].split("/")[-1]

    def test_welcome_page_lists_concepts(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Mes concepts", response.text)
        self.assertIn("Nouveau concept", response.text)

    def test_create_concept_loads_documents_and_starts_evaluation(self):
        concept_id = self.create_concept()

        summary = self.service.summary(concept_id)
        self.assertEqual(summary["phase"], "initial_evaluation")
        self.assertEqual(summary["concept_count"], 2)

        response = self.client.get(f"/concepts/{concept_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Produit scalaire", response.text)
        self.assertIn("Évaluation initiale", response.text)

    def test_full_learning_flow(self):
        concept_id = self.create_concept()

        activity = self.service.get_activity(concept_id)
        self.assertEqual(activity["kind"], "evaluation")
        response = self.client.post(
            f"/concepts/{concept_id}/evaluation",
            data={
                f"response-{question['question_id']}": "la reponse attendue"
                for question in activity["questions"]
            },
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.service.summary(concept_id)["phase"], "learning")

        response = self.client.post(f"/concepts/{concept_id}/lesson")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.service.get_activity(concept_id)["kind"], "lesson")
        response = self.client.get(f"/concepts/{concept_id}")
        self.assertIn("Mini-cours", response.text)

        response = self.client.post(f"/concepts/{concept_id}/exercise")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.service.get_activity(concept_id)["kind"], "exercise")

        response = self.client.post(
            f"/concepts/{concept_id}/exercise/submit",
            data={"response_text": "la reponse type"},
        )
        self.assertEqual(response.status_code, 303)
        summary = self.service.summary(concept_id)
        self.assertGreater(summary["progress"], 0.0)
        self.assertEqual(self.service.get_activity(concept_id)["kind"], "idle")

    def test_final_exam_and_remediation_resume(self):
        concept_id = self.create_concept()
        activity = self.service.get_activity(concept_id)
        self.client.post(
            f"/concepts/{concept_id}/evaluation",
            data={
                f"response-{question['question_id']}": "r"
                for question in activity["questions"]
            },
        )

        store = self.service._store(concept_id)
        graph = store.get_graph(
            "student", self.service.summary(concept_id)["state"]["graph_id"]
        )
        for concept in graph["concepts"]:
            store.save_mastery(
                "student",
                concept["concept_id"],
                current_level=1.0,
                target_level=1.0,
                weight=1.0,
            )

        detail = self.client.get(f"/concepts/{concept_id}").text
        self.assertIn("Passer l'examen final", detail)

        self.client.post(f"/concepts/{concept_id}/final-exam")
        detail = self.client.get(f"/concepts/{concept_id}").text
        self.assertIn("Examen final", detail)
        question_ids = re.findall(r'name="response-([^"]+)"', detail)
        self.assertTrue(question_ids)

        self.client.post(
            f"/concepts/{concept_id}/final-exam/submit",
            data={f"response-{question_id}": "r" for question_id in question_ids},
        )
        detail = self.client.get(f"/concepts/{concept_id}").text
        self.assertIn("Parcours terminé", detail)
        self.assertIn("Reprendre l'apprentissage", detail)

        self.client.post(f"/concepts/{concept_id}/resume")
        summary = self.service.summary(concept_id)
        self.assertEqual(summary["phase"], "learning")

    def test_settings_page_updates_configuration(self):
        concept_id = self.create_concept()

        response = self.client.get(f"/concepts/{concept_id}/settings")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Configuration", response.text)

        response = self.client.post(
            f"/concepts/{concept_id}/settings",
            data={
                "title": "Produit scalaire (mis à jour)",
                "objectives": "nouvel objectif",
                "format": "exemples",
                "daily_minutes": "30",
                "focus": "les applications",
                "difficulties": "les démonstrations",
                "eta": "0.4",
                "theta": "0.8",
            },
        )
        self.assertEqual(response.status_code, 303)
        concept = self.service.get_concept(concept_id)
        self.assertEqual(concept["title"], "Produit scalaire (mis à jour)")
        self.assertEqual(concept["eta"], 0.4)
        self.assertEqual(concept["theta"], 0.8)
        self.assertEqual(concept["preferences"]["focus"], "les applications")

    def test_delete_concept(self):
        concept_id = self.create_concept()

        response = self.client.post(f"/concepts/{concept_id}/delete")
        self.assertEqual(response.status_code, 303)
        self.assertIsNone(self.service.get_concept(concept_id))
        self.assertEqual(self.service.list_concepts(), [])

    def test_invalid_document_is_rejected(self):
        response = self.client.post(
            "/concepts/new",
            data={"title": "Concept"},
            files={"files": ("image.png", b"not a document", "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.service.list_concepts(), [])


if __name__ == "__main__":
    unittest.main()
