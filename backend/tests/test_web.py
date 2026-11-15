import io
import json
import re
import tempfile
import unittest
from pathlib import Path

try:
    from backend.web.app import create_app
    from backend.web.config import Settings
    from backend.tests.generators import ScriptedGenerator
    from backend.web.graphview import layout_graph
    from backend.web.profile import ProfileStore
    from backend.tutor.graph import ConceptGraphProposal
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - dependencies missing
    create_app = None


def make_pdf_bytes(lines):
    """Build a small PDF with one text line per page, as bytes."""
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    canvas_object = canvas.Canvas(buffer)
    for line in lines:
        canvas_object.drawString(72, 720, line)
        canvas_object.showPage()
    canvas_object.save()
    return buffer.getvalue()


COURSE = make_pdf_bytes(
    [
        "Produit scalaire",
        "Le produit scalaire est une forme bilineaire symetrique.",
        "Applications",
        "On l'utilise pour calculer des angles et des longueurs.",
    ]
)


def make_settings(root: Path) -> Settings:
    return Settings(
        data_dir=root,
        profile_path=root / "profile.sqlite3",
    )


class ScriptedGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = ScriptedGenerator()

    def test_proposes_a_valid_graph(self):
        mapping = self.generator.propose_graph([], "objectifs")
        proposal = ConceptGraphProposal.from_mapping(mapping)
        report = proposal.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(len(proposal.concepts), 2)
        self.assertEqual(len(proposal.edges), 1)

    def test_generates_questions_courses_and_grades(self):
        concepts = [
            {"concept_id": "produit-scalaire", "label": "Produit scalaire"},
            {"concept_id": "applications", "label": "Applications"},
        ]
        documents = []

        evaluation = self.generator.generate_initial_evaluation(
            concepts, documents, "objectifs"
        )
        self.assertEqual(len(evaluation["questions"]), 2)

        course = self.generator.generate_course(concepts[0], documents, 0.4)
        self.assertEqual(course["concept_id"], "produit-scalaire")

        exercise = self.generator.generate_exercise(concepts[0], documents, 0.4)
        self.assertEqual(exercise["concept_id"], "produit-scalaire")

        exam = self.generator.generate_final_exam(concepts, documents, "objectifs")
        self.assertEqual(len(exam["questions"]), 2)

        grade = self.generator.grade_response(
            {"reference_answer": "reponse attendue"}, "la reponse attendue"
        )
        self.assertAlmostEqual(grade["score"], 1.0)


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
        self.app = create_app(self.settings, generator=ScriptedGenerator())
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
                "eta": "0.3",
                "theta": "0.75",
            },
            files={"files": ("cours.pdf", COURSE, "application/pdf")},
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
        self.assertGreaterEqual(summary["concept_count"], 1)

        response = self.client.get(f"/concepts/{concept_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Produit scalaire", response.text)
        self.assertIn("Évaluation initiale", response.text)

    def test_a_failed_evaluation_keeps_the_concept_and_can_be_retried(self):
        generator = self.service.generator
        original = generator.generate_initial_evaluation
        failures = {"left": 1}

        def flaky(*args, **kwargs):
            if failures["left"]:
                failures["left"] -= 1
                raise RuntimeError("model at capacity")
            return original(*args, **kwargs)

        generator.generate_initial_evaluation = flaky
        response = self.client.post(
            "/concepts/new",
            data={"title": "Produit scalaire", "objectives": "", "format": "texte"},
            files={"files": ("cours.pdf", COURSE, "application/pdf")},
        )

        self.assertEqual(response.status_code, 303)
        location = response.headers["location"]
        self.assertIn("error=", location)
        concept_id = location.split("?")[0].split("/")[-1]
        self.assertEqual(self.service.get_activity(concept_id)["kind"], "evaluation_pending")
        page = self.client.get(f"/concepts/{concept_id}")
        self.assertIn("Générer l'évaluation initiale", page.text)
        self.assertIn("model at capacity", self.client.get(location).text)

        response = self.client.post(f"/concepts/{concept_id}/evaluation/generate")

        self.assertEqual(response.status_code, 303)
        self.assertNotIn("error=", response.headers["location"])
        self.assertEqual(self.service.summary(concept_id)["phase"], "initial_evaluation")

    def test_dont_know_checkbox_is_offered_and_skips_the_model(self):
        concept_id = self.create_concept()
        page = self.client.get(f"/concepts/{concept_id}")
        self.assertIn("Je ne sais pas", page.text)
        questions = self.service.get_activity(concept_id)["questions"]
        generator = self.service.generator
        calls_before = len([c for c in generator.calls if c[0] == "grade_response"]) if hasattr(generator, "calls") else 0

        response = self.client.post(
            f"/concepts/{concept_id}/evaluation",
            data={f"dontknow-{question['question_id']}": "on" for question in questions},
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.service.summary(concept_id)["phase"], "learning")
        if hasattr(generator, "calls"):
            calls_after = len([c for c in generator.calls if c[0] == "grade_response"])
            self.assertEqual(calls_after, calls_before)

    def test_answers_are_kept_when_grading_fails(self):
        concept_id = self.create_concept()
        questions = self.service.get_activity(concept_id)["questions"]

        def broken(*args, **kwargs):
            raise ValueError("no valid JSON found in the model response")

        self.service.generator.grade_response = broken
        self.service.generator.grade_responses = broken
        response = self.client.post(
            f"/concepts/{concept_id}/evaluation",
            data={f"response-{question['question_id']}": "ma réponse détaillée" for question in questions},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("no valid JSON", response.text)
        self.assertIn("ma réponse détaillée", response.text)
        self.assertEqual(self.service.summary(concept_id)["phase"], "initial_evaluation")

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
            data={"response_text": "Réponse — Produit scalaire"},
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
        self.assertEqual(list((self.settings.data_dir / "uploads").iterdir()), [])

    def test_a_failed_creation_leaves_no_files(self):
        def broken(*args, **kwargs):
            raise ValueError("model unavailable")

        self.service.generator.propose_graph = broken
        response = self.client.post(
            "/concepts/new",
            data={"title": "Concept"},
            files={"files": ("cours.pdf", COURSE, "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(list((self.settings.data_dir / "uploads").iterdir()), [])

    def test_invalid_document_is_rejected(self):
        response = self.client.post(
            "/concepts/new",
            data={"title": "Concept"},
            files={"files": ("image.png", b"not a document", "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.service.list_concepts(), [])

    def test_profile_page_saves_global_profile(self):
        response = self.client.get("/profile")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Mon profil", response.text)

        response = self.client.post(
            "/profile",
            data={
                "display_name": "Noah",
                "format": "exemples",
                "focus": "les applications",
                "difficulties": "les démonstrations",
                "weaknesses": "démonstrations\n géométrie",
            },
        )
        self.assertEqual(response.status_code, 303)
        profile = self.service.get_profile()
        self.assertEqual(profile["display_name"], "Noah")
        self.assertEqual(profile["preferences"]["format"], "exemples")
        self.assertIn("démonstrations", profile["weaknesses"])

    def test_save_course_keeps_it_after_next_exercise(self):
        concept_id = self.create_concept()
        activity = self.service.get_activity(concept_id)
        self.client.post(
            f"/concepts/{concept_id}/evaluation",
            data={
                f"response-{question['question_id']}": "r"
                for question in activity["questions"]
            },
        )
        self.client.post(f"/concepts/{concept_id}/lesson")
        activity = self.service.get_activity(concept_id)
        course_id = activity["course"]["course_id"]

        self.client.post(
            f"/concepts/{concept_id}/course/save",
            data={"course_id": course_id},
        )
        self.client.post(f"/concepts/{concept_id}/exercise")

        store = self.service._store(concept_id)
        self.assertIsNotNone(store.get_course("student", course_id))


class JobApiTests(unittest.TestCase):
    """The JSON job API runs the slow actions in the background."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.settings = make_settings(Path(self.temporary_directory.name))
        self.app = create_app(self.settings, generator=ScriptedGenerator())
        self.client = TestClient(self.app, follow_redirects=False)
        self.service = self.app.state.service

    def tearDown(self):
        self.temporary_directory.cleanup()

    def wait(self, job_id):
        import time

        for _ in range(200):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] != "running":
                return job
            time.sleep(0.02)
        self.fail("job did not finish")

    def create(self):
        response = self.client.post(
            "/api/concepts",
            data={"title": "Produit scalaire", "objectives": "", "format": "texte"},
            files={"files": ("cours.pdf", COURSE, "application/pdf")},
        )
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual([s["key"] for s in job["steps"]], ["documents", "graph", "evaluation"])
        return self.wait(job["job_id"])

    def test_creating_a_concept_reports_its_steps_and_redirects(self):
        job = self.create()

        self.assertEqual(job["status"], "done", job)
        self.assertTrue(all(step["status"] == "done" for step in job["steps"]))
        concept_id = job["redirect"].split("/")[-1]
        self.assertEqual(self.service.summary(concept_id)["phase"], "initial_evaluation")

    def test_the_exercise_result_card_carries_score_and_levels(self):
        concept_id = self.create()["redirect"].split("/")[-1]
        questions = self.service.get_activity(concept_id)["questions"]
        job = self.wait(
            self.client.post(
                f"/api/concepts/{concept_id}/evaluation",
                data={f"response-{q['question_id']}": "la reponse attendue" for q in questions},
            ).json()["job_id"]
        )
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(self.wait(self.client.post(f"/api/concepts/{concept_id}/lesson").json()["job_id"])["status"], "done")
        self.assertEqual(self.wait(self.client.post(f"/api/concepts/{concept_id}/exercise").json()["job_id"])["status"], "done")

        job = self.wait(
            self.client.post(
                f"/api/concepts/{concept_id}/exercise/submit",
                data={"response_text": "la reponse attendue"},
            ).json()["job_id"]
        )

        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"]["card"], "exercise")
        self.assertIn("score", job["result"])
        self.assertIsNotNone(job["result"]["level_before"])

    def test_a_failing_job_reports_the_error(self):
        concept_id = self.create()["redirect"].split("/")[-1]

        def broken(*args, **kwargs):
            raise ValueError("no valid JSON found in the model response")

        self.service.generator.grade_response = broken
        self.service.generator.grade_responses = broken
        questions = self.service.get_activity(concept_id)["questions"]
        job = self.wait(
            self.client.post(
                f"/api/concepts/{concept_id}/evaluation",
                data={f"response-{q['question_id']}": "réponse" for q in questions},
            ).json()["job_id"]
        )

        self.assertEqual(job["status"], "error")
        self.assertIn("no valid JSON", job["error"])
        self.assertEqual(self.service.summary(concept_id)["phase"], "initial_evaluation")

    def test_graph_page_embeds_statuses(self):
        concept_id = self.create()["redirect"].split("/")[-1]

        response = self.client.get(f"/concepts/{concept_id}/graph")

        self.assertEqual(response.status_code, 200)
        self.assertIn("graph-data", response.text)
        view = self.service.graph_view(concept_id)
        statuses = {node["id"]: node["status"] for node in view["nodes"]}
        self.assertEqual(set(statuses.values()) <= {"open", "locked", "current", "mastered"}, True)
        self.assertIn("locked", statuses.values())

    def test_unknown_job_is_404(self):
        self.assertEqual(self.client.get("/api/jobs/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
