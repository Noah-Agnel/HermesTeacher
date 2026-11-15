import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from backend.tutor import agent_tools, documents
from backend.tutor.graph import ConceptGraphProposal, ProposedConcept, ProposedEdge
from backend.tutor.storage import SQLiteStore
from backend.web.profile import ProfileStore


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        self._original_data_dir = agent_tools._data_dir
        agent_tools._data_dir = lambda: self.data_dir

    def tearDown(self):
        agent_tools._data_dir = self._original_data_dir
        self.temporary_directory.cleanup()

    def make_concept(self) -> SQLiteStore:
        profile = ProfileStore(self.data_dir / "profile.sqlite3")
        db_path = self.data_dir / "concepts" / "concept-1.sqlite3"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        profile.create_concept(
            concept_id="concept-1",
            title="Concept 1",
            objectives="objectifs",
            preferences={"format": "texte"},
            eta=0.3,
            theta=0.75,
            db_path=str(db_path),
        )
        store = SQLiteStore(db_path)
        store.create_learner("student")
        return store

    def run_command(self, *argv):
        output = io.StringIO()
        with redirect_stdout(output):
            code = agent_tools.main(list(argv))
        return code, json.loads(output.getvalue())

    def test_concepts_lists_concepts(self):
        self.make_concept()

        code, payload = self.run_command("concepts")

        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"][0]["concept_id"], "concept-1")
        self.assertEqual(payload["data"][0]["title"], "Concept 1")
        self.assertEqual(payload["data"][0]["phase"], "not_started")

    def test_documents_returns_deposited_files(self):
        store = self.make_concept()
        document = documents.store_upload(
            self.data_dir / "uploads",
            document_id="doc-1",
            title="Cours",
            filename="cours.pdf",
            content=b"%PDF-1.4 fake",
        )
        store.save_document("student", document)

        code, payload = self.run_command("documents", "concept-1")

        self.assertEqual(code, 0)
        self.assertEqual(payload["data"][0]["document_id"], "doc-1")
        self.assertEqual(payload["data"][0]["title"], "Cours")

    def test_graph_returns_phase_and_graph(self):
        store = self.make_concept()
        store.save_graph(
            "student",
            ConceptGraphProposal(
                graph_id="graph-1",
                concepts=(
                    ProposedConcept("a", "A"),
                    ProposedConcept("b", "B"),
                ),
                edges=(ProposedEdge("a", "b"),),
            ),
        )
        store.save_parcours_state("student", phase="learning", graph_id="graph-1")

        code, payload = self.run_command("graph", "concept-1")

        self.assertEqual(code, 0)
        self.assertEqual(payload["data"]["graph"]["graph_id"], "graph-1")
        self.assertEqual(len(payload["data"]["graph"]["concepts"]), 2)

    def test_mastery_returns_levels(self):
        store = self.make_concept()
        store.save_mastery(
            "student", "a", current_level=0.5, target_level=1.0, weight=2.0
        )

        code, payload = self.run_command("mastery", "concept-1")

        self.assertEqual(code, 0)
        self.assertEqual(payload["data"][0]["current_level"], 0.5)
        self.assertEqual(payload["data"][0]["weight"], 2.0)

    def test_history_returns_events(self):
        store = self.make_concept()
        store.log_event(
            "student", "document_deposited", payload={"document_id": "doc-1"}
        )

        code, payload = self.run_command("history", "concept-1")

        self.assertEqual(code, 0)
        self.assertEqual(payload["data"][0]["kind"], "document_deposited")

    def test_errors_returns_errors(self):
        store = self.make_concept()
        store.record_error("student", "a", "Erreur de signe.")

        code, payload = self.run_command("errors", "concept-1")

        self.assertEqual(code, 0)
        self.assertEqual(payload["data"][0]["description"], "Erreur de signe.")

    def test_profile_returns_global_profile(self):
        ProfileStore(self.data_dir / "profile.sqlite3").save_learner_profile(
            "student",
            display_name="Noah",
            preferences={"format": "exemples"},
            weaknesses=["démonstrations"],
        )

        code, payload = self.run_command("profile")

        self.assertEqual(code, 0)
        self.assertEqual(payload["data"]["display_name"], "Noah")
        self.assertEqual(payload["data"]["weaknesses"], ["démonstrations"])

    def test_save_profile_updates_global_profile(self):
        path = self.write_json(
            "profile.json",
            {
                "display_name": "Noah",
                "preferences": {"format": "exemples"},
                "weaknesses": ["démonstrations"],
            },
        )

        code, payload = self.run_command("save_profile", str(path))

        self.assertEqual(code, 0)
        profile = ProfileStore(
            self.data_dir / "profile.sqlite3"
        ).get_learner_profile("student")
        self.assertEqual(profile["display_name"], "Noah")
        self.assertEqual(profile["preferences"], {"format": "exemples"})
        self.assertEqual(profile["weaknesses"], ["démonstrations"])

    def test_save_profile_merges_with_the_existing_profile(self):
        ProfileStore(self.data_dir / "profile.sqlite3").save_learner_profile(
            "student",
            display_name="Noah",
            preferences={"format": "texte"},
            weaknesses=["signes"],
        )
        path = self.write_json("profile.json", {"weaknesses": ["démonstrations"]})

        code, payload = self.run_command("save_profile", str(path))

        self.assertEqual(code, 0)
        profile = ProfileStore(
            self.data_dir / "profile.sqlite3"
        ).get_learner_profile("student")
        self.assertEqual(profile["display_name"], "Noah")
        self.assertEqual(profile["preferences"], {"format": "texte"})
        self.assertEqual(profile["weaknesses"], ["démonstrations"])

    def test_save_profile_rejects_an_invalid_payload(self):
        path = self.write_json("profile.json", {"preferences": "pas un dict"})

        code, payload = self.run_command("save_profile", str(path))

        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertIn("preferences must be a mapping", payload["error"])

    def test_save_profile_requires_a_readable_file(self):
        path = self.data_dir / "agent" / "absent.json"

        code, payload = self.run_command("save_profile", str(path))

        self.assertEqual(code, 1)
        self.assertIn("fichier illisible", payload["error"])

    def test_unknown_concept_returns_error(self):
        self.make_concept()

        code, payload = self.run_command("graph", "nope")

        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertIn("concept inconnu", payload["error"])

    @staticmethod
    def _proposal():
        return {
            "graph_id": "graph-1",
            "source_document_ids": [],
            "notes": "proposition de test",
            "concepts": [
                {
                    "concept_id": "a",
                    "label": "Notion A",
                    "description": "description",
                    "target_level": 1.0,
                    "weight": 1.0,
                    "evidence": [],
                },
                {
                    "concept_id": "b",
                    "label": "Notion B",
                    "description": "description",
                    "target_level": 1.0,
                    "weight": 1.0,
                    "evidence": [],
                },
            ],
            "edges": [{"prerequisite_id": "a", "concept_id": "b", "evidence": []}],
        }

    def _write_proposal(self, payload, name="proposal.json"):
        directory = self.data_dir / "agent"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_save_graph_stores_a_valid_proposal(self):
        store = self.make_concept()
        path = self._write_proposal(self._proposal())

        code, payload = self.run_command("save_graph", "concept-1", str(path))

        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["graph_id"], "graph-1")
        self.assertIsNotNone(store.get_graph("student", "graph-1"))

    def test_save_graph_leaves_the_parcours_not_started(self):
        store = self.make_concept()
        path = self._write_proposal(self._proposal())

        code, _ = self.run_command("save_graph", "concept-1", str(path))

        self.assertEqual(code, 0)
        self.assertEqual(
            store.get_parcours_state("student")["phase"], "not_started"
        )

    def test_save_graph_rejects_a_cycle(self):
        self.make_concept()
        proposal = self._proposal()
        proposal["edges"] = [
            {"prerequisite_id": "a", "concept_id": "b", "evidence": []},
            {"prerequisite_id": "b", "concept_id": "a", "evidence": []},
        ]
        path = self._write_proposal(proposal)

        code, payload = self.run_command("save_graph", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertIn("cycle", payload["error"])

    # ------------------------------------------------------------------
    # content proposed by the agent
    # ------------------------------------------------------------------
    def make_graph_concept(self, with_document: bool = False) -> SQLiteStore:
        """Create a concept with a stored graph, but no phase change."""
        store = self.make_concept()
        source_document_ids: tuple[str, ...] = ()
        if with_document:
            document = documents.store_upload(
                self.data_dir / "uploads",
                document_id="doc-1",
                title="Cours",
                filename="cours.pdf",
                content=b"%PDF-1.4 fake",
            )
            store.save_document("student", document)
            source_document_ids = ("doc-1",)
        store.save_graph(
            "student",
            ConceptGraphProposal(
                graph_id="graph-1",
                concepts=(
                    ProposedConcept("a", "A"),
                    ProposedConcept("b", "B"),
                ),
                edges=(ProposedEdge("a", "b"),),
                source_document_ids=source_document_ids,
            ),
        )
        store.save_parcours_state("student", phase="not_started", graph_id="graph-1")
        return store

    def make_learning_concept(self) -> SQLiteStore:
        store = self.make_graph_concept()
        store.save_parcours_state("student", phase="learning", graph_id="graph-1")
        return store

    def write_json(self, name: str, payload) -> Path:
        return self._write_proposal(payload, name=name)

    @staticmethod
    def course_payload(concept_id: str = "a"):
        return {
            "concept_id": concept_id,
            "title": f"Cours {concept_id}",
            "content": "Contenu du cours.",
            "citations": [],
        }

    @staticmethod
    def question_payload(concept_id: str = "a", question_id: str = "q-1"):
        return {
            "question_id": question_id,
            "concept_id": concept_id,
            "prompt": "Question ?",
            "reference_answer": "Réponse.",
            "grading_rubric": "Barème.",
            "difficulty_coefficient": 0.8,
            "citations": [],
        }

    def test_save_course_stores_the_course(self):
        store = self.make_learning_concept()
        path = self.write_json("course.json", self.course_payload())

        code, payload = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 0)
        courses = store.list_courses("student")
        self.assertEqual(len(courses), 1)
        self.assertEqual(courses[0]["concept_id"], "a")
        self.assertEqual(payload["data"]["course_id"], courses[0]["course_id"])
        self.assertEqual(
            store.get_parcours_state("student")["current_concept_id"], "a"
        )

    def test_save_course_cites_a_document_of_the_graph(self):
        store = self.make_learning_concept()
        document = documents.store_upload(
            self.data_dir / "uploads",
            document_id="doc-1",
            title="Cours",
            filename="cours.pdf",
            content=b"%PDF-1.4 fake",
        )
        store.save_document("student", document)
        # The graph was built from doc-1 only when asked for it.
        store.save_graph(
            "student",
            ConceptGraphProposal(
                graph_id="graph-1",
                concepts=(ProposedConcept("a", "A"),),
                source_document_ids=("doc-1",),
            ),
        )
        payload = self.course_payload()
        payload["citations"] = [{"document_id": "doc-1", "page": 2, "quote": "extrait"}]
        path = self.write_json("course.json", payload)

        code, _ = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 0)
        course = store.list_courses("student")[0]
        self.assertEqual(store.get_course("student", course["course_id"])["citations"][0]["page"], 2)

    def test_save_course_rejects_an_unknown_cited_document(self):
        self.make_learning_concept()
        payload = self.course_payload()
        payload["citations"] = [{"document_id": "doc-inconnu", "page": 1}]
        path = self.write_json("course.json", payload)

        code, result = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("unknown document", result["error"])

    def test_save_course_rejects_a_concept_outside_the_graph(self):
        self.make_learning_concept()
        path = self.write_json("course.json", self.course_payload("z"))

        code, result = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("hors du graphe", result["error"])

    def test_save_course_rejects_a_concept_that_is_not_next(self):
        self.make_learning_concept()
        path = self.write_json("course.json", self.course_payload("b"))

        code, result = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("pas prévu maintenant", result["error"])

    def test_save_course_requires_the_learning_phase(self):
        self.make_graph_concept()
        path = self.write_json("course.json", self.course_payload())

        code, result = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("phase attendue 'learning'", result["error"])

    def test_save_course_requires_a_graph(self):
        self.make_concept()
        path = self.write_json("course.json", self.course_payload())

        code, result = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("aucun graphe", result["error"])

    def test_save_exercise_creates_an_assessment(self):
        store = self.make_learning_concept()
        path = self.write_json("exercise.json", self.question_payload())

        code, payload = self.run_command("save_exercise", "concept-1", str(path))

        self.assertEqual(code, 0)
        assessment_id = payload["data"]["assessment_id"]
        assessment = store.get_assessment(assessment_id)
        self.assertEqual(assessment["kind"], "exercise")
        self.assertEqual(assessment["questions"][0]["prompt"], "Question ?")
        state = store.get_parcours_state("student")
        self.assertEqual(state["current_assessment_id"], assessment_id)
        self.assertEqual(state["current_concept_id"], "a")

    def test_save_exercise_discards_only_unsaved_courses(self):
        store = self.make_learning_concept()
        store.save_course("student", "kept", "a", "Cours conservé", "Contenu", saved=True)
        store.save_course("student", "read", "a", "Cours lu", "Contenu")
        path = self.write_json("exercise.json", self.question_payload())

        code, _ = self.run_command("save_exercise", "concept-1", str(path))

        self.assertEqual(code, 0)
        remaining = {course["course_id"] for course in store.list_courses("student")}
        self.assertEqual(remaining, {"kept"})

    def test_save_exercise_rejects_a_difficulty_outside_the_unit_interval(self):
        self.make_learning_concept()
        payload = self.question_payload()
        payload["difficulty_coefficient"] = 1.5
        path = self.write_json("exercise.json", payload)

        code, result = self.run_command("save_exercise", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("difficulty_coefficient must be between 0 and 1", result["error"])

    def test_save_evaluation_stores_the_questions(self):
        store = self.make_graph_concept()
        payload = {
            "questions": [
                self.question_payload("a", "qa"),
                self.question_payload("b", "qb"),
            ]
        }
        path = self.write_json("evaluation.json", payload)

        code, result = self.run_command("save_evaluation", "concept-1", str(path))

        self.assertEqual(code, 0)
        self.assertEqual(result["data"]["questions"], 2)
        assessment = store.get_assessment(result["data"]["assessment_id"])
        self.assertEqual(assessment["kind"], "evaluation")
        self.assertEqual(len(assessment["questions"]), 2)
        self.assertEqual(store.get_parcours_state("student")["phase"], "initial_evaluation")

    def test_save_evaluation_rejects_an_unknown_concept(self):
        self.make_graph_concept()
        payload = {"questions": [self.question_payload("z", "qz")]}
        path = self.write_json("evaluation.json", payload)

        code, result = self.run_command("save_evaluation", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("concept inconnu", result["error"])

    def test_save_evaluation_requires_the_parcours_to_be_not_started(self):
        self.make_learning_concept()
        payload = {"questions": [self.question_payload()]}
        path = self.write_json("evaluation.json", payload)

        code, result = self.run_command("save_evaluation", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("phase attendue 'not_started'", result["error"])

    def test_save_final_exam_stores_the_questions(self):
        store = self.make_learning_concept()
        payload = {"questions": [self.question_payload("a", "fa")]}
        path = self.write_json("final-exam.json", payload)

        code, result = self.run_command("save_final_exam", "concept-1", str(path))

        self.assertEqual(code, 0)
        assessment = store.get_assessment(result["data"]["assessment_id"])
        self.assertEqual(assessment["kind"], "final_exam")
        self.assertEqual(store.get_parcours_state("student")["phase"], "final_exam")

    def test_activity_returns_the_current_questions(self):
        store = self.make_learning_concept()
        path = self.write_json("exercise.json", self.question_payload())
        code, _ = self.run_command("save_exercise", "concept-1", str(path))
        self.assertEqual(code, 0)

        code, payload = self.run_command("activity", "concept-1")

        self.assertEqual(code, 0)
        self.assertEqual(payload["data"]["phase"], "learning")
        self.assertEqual(payload["data"]["questions"][0]["question_id"], "q-1")

    def test_activity_reports_a_concept_that_has_not_started(self):
        self.make_concept()

        code, payload = self.run_command("activity", "concept-1")

        self.assertEqual(code, 0)
        self.assertEqual(payload["data"]["phase"], "not_started")
        self.assertEqual(payload["data"]["questions"], [])

    def test_submit_response_grades_the_exercise(self):
        store = self.make_learning_concept()
        store.save_parcours_state(
            "student", phase="learning", graph_id="graph-1", initial_gap=1.0
        )
        store.save_mastery(
            "student", "a", current_level=0.5, target_level=1.0, weight=1.0
        )
        path = self.write_json("exercise.json", self.question_payload())
        code, _ = self.run_command("save_exercise", "concept-1", str(path))
        self.assertEqual(code, 0)
        grade = {
            "responses": [
                {
                    "question_id": "q-1",
                    "response_text": "ma réponse",
                    "score": 1.0,
                    "assistance_coefficient": 1.0,
                    "feedback": "Bien",
                }
            ]
        }
        grade_path = self.write_json("grade.json", grade)

        code, payload = self.run_command(
            "submit_response", "concept-1", str(grade_path)
        )

        self.assertEqual(code, 0)
        # eta * a * d = 0.3 * 1.0 * 0.8, applied to the 0.5 level.
        self.assertAlmostEqual(payload["data"]["level"], 0.62)
        mastery = {item["concept_id"]: item for item in store.get_mastery("student")}
        self.assertAlmostEqual(mastery["a"]["current_level"], 0.62)
        self.assertIsNone(
            store.get_parcours_state("student")["current_assessment_id"]
        )

    def test_submit_response_grades_the_initial_evaluation(self):
        store = self.make_graph_concept()
        payload = {
            "questions": [
                self.question_payload("a", "qa"),
                self.question_payload("b", "qb"),
            ]
        }
        path = self.write_json("evaluation.json", payload)
        code, _ = self.run_command("save_evaluation", "concept-1", str(path))
        self.assertEqual(code, 0)
        grade = {
            "responses": [
                {
                    "question_id": "qa",
                    "response_text": "r",
                    "score": 1.0,
                    "assistance_coefficient": 1.0,
                    "feedback": "OK",
                },
                {
                    "question_id": "qb",
                    "response_text": "r",
                    "score": 0.0,
                    "assistance_coefficient": 1.0,
                    "feedback": "À revoir",
                },
            ]
        }
        grade_path = self.write_json("grade.json", grade)

        code, payload = self.run_command(
            "submit_response", "concept-1", str(grade_path)
        )

        self.assertEqual(code, 0)
        # The initial evaluation uses eta=1.0 but keeps the question difficulty.
        self.assertEqual(payload["data"]["levels"]["a"], 0.8)
        self.assertEqual(payload["data"]["levels"]["b"], 0.0)
        self.assertEqual(
            store.get_parcours_state("student")["phase"], "learning"
        )

    def test_submit_response_grades_the_final_exam(self):
        store = self.make_learning_concept()
        store.save_parcours_state(
            "student", phase="learning", graph_id="graph-1", initial_gap=1.0
        )
        for concept_id in ("a", "b"):
            store.save_mastery(
                "student",
                concept_id,
                current_level=1.0,
                target_level=1.0,
                weight=1.0,
            )
        payload = {"questions": [self.question_payload("a", "fa")]}
        path = self.write_json("final-exam.json", payload)
        code, _ = self.run_command("save_final_exam", "concept-1", str(path))
        self.assertEqual(code, 0)
        grade = {
            "responses": [
                {
                    "question_id": "fa",
                    "response_text": "r",
                    "score": 1.0,
                    "assistance_coefficient": 1.0,
                    "feedback": "Bien",
                }
            ]
        }
        grade_path = self.write_json("grade.json", grade)

        code, payload = self.run_command(
            "submit_response", "concept-1", str(grade_path)
        )

        self.assertEqual(code, 0)
        self.assertTrue(payload["data"]["sufficient"])
        self.assertEqual(
            store.get_parcours_state("student")["phase"], "completed"
        )

    def test_submit_response_requires_an_assessment_in_progress(self):
        self.make_graph_concept()
        grade = {
            "responses": [
                {"question_id": "q-1", "response_text": "r", "score": 1.0}
            ]
        }
        grade_path = self.write_json("grade.json", grade)

        code, payload = self.run_command(
            "submit_response", "concept-1", str(grade_path)
        )

        self.assertEqual(code, 1)
        self.assertIn("aucune évaluation", payload["error"])

    def test_submit_response_rejects_a_wrong_question(self):
        store = self.make_learning_concept()
        path = self.write_json("exercise.json", self.question_payload())
        self.run_command("save_exercise", "concept-1", str(path))
        grade = {
            "responses": [
                {"question_id": "autre", "response_text": "r", "score": 1.0}
            ]
        }
        grade_path = self.write_json("grade.json", grade)

        code, payload = self.run_command(
            "submit_response", "concept-1", str(grade_path)
        )

        self.assertEqual(code, 1)
        self.assertIn("question attendue", payload["error"])

    def test_submit_response_rejects_an_invalid_grade(self):
        store = self.make_learning_concept()
        path = self.write_json("exercise.json", self.question_payload())
        self.run_command("save_exercise", "concept-1", str(path))
        grade = {
            "responses": [
                {"question_id": "q-1", "response_text": "r", "score": 2.0}
            ]
        }
        grade_path = self.write_json("grade.json", grade)

        code, payload = self.run_command(
            "submit_response", "concept-1", str(grade_path)
        )

        self.assertEqual(code, 1)
        self.assertIn("score must be between 0 and 1", payload["error"])

    def test_missing_file_returns_an_error(self):
        self.make_learning_concept()
        path = self.data_dir / "agent" / "absent.json"

        code, result = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("fichier illisible", result["error"])

    def test_invalid_json_returns_an_error(self):
        self.make_learning_concept()
        path = self.data_dir / "agent" / "broken.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{pas du json", encoding="utf-8")

        code, result = self.run_command("save_course", "concept-1", str(path))

        self.assertEqual(code, 1)
        self.assertIn("JSON invalide", result["error"])


if __name__ == "__main__":
    unittest.main()
