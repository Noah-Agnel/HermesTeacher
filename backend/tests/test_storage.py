import tempfile
import unittest
from pathlib import Path

try:
    from backend.tutor import storage
except ImportError:
    storage = None

try:
    from backend.tutor import documents
except ImportError:
    documents = None

try:
    from backend.tutor import graph
except ImportError:
    graph = None


class SQLiteStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "learner.sqlite3"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_store(self):
        store_type = getattr(storage, "SQLiteStore", None)
        self.assertIsNotNone(store_type, "SQLiteStore should be implemented")
        return store_type(self.database_path)

    def test_learner_profile_and_preferences_survive_reopening(self):
        store = self.make_store()
        store.create_learner(
            learner_id="student-1",
            display_name="Étudiant",
            preferences={"format": "examples", "daily_minutes": 20},
        )

        reopened_store = self.make_store()
        learner = reopened_store.get_learner("student-1")

        self.assertEqual(learner["display_name"], "Étudiant")
        self.assertEqual(
            learner["preferences"],
            {"format": "examples", "daily_minutes": 20},
        )

    def test_assessment_questions_responses_and_result_survive_reopening(self):
        store = self.make_store()
        required_methods = (
            "create_assessment",
            "add_question",
            "record_response",
            "complete_assessment",
            "get_assessment",
        )
        for name in required_methods:
            self.assertTrue(callable(getattr(store, name, None)), f"{name} should be implemented")

        store.create_learner("student-1")
        store.create_assessment(
            assessment_id="assessment-1",
            learner_id="student-1",
            kind="exercise",
            source_metadata={"reference": "td-1.pdf"},
        )
        store.add_question(
            assessment_id="assessment-1",
            question_id="question-1",
            position=0,
            concept_id="concept-opaque-id",
            prompt="Résoudre l'équation.",
            reference_answer="x = 2",
            grading_rubric="Une réponse exacte vaut 1.",
            difficulty_coefficient=0.8,
        )
        store.record_response(
            response_id="response-1",
            question_id="question-1",
            response_text="x=2",
            score=0.75,
            assistance_coefficient=0.4,
        )
        store.record_response(
            response_id="response-2",
            question_id="question-1",
            response_text="x = 2",
            score=1.0,
            assistance_coefficient=1.0,
        )
        store.complete_assessment("assessment-1", raw_score=1.0)

        reopened_store = self.make_store()
        assessment = reopened_store.get_assessment("assessment-1")

        self.assertEqual(assessment["status"], "completed")
        self.assertEqual(assessment["raw_score"], 1.0)
        self.assertEqual(assessment["source_metadata"], {"reference": "td-1.pdf"})
        question = assessment["questions"][0]
        self.assertEqual(question["concept_id"], "concept-opaque-id")
        self.assertEqual(question["difficulty_coefficient"], 0.8)
        self.assertEqual(len(question["responses"]), 2)
        self.assertEqual(question["responses"][0]["assistance_coefficient"], 0.4)

    def test_mastery_and_progress_history_survive_reopening(self):
        store = self.make_store()
        required_methods = ("save_mastery", "get_mastery", "record_progress", "get_progress_history")
        for name in required_methods:
            self.assertTrue(callable(getattr(store, name, None)), f"{name} should be implemented")

        store.create_learner("student-1")
        store.save_mastery(
            learner_id="student-1",
            concept_id="concept-opaque-id",
            concept_label="Intégrales",
            current_level=0.72,
            target_level=0.9,
            weight=2.0,
        )
        snapshot_id = store.record_progress(
            learner_id="student-1",
            initial_gap=1.6,
            current_gap=0.4,
        )

        reopened_store = self.make_store()
        mastery = reopened_store.get_mastery("student-1")
        history = reopened_store.get_progress_history("student-1")

        self.assertEqual(len(mastery), 1)
        self.assertEqual(mastery[0]["concept_id"], "concept-opaque-id")
        self.assertEqual(mastery[0]["current_level"], 0.72)
        self.assertEqual(mastery[0]["weight"], 2.0)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["snapshot_id"], snapshot_id)
        self.assertAlmostEqual(history[0]["progress_fraction"], 0.75)

    def test_in_progress_assessment_can_be_resumed_after_reopening(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.create_assessment(
            assessment_id="assessment-1",
            learner_id="student-1",
            kind="evaluation",
        )
        store.add_question(
            assessment_id="assessment-1",
            question_id="question-1",
            position=0,
            prompt="Résoudre l'équation.",
        )
        store.record_response(
            response_id="response-1",
            question_id="question-1",
            response_text="x = 1",
            score=0.5,
        )

        reopened_store = self.make_store()
        assessment = reopened_store.get_assessment("assessment-1")
        self.assertEqual(assessment["status"], "in_progress")

        attempt_number = reopened_store.record_response(
            response_id="response-2",
            question_id="question-1",
            response_text="x = 2",
            score=1.0,
        )
        self.assertEqual(attempt_number, 2)
        reopened_store.complete_assessment("assessment-1", raw_score=1.0)

        final_store = self.make_store()
        final_assessment = final_store.get_assessment("assessment-1")
        self.assertEqual(final_assessment["status"], "completed")
        self.assertEqual(len(final_assessment["questions"][0]["responses"]), 2)

    def test_invalid_kinds_coefficients_and_levels_are_rejected(self):
        store = self.make_store()
        store.create_learner("student-1")

        with self.assertRaises(ValueError):
            store.create_assessment("assessment-1", "student-1", kind="quiz")
        with self.assertRaises(ValueError):
            store.create_assessment("", "student-1", kind="exercise")

        store.create_assessment("assessment-1", "student-1", kind="exercise")
        with self.assertRaises(ValueError):
            store.add_question(
                "assessment-1", "question-1", position=-1, prompt="Prompt"
            )
        with self.assertRaises(ValueError):
            store.add_question(
                "assessment-1",
                "question-1",
                position=0,
                prompt="Prompt",
                difficulty_coefficient=1.5,
            )
        store.add_question(
            "assessment-1", "question-1", position=0, prompt="Prompt"
        )

        with self.assertRaises(ValueError):
            store.record_response("response-1", "question-1", "text", score=1.5)
        with self.assertRaises(ValueError):
            store.record_response(
                "response-1",
                "question-1",
                "text",
                score=0.5,
                assistance_coefficient=-0.1,
            )
        with self.assertRaises(ValueError):
            store.save_mastery(
                "student-1", "concept-1", current_level=0.5, weight=-1.0
            )
        with self.assertRaises(ValueError):
            store.record_progress("student-1", initial_gap=-1.0, current_gap=0.0)

    def test_completed_assessment_rejects_new_questions_and_responses(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.create_assessment("assessment-1", "student-1", kind="exercise")
        store.add_question(
            "assessment-1", "question-1", position=0, prompt="Prompt"
        )
        store.record_response("response-1", "question-1", "text", score=1.0)
        store.complete_assessment("assessment-1", raw_score=1.0)

        with self.assertRaises(ValueError):
            store.add_question(
                "assessment-1", "question-2", position=1, prompt="Prompt"
            )
        with self.assertRaises(ValueError):
            store.record_response("response-2", "question-1", "text", score=1.0)
        with self.assertRaises(ValueError):
            store.complete_assessment("assessment-1", raw_score=1.0)

    def test_unknown_identifiers_are_rejected_or_return_empty(self):
        store = self.make_store()
        self.assertIsNone(store.get_learner("missing"))
        self.assertEqual(store.get_mastery("missing"), [])
        self.assertEqual(store.get_progress_history("missing"), [])
        with self.assertRaises(ValueError):
            store.add_question(
                "missing-assessment", "question-1", position=0, prompt="Prompt"
            )
        with self.assertRaises(ValueError):
            store.record_response(
                "response-1", "missing-question", "text", score=0.5
            )


class DocumentStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "learner.sqlite3"
        self.assertIsNotNone(documents, "documents module should be implemented")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_store(self):
        store_type = getattr(storage, "SQLiteStore", None)
        self.assertIsNotNone(store_type, "SQLiteStore should be implemented")
        return store_type(self.database_path)

    def test_document_and_passages_survive_reopening(self):
        store = self.make_store()
        store.create_learner("student-1")
        document = documents.load_text(
            "# Titre\n\nContenu du passage.",
            document_id="doc-1",
            title="Cours",
            media_type="text/markdown",
        )

        store.save_document("student-1", document)
        reopened_store = self.make_store()
        fetched = reopened_store.get_document("student-1", "doc-1")

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["title"], "Cours")
        self.assertEqual(fetched["media_type"], "text/markdown")
        self.assertEqual(len(fetched["passages"]), 1)
        self.assertEqual(fetched["passages"][0]["passage_id"], "doc-1:0000")
        self.assertEqual(fetched["passages"][0]["heading"], "Titre")

    def test_list_documents_reports_passage_counts(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_document(
            "student-1",
            documents.load_text("A.\n\nB.", document_id="doc-1", title="A"),
        )
        store.save_document(
            "student-1",
            documents.load_text("C.", document_id="doc-2", title="B"),
        )

        listed = store.list_documents("student-1")

        self.assertEqual(
            {item["document_id"]: item["passage_count"] for item in listed},
            {"doc-1": 2, "doc-2": 1},
        )

    def test_get_document_is_scoped_to_the_learner(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.create_learner("student-2")
        store.save_document(
            "student-1",
            documents.load_text("A.", document_id="doc-1", title="A"),
        )

        self.assertIsNone(store.get_document("student-2", "doc-1"))

    def test_saving_a_document_again_replaces_it(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_document(
            "student-1",
            documents.load_text("A.\n\nB.", document_id="doc-1", title="A"),
        )
        store.save_document(
            "student-1",
            documents.load_text("C.", document_id="doc-1", title="A"),
        )

        fetched = store.get_document("student-1", "doc-1")

        self.assertEqual(len(fetched["passages"]), 1)
        self.assertEqual(fetched["passages"][0]["text"], "C.")


class GraphStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "learner.sqlite3"
        self.assertIsNotNone(graph, "graph module should be implemented")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_store(self):
        store_type = getattr(storage, "SQLiteStore", None)
        self.assertIsNotNone(store_type, "SQLiteStore should be implemented")
        return store_type(self.database_path)

    def make_proposal(self):
        return graph.ConceptGraphProposal(
            graph_id="graph-1",
            concepts=(
                graph.ProposedConcept(
                    "a",
                    "A",
                    description="Description",
                    target_level=0.8,
                    weight=2.0,
                    evidence=(graph.Evidence("doc-1", "doc-1:0000", "quote"),),
                ),
                graph.ProposedConcept("b", "B"),
            ),
            edges=(
                graph.ProposedEdge(
                    "a",
                    "b",
                    evidence=(graph.Evidence("doc-1", "doc-1:0000"),),
                ),
            ),
            source_document_ids=("doc-1",),
            notes="proposé",
        )

    def test_graph_concepts_edges_and_evidence_survive_reopening(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_graph("student-1", self.make_proposal(), status="proposed")

        reopened_store = self.make_store()
        fetched = reopened_store.get_graph("student-1", "graph-1")

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["status"], "proposed")
        self.assertEqual(fetched["notes"], "proposé")
        self.assertEqual(fetched["source_document_ids"], ["doc-1"])
        concept_a = next(
            concept for concept in fetched["concepts"] if concept["concept_id"] == "a"
        )
        self.assertEqual(concept_a["label"], "A")
        self.assertEqual(concept_a["target_level"], 0.8)
        self.assertEqual(concept_a["weight"], 2.0)
        self.assertEqual(concept_a["evidence"][0]["passage_id"], "doc-1:0000")
        self.assertEqual(concept_a["evidence"][0]["quote"], "quote")
        edge = fetched["edges"][0]
        self.assertEqual((edge["prerequisite_id"], edge["concept_id"]), ("a", "b"))
        self.assertEqual(edge["evidence"][0]["passage_id"], "doc-1:0000")

    def test_list_graphs_reports_counts(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_graph("student-1", self.make_proposal())

        listed = store.list_graphs("student-1")

        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["graph_id"], "graph-1")
        self.assertEqual(listed[0]["status"], "proposed")
        self.assertEqual(listed[0]["concept_count"], 2)
        self.assertEqual(listed[0]["edge_count"], 1)

    def test_set_graph_status(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_graph("student-1", self.make_proposal(), status="proposed")

        store.set_graph_status("student-1", "graph-1", "validated")
        reopened_store = self.make_store()
        fetched = reopened_store.get_graph("student-1", "graph-1")

        self.assertEqual(fetched["status"], "validated")

    def test_set_graph_status_rejects_unknown_status(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_graph("student-1", self.make_proposal())

        with self.assertRaises(ValueError):
            store.set_graph_status("student-1", "graph-1", "unknown")

    def test_get_graph_is_scoped_to_the_learner(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.create_learner("student-2")
        store.save_graph("student-1", self.make_proposal())

        self.assertIsNone(store.get_graph("student-2", "graph-1"))

    def test_saving_a_graph_again_replaces_it(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_graph("student-1", self.make_proposal())

        replacement = graph.ConceptGraphProposal(
            graph_id="graph-1",
            concepts=(graph.ProposedConcept("c", "C"),),
        )
        store.save_graph("student-1", replacement, status="corrected")

        fetched = store.get_graph("student-1", "graph-1")

        self.assertEqual(fetched["status"], "corrected")
        self.assertEqual([concept["concept_id"] for concept in fetched["concepts"]], ["c"])
        self.assertEqual(fetched["edges"], [])


class CourseStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "learner.sqlite3"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_store(self):
        store_type = getattr(storage, "SQLiteStore", None)
        self.assertIsNotNone(store_type, "SQLiteStore should be implemented")
        return store_type(self.database_path)

    def test_course_and_citations_survive_reopening(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_course(
            "student-1",
            course_id="course-1",
            concept_id="a",
            title="Cours",
            content="Contenu du cours.",
            citations=[
                {"document_id": "doc-1", "passage_id": "doc-1:0000", "quote": "extrait"}
            ],
        )

        reopened_store = self.make_store()
        fetched = reopened_store.get_course("student-1", "course-1")

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["title"], "Cours")
        self.assertEqual(fetched["content"], "Contenu du cours.")
        self.assertEqual(
            fetched["citations"],
            [{"document_id": "doc-1", "passage_id": "doc-1:0000", "quote": "extrait"}],
        )

    def test_list_courses_can_filter_by_concept(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_course("student-1", course_id="course-1", concept_id="a", title="A")
        store.save_course("student-1", course_id="course-2", concept_id="b", title="B")

        listed = store.list_courses("student-1", concept_id="a")

        self.assertEqual([item["course_id"] for item in listed], ["course-1"])

    def test_get_course_is_scoped_to_the_learner(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.create_learner("student-2")
        store.save_course("student-1", course_id="course-1", concept_id="a", title="A")

        self.assertIsNone(store.get_course("student-2", "course-1"))


class CitationAndFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "learner.sqlite3"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_store(self):
        store_type = getattr(storage, "SQLiteStore", None)
        self.assertIsNotNone(store_type, "SQLiteStore should be implemented")
        return store_type(self.database_path)

    def test_question_citations_survive_reopening(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.create_assessment("assessment-1", "student-1", kind="exercise")
        store.add_question(
            "assessment-1",
            "question-1",
            position=0,
            prompt="Prompt",
            citations=[
                {"document_id": "doc-1", "passage_id": "doc-1:0000", "quote": "extrait"}
            ],
        )

        reopened_store = self.make_store()
        assessment = reopened_store.get_assessment("assessment-1")

        self.assertEqual(
            assessment["questions"][0]["citations"],
            [{"document_id": "doc-1", "passage_id": "doc-1:0000", "quote": "extrait"}],
        )

    def test_response_feedback_survives_reopening(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.create_assessment("assessment-1", "student-1", kind="exercise")
        store.add_question("assessment-1", "question-1", position=0, prompt="Prompt")
        store.record_response(
            "response-1",
            "question-1",
            "réponse",
            score=0.5,
            feedback="Presque, regarde la définition.",
        )

        reopened_store = self.make_store()
        assessment = reopened_store.get_assessment("assessment-1")

        self.assertEqual(
            assessment["questions"][0]["responses"][0]["feedback"],
            "Presque, regarde la définition.",
        )


class ParcoursStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "learner.sqlite3"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_store(self):
        store_type = getattr(storage, "SQLiteStore", None)
        self.assertIsNotNone(store_type, "SQLiteStore should be implemented")
        return store_type(self.database_path)

    def test_state_survives_reopening(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_parcours_state(
            "student-1",
            phase="learning",
            graph_id="graph-1",
            initial_gap=1.5,
            current_assessment_id="assessment-1",
            current_concept_id="a",
        )

        reopened_store = self.make_store()
        state = reopened_store.get_parcours_state("student-1")

        self.assertIsNotNone(state)
        self.assertEqual(state["phase"], "learning")
        self.assertEqual(state["graph_id"], "graph-1")
        self.assertEqual(state["initial_gap"], 1.5)
        self.assertEqual(state["current_assessment_id"], "assessment-1")
        self.assertEqual(state["current_concept_id"], "a")

    def test_state_is_none_initially(self):
        store = self.make_store()
        store.create_learner("student-1")

        self.assertIsNone(store.get_parcours_state("student-1"))

    def test_saving_state_again_updates_it(self):
        store = self.make_store()
        store.create_learner("student-1")
        store.save_parcours_state("student-1", phase="initial_evaluation")
        store.save_parcours_state("student-1", phase="completed")

        self.assertEqual(store.get_parcours_state("student-1")["phase"], "completed")


if __name__ == "__main__":
    unittest.main()
