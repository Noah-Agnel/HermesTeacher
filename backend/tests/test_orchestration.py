import tempfile
import unittest
from pathlib import Path

try:
    from backend.tutor import documents, orchestration, storage
except ImportError:
    documents = None
    orchestration = None
    storage = None


GRAPH = {
    "graph_id": "graph-1",
    "source_document_ids": ["doc-1"],
    "notes": "",
    "concepts": [
        {
            "concept_id": "a",
            "label": "Concept A",
            "target_level": 1.0,
            "weight": 1.0,
            "evidence": [{"document_id": "doc-1", "passage_id": "doc-1:0000"}],
        },
        {
            "concept_id": "b",
            "label": "Concept B",
            "target_level": 1.0,
            "weight": 2.0,
            "evidence": [{"document_id": "doc-1", "passage_id": "doc-1:0001"}],
        },
    ],
    "edges": [{"prerequisite_id": "a", "concept_id": "b", "evidence": []}],
}

INITIAL_EVALUATION = {
    "questions": [
        {
            "question_id": "q1",
            "concept_id": "a",
            "prompt": "Question 1",
            "reference_answer": "Réponse 1",
            "grading_rubric": "Barème 1",
            "difficulty_coefficient": 1.0,
            "citations": [{"document_id": "doc-1", "passage_id": "doc-1:0000"}],
        },
        {
            "question_id": "q2",
            "concept_id": "b",
            "prompt": "Question 2",
            "reference_answer": "Réponse 2",
            "grading_rubric": "Barème 2",
            "difficulty_coefficient": 1.0,
            "citations": [{"document_id": "doc-1", "passage_id": "doc-1:0001"}],
        },
    ]
}

COURSE = {
    "concept_id": "a",
    "title": "Cours A",
    "content": "Contenu du cours A.",
    "citations": [{"document_id": "doc-1", "passage_id": "doc-1:0000"}],
}

EXERCISE = {
    "question_id": "q3",
    "concept_id": "a",
    "prompt": "Question 3",
    "reference_answer": "Réponse 3",
    "grading_rubric": "Barème 3",
    "difficulty_coefficient": 1.0,
    "citations": [{"document_id": "doc-1", "passage_id": "doc-1:0000"}],
}

FINAL_EXAM = {
    "questions": [
        {
            "question_id": "q4",
            "concept_id": "a",
            "prompt": "Question 4",
            "reference_answer": "Réponse 4",
            "grading_rubric": "Barème 4",
            "difficulty_coefficient": 1.0,
            "citations": [{"document_id": "doc-1", "passage_id": "doc-1:0000"}],
        },
        {
            "question_id": "q5",
            "concept_id": "b",
            "prompt": "Question 5",
            "reference_answer": "Réponse 5",
            "grading_rubric": "Barème 5",
            "difficulty_coefficient": 1.0,
            "citations": [{"document_id": "doc-1", "passage_id": "doc-1:0001"}],
        },
    ]
}

GRADES = {
    "q1": {"score": 0.8, "assistance_coefficient": 1.0, "feedback": "OK"},
    "q2": {"score": 0.4, "assistance_coefficient": 1.0, "feedback": "À revoir"},
    "q3": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "Bien"},
    "q4": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "Bien"},
    "q5": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "Bien"},
}


class ScriptedGenerator:
    """Deterministic stand-in for the LLM agent; records every call."""

    def __init__(self, graph=None, grades=None):
        self.graph = GRAPH if graph is None else graph
        self.grades = GRADES if grades is None else grades
        self.calls = []

    def propose_graph(self, passages, objectives, preferences=None):
        self.calls.append(("propose_graph", objectives, len(passages)))
        return self.graph

    def generate_initial_evaluation(self, concepts, passages, objectives, preferences=None):
        self.calls.append(("generate_initial_evaluation", objectives, len(concepts)))
        return INITIAL_EVALUATION

    def generate_course(self, concept, passages, current_level, preferences=None):
        self.calls.append(("generate_course", concept["concept_id"], current_level))
        return COURSE

    def generate_exercise(self, concept, passages, current_level, preferences=None):
        self.calls.append(("generate_exercise", concept["concept_id"], current_level))
        return EXERCISE

    def generate_final_exam(self, concepts, passages, objectives, preferences=None):
        self.calls.append(("generate_final_exam", objectives, len(concepts)))
        return FINAL_EXAM

    def grade_response(self, question, response_text):
        self.calls.append(("grade_response", question["question_id"], response_text))
        return self.grades[question["question_id"]]


class ParcoursTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "learner.sqlite3"
        self.store = storage.SQLiteStore(self.database_path)
        self.store.create_learner("student-1")
        self.document = documents.load_text(
            "# Concept A\n\nContenu A.\n\n# Concept B\n\nContenu B.",
            document_id="doc-1",
            title="Cours",
            media_type="text/markdown",
        )
        self.generator = ScriptedGenerator()
        self.parcours = orchestration.Parcours(
            self.store, "student-1", self.generator
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def start(self):
        return self.parcours.start([self.document], "Maîtriser la matière")

    def test_start_proposes_graph_and_creates_initial_evaluation(self):
        result = self.start()

        state = self.store.get_parcours_state("student-1")
        self.assertEqual(state["phase"], "initial_evaluation")
        self.assertEqual(state["graph_id"], "graph-1")
        self.assertEqual(state["current_assessment_id"], result["assessment_id"])
        self.assertEqual(state["objectives"], "Maîtriser la matière")

        graph = self.store.get_graph("student-1", "graph-1")
        self.assertEqual([c["concept_id"] for c in graph["concepts"]], ["a", "b"])
        assessment = self.store.get_assessment(result["assessment_id"])
        self.assertEqual(assessment["kind"], "evaluation")
        self.assertEqual(len(assessment["questions"]), 2)
        self.assertEqual(self.generator.calls[0][0], "propose_graph")

    def test_start_rejects_an_invalid_graph_proposal(self):
        cyclic = {
            "graph_id": "graph-1",
            "concepts": [
                {"concept_id": "a", "label": "A"},
                {"concept_id": "b", "label": "B"},
            ],
            "edges": [
                {"prerequisite_id": "a", "concept_id": "b"},
                {"prerequisite_id": "b", "concept_id": "a"},
            ],
        }
        self.generator.graph = cyclic

        with self.assertRaises(ValueError):
            self.start()

    def test_submit_initial_evaluation_computes_levels_and_gap(self):
        self.start()

        result = self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )

        self.assertEqual(result["levels"]["a"], 0.8)
        self.assertEqual(result["levels"]["b"], 0.4)
        self.assertAlmostEqual(result["gap"], 1.4)

        mastery = {
            item["concept_id"]: item for item in self.store.get_mastery("student-1")
        }
        self.assertEqual(mastery["a"]["current_level"], 0.8)
        self.assertEqual(mastery["a"]["target_level"], 1.0)
        self.assertEqual(mastery["b"]["weight"], 2.0)

        state = self.store.get_parcours_state("student-1")
        self.assertEqual(state["phase"], "learning")
        self.assertAlmostEqual(state["initial_gap"], 1.4)
        self.assertIsNone(state["current_assessment_id"])

    def test_next_activity_selects_the_next_concept_and_generates_content(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )

        activity = self.parcours.next_activity()

        self.assertEqual(activity["type"], "activity")
        self.assertEqual(activity["concept_id"], "a")
        self.assertEqual(activity["course"]["title"], "Cours A")
        self.assertEqual(activity["exercise"]["question_id"], "q3")
        self.assertIn(("generate_course", "a", 0.8), self.generator.calls)
        self.assertIn(("generate_exercise", "a", 0.8), self.generator.calls)

        state = self.store.get_parcours_state("student-1")
        self.assertEqual(state["current_concept_id"], "a")
        self.assertEqual(
            state["current_assessment_id"], activity["assessment_id"]
        )
        courses = self.store.list_courses("student-1", concept_id="a")
        self.assertEqual(len(courses), 1)

    def test_submit_exercise_updates_level_and_records_progress(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        activity = self.parcours.next_activity()

        result = self.parcours.submit_exercise("ma réponse")

        self.assertAlmostEqual(result["level"], 0.86)
        self.assertAlmostEqual(result["gap"], 1.34)
        self.assertAlmostEqual(result["progress"], (1.4 - 1.34) / 1.4)
        self.assertEqual(result["feedback"], "Bien")

        mastery = {
            item["concept_id"]: item for item in self.store.get_mastery("student-1")
        }
        self.assertAlmostEqual(mastery["a"]["current_level"], 0.86)
        history = self.store.get_progress_history("student-1")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["assessment_id"], activity["assessment_id"])

        state = self.store.get_parcours_state("student-1")
        self.assertIsNone(state["current_assessment_id"])

    def test_next_activity_returns_final_exam_when_all_concepts_are_at_target(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        for concept_id in ("a", "b"):
            self.store.save_mastery(
                "student-1",
                concept_id,
                current_level=1.0,
                target_level=1.0,
                weight=1.0,
            )

        activity = self.parcours.next_activity()

        self.assertEqual(activity["type"], "final_exam")

    def test_final_exam_marks_the_parcours_completed_when_sufficient(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        for concept_id in ("a", "b"):
            self.store.save_mastery(
                "student-1",
                concept_id,
                current_level=1.0,
                target_level=1.0,
                weight=1.0,
            )
        self.parcours.next_activity()

        exam = self.parcours.start_final_exam()
        self.assertEqual(len(exam["questions"]), 2)

        result = self.parcours.submit_final_exam(
            [
                {"question_id": "q4", "response_text": "réponse 4"},
                {"question_id": "q5", "response_text": "réponse 5"},
            ]
        )

        self.assertTrue(result["sufficient"])
        self.assertAlmostEqual(result["levels"]["a"], 1.0)
        state = self.store.get_parcours_state("student-1")
        self.assertEqual(state["phase"], "completed")

    def test_insufficient_final_exam_flags_remediation(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        self.parcours.next_activity()

        self.parcours.start_final_exam()
        result = self.parcours.submit_final_exam(
            [
                {"question_id": "q4", "response_text": "réponse 4"},
                {"question_id": "q5", "response_text": "réponse 5"},
            ]
        )

        self.assertFalse(result["sufficient"])

        remediation = self.parcours.remediate()

        self.assertIn("a", remediation["concepts"])
        self.assertIn("b", remediation["concepts"])

    def test_state_survives_reopening(self):
        self.start()

        reopened_store = storage.SQLiteStore(self.database_path)
        state = reopened_store.get_parcours_state("student-1")

        self.assertEqual(state["phase"], "initial_evaluation")
        self.assertEqual(state["graph_id"], "graph-1")

    def test_unknown_citation_is_rejected(self):
        invalid_exercise = dict(EXERCISE)
        invalid_exercise["citations"] = [
            {"document_id": "doc-1", "passage_id": "doc-1:9999"}
        ]
        generator = ScriptedGenerator()
        generator.generate_exercise = lambda *args, **kwargs: invalid_exercise
        parcours = orchestration.Parcours(self.store, "student-1", generator)
        parcours.start([self.document], "objectifs")
        parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )

        with self.assertRaises(ValueError):
            parcours.next_activity()

    def test_start_without_documents_supports_an_isolated_concept(self):
        graph = {
            "graph_id": "graph-1",
            "concepts": [{"concept_id": "a", "label": "Concept A"}],
            "edges": [],
        }
        self.generator.graph = graph
        self.generator.generate_initial_evaluation = lambda *args, **kwargs: {
            "questions": [
                {
                    "question_id": "q1",
                    "concept_id": "a",
                    "prompt": "Question 1",
                    "reference_answer": "Réponse 1",
                    "grading_rubric": "Barème 1",
                    "difficulty_coefficient": 1.0,
                    "citations": [],
                }
            ]
        }

        result = self.parcours.start([], "Apprendre le concept A")

        self.assertEqual(result["graph_id"], "graph-1")
        state = self.store.get_parcours_state("student-1")
        self.assertEqual(state["phase"], "initial_evaluation")


if __name__ == "__main__":
    unittest.main()
