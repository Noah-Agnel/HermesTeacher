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
            "evidence": [{"document_id": "doc-1", "page": 1}],
        },
        {
            "concept_id": "b",
            "label": "Concept B",
            "target_level": 1.0,
            "weight": 2.0,
            "evidence": [{"document_id": "doc-1", "page": 1}],
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
            "citations": [{"document_id": "doc-1", "page": 1}],
        },
        {
            "question_id": "q2",
            "concept_id": "b",
            "prompt": "Question 2",
            "reference_answer": "Réponse 2",
            "grading_rubric": "Barème 2",
            "difficulty_coefficient": 1.0,
            "citations": [{"document_id": "doc-1", "page": 1}],
        },
    ]
}

COURSE = {
    "concept_id": "a",
    "title": "Cours A",
    "content": "Contenu du cours A.",
    "citations": [{"document_id": "doc-1", "page": 1}],
}

EXERCISE = {
    "question_id": "q3",
    "concept_id": "a",
    "prompt": "Question 3",
    "reference_answer": "Réponse 3",
    "grading_rubric": "Barème 3",
    "difficulty_coefficient": 1.0,
    "citations": [{"document_id": "doc-1", "page": 1}],
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
            "citations": [{"document_id": "doc-1", "page": 1}],
        },
        {
            "question_id": "q5",
            "concept_id": "b",
            "prompt": "Question 5",
            "reference_answer": "Réponse 5",
            "grading_rubric": "Barème 5",
            "difficulty_coefficient": 1.0,
            "citations": [{"document_id": "doc-1", "page": 1}],
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
        self.document = documents.Document(
            document_id="doc-1",
            title="Cours",
            path="/tmp/cours.pdf",
            media_type="application/pdf",
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

    def test_submit_exercise_accepts_the_agent_grade(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        self.parcours.next_activity()
        self.generator.calls.clear()
        agent_parcours = orchestration.Parcours(self.store, "student-1")

        result = agent_parcours.submit_exercise(
            "ma réponse",
            {"score": 1.0, "assistance_coefficient": 0.5, "feedback": "Avec aide"},
        )

        # eta * a * d = 0.3 * 0.5 * 1.0, applied to the 0.8 level.
        self.assertAlmostEqual(result["level"], 0.83)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["feedback"], "Avec aide")
        self.assertNotIn(
            "grade_response", [call[0] for call in self.generator.calls]
        )

    def test_submit_exercise_without_grade_or_generator_raises(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        self.parcours.next_activity()
        agent_parcours = orchestration.Parcours(self.store, "student-1")

        with self.assertRaises(ValueError):
            agent_parcours.submit_exercise("ma réponse")

    def test_submit_initial_evaluation_accepts_the_agent_grades(self):
        self.start()

        result = self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ],
            {
                "q1": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "OK"},
                "q2": {"score": 0.0, "assistance_coefficient": 1.0, "feedback": "À revoir"},
            },
        )

        self.assertEqual(result["levels"]["a"], 1.0)
        self.assertEqual(result["levels"]["b"], 0.0)
        self.assertNotIn(
            "grade_response", [call[0] for call in self.generator.calls]
        )

    def test_explicit_grades_must_cover_every_response(self):
        self.start()

        with self.assertRaises(ValueError):
            self.parcours.submit_initial_evaluation(
                [
                    {"question_id": "q1", "response_text": "réponse 1"},
                    {"question_id": "q2", "response_text": "réponse 2"},
                ],
                {"q1": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "OK"}},
            )

    def test_explicit_grades_reject_an_unknown_question(self):
        self.start()

        with self.assertRaises(ValueError):
            self.parcours.submit_initial_evaluation(
                [{"question_id": "q1", "response_text": "réponse 1"}],
                {
                    "q1": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "OK"},
                    "qz": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "OK"},
                },
            )

    def test_submit_final_exam_accepts_the_agent_grades(self):
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
        self.parcours.start_final_exam()
        self.generator.calls.clear()
        agent_parcours = orchestration.Parcours(self.store, "student-1")

        result = agent_parcours.submit_final_exam(
            [
                {"question_id": "q4", "response_text": "réponse 4"},
                {"question_id": "q5", "response_text": "réponse 5"},
            ],
            {
                "q4": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "Bien"},
                "q5": {"score": 1.0, "assistance_coefficient": 1.0, "feedback": "Bien"},
            },
        )

        self.assertTrue(result["sufficient"])
        self.assertNotIn(
            "grade_response", [call[0] for call in self.generator.calls]
        )

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
            {"document_id": "doc-unknown", "page": 1}
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

    def test_next_exercise_deletes_unsaved_courses(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        self.parcours.next_lesson()
        state = self.store.get_parcours_state("student-1")
        concept_id = state["current_concept_id"]
        courses = self.store.list_courses("student-1", concept_id=concept_id)
        self.assertEqual(len(courses), 1)
        course_id = courses[0]["course_id"]

        self.parcours.next_exercise()

        self.assertIsNone(self.store.get_course("student-1", course_id))

    def test_keep_course_preserves_the_course(self):
        self.start()
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        lesson = self.parcours.next_lesson()
        self.parcours.keep_course(lesson["course_id"])
        self.parcours.next_exercise()

        self.assertIsNotNone(self.store.get_course("student-1", lesson["course_id"]))


def _question(question_id, concept_id):
    return {
        "question_id": question_id,
        "concept_id": concept_id,
        "prompt": f"Question {question_id}",
        "reference_answer": "Réponse",
        "grading_rubric": "Barème",
        "difficulty_coefficient": 1.0,
        "citations": [{"document_id": "doc-1", "page": 1}],
    }


class QueuedEvaluationGenerator(ScriptedGenerator):
    """Returns the queued evaluations in order and records the feedback."""

    def __init__(self, evaluations):
        super().__init__()
        self.evaluations = list(evaluations)
        self.feedbacks = []

    def generate_initial_evaluation(
        self, concepts, passages, objectives, preferences=None, feedback=None
    ):
        self.calls.append(("generate_initial_evaluation", objectives, len(concepts)))
        self.feedbacks.append(feedback)
        item = self.evaluations.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FaultyAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = storage.SQLiteStore(
            Path(self.temporary_directory.name) / "learner.sqlite3"
        )
        self.store.create_learner("student-1")
        self.document = documents.Document(
            document_id="doc-1",
            title="Cours",
            path="/tmp/cours.pdf",
            media_type="application/pdf",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def parcours(self, evaluations):
        self.generator = QueuedEvaluationGenerator(evaluations)
        return orchestration.Parcours(self.store, "student-1", self.generator)

    def evaluation_calls(self):
        return [call for call in self.generator.calls if call[0] == "generate_initial_evaluation"]

    def test_an_unknown_concept_question_is_dropped_when_coverage_is_kept(self):
        parcours = self.parcours(
            [{"questions": [_question("q1", "a"), _question("q2", "b"), _question("q3", "invente")]}]
        )

        result = parcours.start([self.document], "objectifs")

        self.assertEqual([q["question_id"] for q in result["questions"]], ["q1", "q2"])
        self.assertEqual(len(self.evaluation_calls()), 1)
        events = [e for e in self.store.list_history("student-1") if e["kind"] == "questions_discarded"]
        self.assertEqual(len(events), 1)

    def test_lost_coverage_triggers_one_retry_with_feedback(self):
        parcours = self.parcours(
            [
                {"questions": [_question("q1", "a"), _question("q2", "invente")]},
                {"questions": [_question("q1", "a"), _question("q2", "b")]},
            ]
        )

        result = parcours.start([self.document], "objectifs")

        self.assertEqual([q["concept_id"] for q in result["questions"]], ["a", "b"])
        self.assertIsNone(self.generator.feedbacks[0])
        self.assertIn("invente", self.generator.feedbacks[1])
        self.assertIn("a, b", self.generator.feedbacks[1])

    def test_the_best_answer_is_kept_after_a_failed_retry(self):
        parcours = self.parcours(
            [
                {"questions": [_question("q1", "a"), _question("q2", "invente")]},
                {"questions": [_question("q3", "invente")]},
            ]
        )

        result = parcours.start([self.document], "objectifs")

        self.assertEqual([q["question_id"] for q in result["questions"]], ["q1"])
        self.assertEqual(len(self.evaluation_calls()), 2)

    def test_a_malformed_question_is_dropped_too(self):
        broken = _question("q3", "a")
        broken["difficulty_coefficient"] = 7
        parcours = self.parcours(
            [{"questions": [_question("q1", "a"), _question("q2", "b"), broken]}]
        )

        result = parcours.start([self.document], "objectifs")

        self.assertEqual(len(result["questions"]), 2)

    def test_the_graph_is_kept_when_the_evaluation_fails(self):
        parcours = self.parcours(
            [
                {"questions": [_question("q1", "invente")]},
                {"questions": [_question("q2", "invente")]},
            ]
        )

        with self.assertRaises(ValueError):
            parcours.start([self.document], "objectifs")

        state = self.store.get_parcours_state("student-1")
        self.assertEqual(state["phase"], "not_started")
        self.assertEqual(state["graph_id"], "graph-1")
        self.assertIsNotNone(self.store.get_graph("student-1", "graph-1"))
        with self.assertRaises(ValueError):
            parcours.start([self.document], "objectifs")

    def test_the_evaluation_can_be_retried_without_a_new_graph(self):
        parcours = self.parcours(
            [
                RuntimeError("model at capacity"),
                {"questions": [_question("q1", "a"), _question("q2", "b")]},
            ]
        )
        with self.assertRaises(RuntimeError):
            parcours.start([self.document], "objectifs")

        result = parcours.prepare_initial_evaluation()

        self.assertEqual(len(result["questions"]), 2)
        self.assertEqual(self.store.get_parcours_state("student-1")["phase"], "initial_evaluation")
        graph_calls = [call for call in self.generator.calls if call[0] == "propose_graph"]
        self.assertEqual(len(graph_calls), 1)

    def test_prepare_initial_evaluation_requires_a_saved_graph(self):
        parcours = self.parcours([])

        with self.assertRaises(ValueError):
            parcours.prepare_initial_evaluation()

    def test_a_generator_without_feedback_support_is_still_retried(self):
        class LegacyGenerator(ScriptedGenerator):
            def __init__(self):
                super().__init__()
                self.answers = [
                    {"questions": [_question("q1", "a"), _question("q2", "invente")]},
                    {"questions": [_question("q1", "a"), _question("q2", "b")]},
                ]

            def generate_initial_evaluation(self, concepts, passages, objectives, preferences=None):
                return self.answers.pop(0)

        parcours = orchestration.Parcours(self.store, "student-1", LegacyGenerator())

        result = parcours.start([self.document], "objectifs")

        self.assertEqual(len(result["questions"]), 2)


class GradingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = storage.SQLiteStore(
            Path(self.temporary_directory.name) / "learner.sqlite3"
        )
        self.store.create_learner("student-1")
        self.document = documents.Document(
            document_id="doc-1", title="Cours", path="/tmp/cours.pdf",
            media_type="application/pdf",
        )
        self.generator = ScriptedGenerator()
        self.parcours = orchestration.Parcours(self.store, "student-1", self.generator)
        self.parcours.start([self.document], "objectifs")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def grading_calls(self):
        return [call for call in self.generator.calls if call[0] in ("grade_response", "grade_responses")]

    def test_dont_know_and_empty_answers_are_not_sent_to_the_model(self):
        result = self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "texte ignoré", "dont_know": True},
                {"question_id": "q2", "response_text": "   "},
            ]
        )

        self.assertEqual(self.grading_calls(), [])
        self.assertEqual(result["levels"], {"a": 0.0, "b": 0.0})

    def test_only_the_answered_questions_are_graded(self):
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse"},
                {"question_id": "q2", "response_text": "", "dont_know": True},
            ]
        )

        self.assertEqual(self.grading_calls(), [("grade_response", "q1", "réponse")])

    def test_answers_are_graded_in_one_batch_call(self):
        batches = []

        def grade_responses(items):
            batches.append([item["question"]["question_id"] for item in items])
            return {item["question"]["question_id"]: GRADES[item["question"]["question_id"]] for item in items}

        self.generator.grade_responses = grade_responses
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )

        self.assertEqual(batches, [["q1", "q2"]])
        self.assertEqual(self.grading_calls(), [])

    def test_a_grade_missing_from_the_batch_is_graded_alone(self):
        self.generator.grade_responses = lambda items: {"q1": GRADES["q1"], "q2": {"score": 9}}
        self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )

        self.assertEqual(self.grading_calls(), [("grade_response", "q2", "réponse 2")])

    def test_a_grading_failure_records_nothing(self):
        def broken(question, response_text):
            if question["question_id"] == "q2":
                raise ValueError("no valid JSON found in the model response")
            return GRADES[question["question_id"]]

        self.generator.grade_response = broken
        with self.assertRaises(ValueError):
            self.parcours.submit_initial_evaluation(
                [
                    {"question_id": "q1", "response_text": "réponse 1"},
                    {"question_id": "q2", "response_text": "réponse 2"},
                ]
            )

        state = self.store.get_parcours_state("student-1")
        assessment = self.store.get_assessment(state["current_assessment_id"])
        self.assertEqual(state["phase"], "initial_evaluation")
        self.assertTrue(all(not q.get("responses") for q in assessment["questions"]))
        self.generator.grade_response = lambda question, text: GRADES[question["question_id"]]
        result = self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse 1"},
                {"question_id": "q2", "response_text": "réponse 2"},
            ]
        )
        self.assertIn("gap", result)

    def test_explicit_grades_need_not_cover_dont_know_answers(self):
        result = self.parcours.submit_initial_evaluation(
            [
                {"question_id": "q1", "response_text": "réponse"},
                {"question_id": "q2", "response_text": "", "dont_know": True},
            ],
            grades={"q1": GRADES["q1"]},
        )

        self.assertEqual(result["levels"]["b"], 0.0)
        self.assertEqual(self.grading_calls(), [])


class InitialEvaluationSizeTests(FaultyAssessmentTests):
    def test_the_initial_evaluation_is_capped_at_ten_questions(self):
        questions = [_question(f"q{i}", "a" if i % 2 else "b") for i in range(14)]
        parcours = self.parcours([{"questions": questions}])

        result = parcours.start([self.document], "objectifs")

        self.assertEqual(len(result["questions"]), orchestration.MAX_INITIAL_QUESTIONS)

    def test_one_bad_question_out_of_many_needs_no_retry_even_without_full_coverage(self):
        questions = [_question(f"q{i}", "a") for i in range(4)] + [_question("q9", "invente")]
        parcours = self.parcours([{"questions": questions}])

        result = parcours.start([self.document], "objectifs")

        self.assertEqual(len(result["questions"]), 4)
        self.assertEqual(len(self.evaluation_calls()), 1)


if __name__ == "__main__":
    unittest.main()
