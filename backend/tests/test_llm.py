import unittest

try:
    from backend.tutor import llm
except ImportError:
    llm = None


class FakeCompleter:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.response


class ExtractJsonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(llm, "llm module should be implemented")

    def test_extracts_object_from_prose(self):
        self.assertEqual(
            llm.extract_json('Here you go: {"a": 1} thanks'), {"a": 1}
        )

    def test_extracts_object_from_code_fence(self):
        self.assertEqual(
            llm.extract_json('```json\n{"a": 1}\n```'), {"a": 1}
        )

    def test_extracts_nested_structures(self):
        self.assertEqual(
            llm.extract_json('{"a": {"b": [1, 2]}}'), {"a": {"b": [1, 2]}}
        )

    def test_raises_without_json(self):
        with self.assertRaises(ValueError):
            llm.extract_json("je ne peux pas répondre")

    def test_raises_on_non_text(self):
        with self.assertRaises(ValueError):
            llm.extract_json(None)


class PromptLLMGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(llm, "llm module should be implemented")

    def test_propose_graph_parses_json_and_includes_context(self):
        completer = FakeCompleter('{"graph_id": "g", "concepts": []}')
        generator = llm.PromptLLMGenerator(completer)

        result = generator.propose_graph(
            [{"passage_id": "doc-1:0000", "text": "Contenu."}],
            "objectifs",
            {"format": "examples"},
        )

        self.assertEqual(result["graph_id"], "g")
        prompt = completer.prompts[0]
        self.assertIn("objectifs", prompt)
        self.assertIn("doc-1:0000", prompt)
        self.assertIn("format=examples", prompt)

    def test_generate_initial_evaluation_lists_concepts(self):
        completer = FakeCompleter('{"questions": []}')
        generator = llm.PromptLLMGenerator(completer)

        generator.generate_initial_evaluation(
            [{"concept_id": "a", "label": "Concept A", "target_level": 1.0, "weight": 1.0}],
            [],
            "objectifs",
        )

        self.assertIn("Concept A", completer.prompts[0])

    def test_generate_course_includes_current_level(self):
        completer = FakeCompleter(
            '{"concept_id": "a", "title": "Cours", "content": "Contenu", "citations": []}'
        )
        generator = llm.PromptLLMGenerator(completer)

        generator.generate_course(
            {"concept_id": "a", "label": "Concept A"}, [], 0.42
        )

        self.assertIn("0.42", completer.prompts[0])

    def test_grade_response_includes_the_answer(self):
        completer = FakeCompleter(
            '{"score": 0.5, "assistance_coefficient": 0.8, "feedback": "ok"}'
        )
        generator = llm.PromptLLMGenerator(completer)

        result = generator.grade_response(
            {"prompt": "Q", "reference_answer": "R", "grading_rubric": "G"},
            "réponse de l'apprenant",
        )

        self.assertEqual(result["score"], 0.5)
        self.assertIn("réponse de l'apprenant", completer.prompts[0])

    def test_raises_when_the_model_returns_no_json(self):
        completer = FakeCompleter("je ne peux pas répondre")
        generator = llm.PromptLLMGenerator(completer)

        with self.assertRaises(ValueError):
            generator.generate_exercise({"concept_id": "a", "label": "A"}, [], 0.5)

    def test_rejects_a_non_callable_completer(self):
        with self.assertRaises(ValueError):
            llm.PromptLLMGenerator("not callable")


if __name__ == "__main__":
    unittest.main()
