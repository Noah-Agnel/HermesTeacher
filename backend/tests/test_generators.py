import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from backend.web.generators import HermesGenerator, extract_json
except ImportError:  # pragma: no cover - web dependencies missing
    HermesGenerator = None
    extract_json = None


class ExtractJsonTests(unittest.TestCase):
    def test_extracts_object_from_prose(self):
        self.assertEqual(
            extract_json('Here you go: {"a": 1} thanks'), {"a": 1}
        )

    def test_extracts_object_from_code_fence(self):
        self.assertEqual(
            extract_json('```json\n{"a": 1}\n```'), {"a": 1}
        )

    def test_extracts_nested_structures(self):
        self.assertEqual(
            extract_json('{"a": {"b": [1, 2]}}'), {"a": {"b": [1, 2]}}
        )

    def test_raises_without_json(self):
        with self.assertRaises(ValueError):
            extract_json("je ne peux pas répondre")

    def test_raises_on_non_text(self):
        with self.assertRaises(ValueError):
            extract_json(None)


@unittest.skipIf(HermesGenerator is None, "web dependencies are not installed")
class HermesGeneratorTests(unittest.TestCase):
    """The agent is stubbed by a command that prints a fixed JSON response."""

    @staticmethod
    def _agent(payload: str) -> list[str]:
        return [sys.executable, "-c", f"print({payload!r})"]

    def test_propose_graph_parses_the_agent_json(self):
        generator = HermesGenerator(
            command=self._agent('{"graph_id": "g", "concepts": []}')
        )

        result = generator.propose_graph(
            [{"document_id": "doc-1", "title": "Cours", "path": "/tmp/cours.pdf"}],
            "objectifs",
            {"format": "exemples"},
        )

        self.assertEqual(result["graph_id"], "g")

    def test_generate_course_parses_the_agent_json(self):
        generator = HermesGenerator(
            command=self._agent(
                '{"concept_id": "a", "title": "C", "content": "x", "citations": []}'
            )
        )

        result = generator.generate_course(
            {"concept_id": "a", "label": "A"}, [], 0.42
        )

        self.assertEqual(result["concept_id"], "a")

    def test_the_prompt_carries_the_context(self):
        capture = Path(tempfile.mkdtemp()) / "prompt.txt"
        command = [
            sys.executable,
            "-c",
            "import os, sys\n"
            "open(os.environ['PROMPT_CAPTURE'], 'w').write(sys.argv[1])\n"
            "print('{\"graph_id\": \"g\", \"concepts\": []}')",
        ]
        generator = HermesGenerator(command=command)
        os.environ["PROMPT_CAPTURE"] = str(capture)
        try:
            generator.propose_graph(
                [{"document_id": "doc-1", "title": "Cours", "path": "/tmp/cours.pdf"}],
                "objectifs",
                {"format": "exemples"},
            )
        finally:
            os.environ.pop("PROMPT_CAPTURE", None)

        prompt = capture.read_text(encoding="utf-8")
        self.assertIn("doc-1", prompt)
        self.assertIn("/tmp/cours.pdf", prompt)
        self.assertIn("format=exemples", prompt)
        self.assertIn("agent_tools profile", prompt)

    def test_the_document_text_is_inlined_in_the_prompt(self):
        try:
            import pymupdf
        except ImportError:
            self.skipTest("pymupdf is not installed")
        directory = Path(tempfile.mkdtemp())
        pdf_path = directory / "doc-1.pdf"
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), "Propriete de Markov forte")
        pdf.save(str(pdf_path))
        pdf.close()
        capture = directory / "prompt.txt"
        command = [
            sys.executable,
            "-c",
            "import os, sys\n"
            "open(os.environ['PROMPT_CAPTURE'], 'w').write(sys.argv[1])\n"
            "print('{\"graph_id\": \"g\", \"concepts\": []}')",
        ]
        os.environ["PROMPT_CAPTURE"] = str(capture)
        try:
            HermesGenerator(command=command).propose_graph(
                [{"document_id": "doc-1", "title": "Cours", "path": str(pdf_path)}],
                "objectifs",
            )
        finally:
            os.environ.pop("PROMPT_CAPTURE", None)

        prompt = capture.read_text(encoding="utf-8")
        self.assertIn("Propriete de Markov forte", prompt)
        self.assertIn("=== Page 1 ===", prompt)
        self.assertNotIn("read_file` (chemins", prompt)

    def test_assessment_prompts_restrict_concepts_and_carry_feedback(self):
        capture = Path(tempfile.mkdtemp()) / "prompt.txt"
        command = [
            sys.executable,
            "-c",
            "import os, sys\n"
            "open(os.environ['PROMPT_CAPTURE'], 'w').write(sys.argv[1])\n"
            "print('{\"questions\": []}')",
        ]
        os.environ["PROMPT_CAPTURE"] = str(capture)
        try:
            HermesGenerator(command=command).generate_initial_evaluation(
                [{"concept_id": "markov", "label": "Markov"}, {"concept_id": "ergodique", "label": "E"}],
                [],
                "objectifs",
                feedback="q6 : unknown concept 'invente'",
            )
        finally:
            os.environ.pop("PROMPT_CAPTURE", None)

        prompt = capture.read_text(encoding="utf-8")
        self.assertIn("AUTORISÉS (aucun autre) : markov, ergodique", prompt)
        self.assertIn("CORRECTION DEMANDÉE", prompt)
        self.assertIn("unknown concept 'invente'", prompt)
        self.assertIn("n'écris aucun", prompt)

    def test_a_missing_json_is_retried_once(self):
        counter = Path(tempfile.mkdtemp()) / "count"
        command = [
            sys.executable,
            "-c",
            "import os, sys\n"
            f"p = {str(counter)!r}\n"
            "n = int(open(p).read()) if os.path.exists(p) else 0\n"
            "open(p, 'w').write(str(n + 1))\n"
            "print('⚠️ model at capacity' if n == 0 else '{\"graph_id\": \"g\"}')",
        ]

        result = HermesGenerator(command=command).propose_graph([], "objectifs")

        self.assertEqual(result["graph_id"], "g")
        self.assertEqual(counter.read_text(), "2")

    def test_the_error_quotes_what_hermes_answered(self):
        generator = HermesGenerator(
            command=[sys.executable, "-c", "print('⚠️ Hermes is not logged into Nous Portal.')"]
        )

        with self.assertRaises(ValueError) as context:
            generator.propose_graph([], "objectifs")

        self.assertIn("not logged into Nous Portal", str(context.exception))

    def test_grade_responses_returns_one_grade_per_question(self):
        payload = '{"grades": [{"question_id": "q1", "score": 1, "assistance_coefficient": 1, "feedback": "ok"}, {"question_id": "q2", "score": 0, "assistance_coefficient": 1, "feedback": "non"}]}'
        generator = HermesGenerator(command=self._agent(payload))

        grades = generator.grade_responses(
            [
                {"question": {"question_id": "q1", "prompt": "P1"}, "response_text": "r1"},
                {"question": {"question_id": "q2", "prompt": "P2"}, "response_text": "r2"},
            ]
        )

        self.assertEqual(set(grades), {"q1", "q2"})
        self.assertEqual(grades["q2"]["score"], 0)

    def test_the_initial_evaluation_prompt_asks_for_few_general_questions(self):
        capture = Path(tempfile.mkdtemp()) / "prompt.txt"
        command = [
            sys.executable,
            "-c",
            "import os, sys\n"
            "open(os.environ['PROMPT_CAPTURE'], 'w').write(sys.argv[1])\n"
            "print('{\"questions\": []}')",
        ]
        os.environ["PROMPT_CAPTURE"] = str(capture)
        try:
            HermesGenerator(command=command).generate_initial_evaluation(
                [{"concept_id": "markov", "label": "Markov"}], [], "objectifs"
            )
        finally:
            os.environ.pop("PROMPT_CAPTURE", None)

        prompt = capture.read_text(encoding="utf-8")
        self.assertIn("10 questions au maximum", prompt)
        self.assertIn("pas de sous-questions", prompt)
        self.assertNotIn("chaque concept doit être couvert", prompt)

    def test_raises_when_the_agent_fails(self):
        generator = HermesGenerator(
            command=[sys.executable, "-c", "import sys; sys.exit(3)"]
        )

        with self.assertRaises(RuntimeError):
            generator.propose_graph([], "objectifs")

    def test_raises_when_the_agent_returns_no_json(self):
        generator = HermesGenerator(
            command=[sys.executable, "-c", "print('je ne peux pas répondre')"]
        )

        with self.assertRaises(ValueError):
            generator.propose_graph([], "objectifs")


if __name__ == "__main__":
    unittest.main()
