import tempfile
import unittest
from pathlib import Path

try:
    from backend.tutor import documents
except ImportError:
    documents = None

try:
    import pypdf  # noqa: F401
    import reportlab  # noqa: F401

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class TextDocumentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(documents, "documents module should be implemented")

    def test_load_text_splits_paragraphs_and_assigns_ordinals(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        document = documents.load_text(text, document_id="doc-1", title="Cours")

        self.assertEqual(document.document_id, "doc-1")
        self.assertEqual(document.title, "Cours")
        self.assertEqual(document.media_type, "text/plain")
        self.assertEqual([p.ordinal for p in document.passages], [0, 1, 2])
        self.assertEqual(
            [p.text for p in document.passages],
            ["First paragraph.", "Second paragraph.", "Third paragraph."],
        )
        self.assertEqual(
            [p.passage_id for p in document.passages],
            ["doc-1:0000", "doc-1:0001", "doc-1:0002"],
        )

    def test_load_text_uses_headings_for_markdown(self):
        text = "# Titre\n\nIntro.\n\n## Section A\n\nContenu A.\n\n## Section B\n\nContenu B."
        document = documents.load_text(
            text, document_id="doc-1", media_type="text/markdown"
        )

        self.assertEqual(
            [p.heading for p in document.passages],
            ["Titre", "Section A", "Section B"],
        )
        self.assertEqual(len(document.passages), 3)

    def test_load_text_splits_long_blocks(self):
        text = "a" * 500 + "\n\n" + "b" * 500
        document = documents.load_text(text, document_id="doc-1", max_chars=400)

        self.assertGreater(len(document.passages), 2)
        for passage in document.passages:
            self.assertLessEqual(len(passage.text), 400)

    def test_load_text_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            documents.load_text("   \n  ", document_id="doc-1")

    def test_load_text_defaults_title_to_document_id(self):
        document = documents.load_text("Hello.", document_id="doc-1")

        self.assertEqual(document.title, "doc-1")

    def test_load_text_keeps_metadata(self):
        document = documents.load_text(
            "Hello.", document_id="doc-1", metadata={"author": "X"}
        )

        self.assertEqual(document.metadata["author"], "X")

    def test_load_file_dispatches_text_and_markdown(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            text_file = directory / "note.txt"
            text_file.write_text("Hello paragraph.", encoding="utf-8")
            markdown_file = directory / "cours.md"
            markdown_file.write_text("# Titre\n\nContenu.", encoding="utf-8")

            text_document = documents.load_file(text_file, document_id="doc-1")
            markdown_document = documents.load_file(markdown_file, document_id="doc-2")

            self.assertEqual(text_document.media_type, "text/plain")
            self.assertEqual(markdown_document.media_type, "text/markdown")
            self.assertEqual(markdown_document.passages[0].heading, "Titre")

    def test_load_file_rejects_unknown_extension(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "file.docx"
            path.write_text("x", encoding="utf-8")

            with self.assertRaises(ValueError):
                documents.load_file(path, document_id="doc-1")


@unittest.skipUnless(PDF_AVAILABLE, "pypdf and reportlab required for PDF tests")
class PdfDocumentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(documents, "documents module should be implemented")

    @staticmethod
    def _write_pdf(path, pages):
        from reportlab.pdfgen import canvas

        canvas_object = canvas.Canvas(str(path))
        for lines in pages:
            for index, line in enumerate(lines):
                canvas_object.drawString(72, 720 - index * 20, line)
            canvas_object.showPage()
        canvas_object.save()

    def test_load_pdf_extracts_pages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cours.pdf"
            self._write_pdf(path, [["Page one text."], ["Page two text."]])

            document = documents.load_pdf(path, document_id="doc-1")

            self.assertEqual(document.media_type, "application/pdf")
            self.assertEqual([p.page for p in document.passages], [1, 2])
            self.assertIn("Page one text.", document.passages[0].text)
            self.assertIn("Page two text.", document.passages[1].text)

    def test_load_pdf_rejects_document_without_text_layer(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scanned.pdf"
            self._write_pdf(path, [[]])

            with self.assertRaises(ValueError):
                documents.load_pdf(path, document_id="doc-1")


if __name__ == "__main__":
    unittest.main()
