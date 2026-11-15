import tempfile
import unittest
from pathlib import Path

try:
    from backend.tutor import documents
except ImportError:
    documents = None


class DocumentStorageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(documents, "documents module should be implemented")
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temporary_directory.name) / "uploads"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_store_upload_writes_the_file_and_returns_its_record(self):
        document = documents.store_upload(
            self.upload_dir,
            document_id="doc-1",
            title="Cours",
            filename="cours.pdf",
            content=b"%PDF-1.4 fake",
        )

        self.assertEqual(document.document_id, "doc-1")
        self.assertEqual(document.title, "Cours")
        self.assertEqual(document.media_type, "application/pdf")
        self.assertTrue(Path(document.path).exists())
        self.assertEqual(Path(document.path).read_bytes(), b"%PDF-1.4 fake")

    def test_store_upload_rejects_unsupported_format(self):
        with self.assertRaises(ValueError):
            documents.store_upload(
                self.upload_dir,
                document_id="doc-1",
                title="Image",
                filename="image.png",
                content=b"not a pdf",
            )

    def test_store_upload_rejects_invalid_document_id(self):
        with self.assertRaises(ValueError):
            documents.store_upload(
                self.upload_dir,
                document_id="",
                title="Cours",
                filename="cours.pdf",
                content=b"%PDF-1.4 fake",
            )

    def test_store_upload_uses_the_document_id_as_filename(self):
        document = documents.store_upload(
            self.upload_dir,
            document_id="doc-1",
            title="Cours",
            filename="cours.pdf",
            content=b"%PDF-1.4 fake",
        )

        self.assertEqual(Path(document.path).name, "doc-1.pdf")


def _make_pdf(path: Path, pages: list[str]) -> None:
    import pymupdf

    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(str(path))
    pdf.close()


class DocumentTextTests(unittest.TestCase):
    def setUp(self):
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            self.skipTest("pymupdf is not installed")
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temporary_directory.name) / "uploads"
        self.upload_dir.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_upload_creates_a_text_companion_with_page_markers(self):
        source = self.upload_dir / "source.pdf"
        _make_pdf(source, ["Chaines de Markov", "Matrice de transition"])

        document = documents.store_upload(
            self.upload_dir,
            document_id="doc-1",
            title="Cours",
            filename="cours.pdf",
            content=source.read_bytes(),
        )

        text_file = documents.text_path(document.path)
        self.assertEqual(text_file.name, "doc-1.txt")
        text = text_file.read_text(encoding="utf-8")
        self.assertIn("=== Page 1 ===", text)
        self.assertIn("Chaines de Markov", text)
        self.assertIn("=== Page 2 ===", text)
        self.assertIn("Matrice de transition", text)

    def test_a_broken_pdf_is_still_stored(self):
        document = documents.store_upload(
            self.upload_dir,
            document_id="doc-1",
            title="Cours",
            filename="cours.pdf",
            content=b"%PDF-1.4 fake",
        )

        self.assertTrue(Path(document.path).exists())
        self.assertIsNone(documents.read_text(document.path))

    def test_read_text_converts_older_uploads_on_demand(self):
        pdf = self.upload_dir / "old.pdf"
        _make_pdf(pdf, ["Ancien depot"])

        self.assertIn("Ancien depot", documents.read_text(pdf))
        self.assertTrue(documents.text_path(pdf).exists())

    def test_control_characters_never_reach_the_text(self):
        pdf = self.upload_dir / "old.pdf"
        _make_pdf(pdf, ["Cours"])
        documents.text_path(pdf).write_text("a\x00b\x0cc\nd", encoding="utf-8")

        self.assertEqual(documents.read_text(pdf), "a b c\nd")

    def test_latex_accents_are_composed(self):
        self.assertEqual(documents._fix_accents("probabilit´e `a"), "probabilité à")


if __name__ == "__main__":
    unittest.main()
