"""Deposited document storage and text conversion.

The backend stores the learner's files as-is and records their location. At
upload it also converts each PDF once into a plain-text companion file
(``{document_id}.txt`` next to the PDF) with one ``=== Page N ===`` marker per
page, so the agent can receive the whole course in its prompt instead of
reading the PDF piece by piece with its tools (one model call instead of
dozens).

The conversion keeps the mathematics readable: LaTeX-generated PDFs draw big
operators (∑, ∫, ⋂, ∏ ...) with the ``cmex`` font, whose glyphs extract as
plain letters (``P``, ``Z``, ``T`` ...); they are mapped back to Unicode, as
are blackboard letters (ℝ, ℕ, ℙ ...), calligraphic letters (ℱ, ℰ ...), LaTeX accents (``´e`` → ``é``) and
sub/superscripts (``X_{n}``, ``x^{2}``). Fraction bars and operator limits
stay on separate lines, as in the PDF layout.

The agent still cites documents by id, with a page and a quote.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SUPPORTED_MEDIA_TYPES = {".pdf": "application/pdf"}
TEXT_SUFFIX = ".txt"


@dataclass(frozen=True)
class Document:
    """A deposited file stored as-is, with its metadata."""

    document_id: str
    title: str
    path: str
    media_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def store_upload(
    upload_dir: str | Path,
    *,
    document_id: str,
    title: str,
    filename: str,
    content: bytes,
) -> Document:
    """Write an uploaded file to the upload directory and return its record.

    The file is stored unchanged. A text companion is then created next to it
    (see :func:`ensure_text`); a conversion failure never blocks the upload,
    the agent can still read the PDF itself.
    """
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(title, str):
        raise ValueError("title must be a string")
    if not isinstance(content, bytes):
        raise ValueError("content must be bytes")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_MEDIA_TYPES:
        raise ValueError(f"format de document non supporté : {suffix or '(aucun)'}")
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{document_id}{suffix}"
    path.write_bytes(content)
    ensure_text(path)
    return Document(
        document_id=document_id,
        title=title,
        path=str(path),
        media_type=SUPPORTED_MEDIA_TYPES[suffix],
    )


# ----------------------------------------------------------------------
# text conversion
# ----------------------------------------------------------------------

# cmex10 glyph codes → the symbol they draw (big operators and delimiters).
_CMEX = {
    "P": "∑", "X": "∑", "Q": "∏", "Y": "∏", "R": "∫", "Z": "∫",
    "S": "⋃", "[": "⋃", "T": "⋂", "\\": "⋂", "V": "⋀", "W": "⋁", "U": "⊎",
    " ": "(", "!": ")", "\x10": "(", "\x11": ")", "\x12": "(", "\x13": ")",
    '"': "[", "#": "]", "(": "{", ")": "}", "\x08": "{", "\t": "}",
    "b": "^", "z": "", "{": "", "|": "", "}": "",
    "\x00": "(", "\x01": ")", "\x02": "[", "\x03": "]", "\x04": "⌊", "\x05": "⌋",
    "\x06": "⌈", "\x07": "⌉", "\x0a": "⟨", "\x0b": "⟩", "\x0c": "|", "\x0d": "‖",
    "\x0e": "/", "\x0f": "\\", "\x14": "[", "\x15": "]", "\x16": "⌊", "\x17": "⌋",
    "\x18": "⌈", "\x19": "⌉", "\x1a": "{", "\x1b": "}", "\x1c": "⟨", "\x1d": "⟩",
}
# Control characters left over after mapping (never valid in a prompt).
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# msbm (blackboard bold) glyph codes.
_BLACKBOARD = {"N": "ℕ", "R": "ℝ", "Z": "ℤ", "P": "ℙ", "E": "𝔼", "Q": "ℚ", "C": "ℂ"}
# cmsy (math symbols) draws \\mathcal capitals with plain letter codes.
_CALLIGRAPHIC = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "𝒜ℬ𝒞𝒟ℰℱ𝒢ℋℐ𝒥𝒦ℒℳ𝒩𝒪𝒫𝒬ℛ𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵",
))
# LaTeX spacing accents → combining accents.
_ACCENTS = {"´": "\u0301", "`": "\u0300", "¨": "\u0308", "ˆ": "\u0302"}


def text_path(pdf_path: str | Path) -> Path:
    """Return the path of the text companion of a deposited PDF."""
    return Path(pdf_path).with_suffix(TEXT_SUFFIX)


def ensure_text(pdf_path: str | Path) -> Path | None:
    """Create the text companion of a PDF if missing; return its path.

    Returns ``None`` when the PDF cannot be converted (missing file, damaged
    PDF, converter not installed).
    """
    target = text_path(pdf_path)
    if target.exists():
        return target
    try:
        text = extract_pdf_text(pdf_path)
    except Exception:  # a broken PDF must not break the upload
        return None
    target.write_text(text, encoding="utf-8")
    return target


def read_text(pdf_path: str | Path) -> str | None:
    """Return the converted text of a deposited PDF, converting it if needed."""
    target = ensure_text(pdf_path)
    if target is None:
        return None
    return _CONTROL.sub(" ", target.read_text(encoding="utf-8"))


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Convert a PDF to text with ``=== Page N ===`` markers."""
    import pymupdf  # imported lazily: only needed when a PDF is converted

    pages = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for number, page in enumerate(pdf, start=1):
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    rendered = _render_line(line.get("spans", []))
                    if rendered is not None:
                        lines.append(rendered)
            pages.append(f"=== Page {number} ===\n" + _fix_accents("\n".join(lines)))
    return _CONTROL.sub(" ", "\n\n".join(pages)) + "\n"


def _render_line(spans: list[dict[str, Any]]) -> str | None:
    spans = [span for span in spans if span.get("text")]
    if not spans:
        return None
    largest = max(span["size"] for span in spans)
    parts = []
    for index, span in enumerate(spans):
        text = span["text"]
        font = span.get("font", "")
        if "CMEX" in font.upper():
            text = "".join(_CMEX.get(character, character) for character in text)
        elif "MSBM" in font.upper():
            text = "".join(_BLACKBOARD.get(character, character) for character in text)
        elif "CMSY" in font.upper():
            text = "".join(_CALLIGRAPHIC.get(character, character) for character in text)
        small = span["size"] < 0.85 * largest
        previous = [s for s in spans[:index] if s["size"] >= 0.85 * largest]
        if small and text.strip() and previous and "CMEX" not in previous[-1].get("font", "").upper():
            shift = span["origin"][1] - previous[-1]["origin"][1]
            if shift > 0.8:
                text = "_{" + text.strip() + "}"
            elif shift < -0.8:
                text = "^{" + text.strip() + "}"
        parts.append(text)
    rendered = "".join(parts)
    # merge consecutive scripts: X_{n}_{+}_{k} → X_{n+k}
    rendered = re.sub(r"\}_\{", "", rendered)
    rendered = re.sub(r"\}\^\{", "", rendered)
    return (
        rendered.replace("7−→", "↦")
        .replace("7→", "↦")
        .replace("\u0338 =", "≠")
        .replace("\u0338=", "≠")
    )


def _fix_accents(text: str) -> str:
    def compose(match: re.Match[str]) -> str:
        return unicodedata.normalize("NFC", match.group(2) + _ACCENTS[match.group(1)])

    text = text.replace("ˆı", "î").replace("¨ı", "ï")
    text = re.sub(r"([´`¨ˆ])\s?([A-Za-z])", compose, text)
    return re.sub(
        r"([A-Za-z])¸",
        lambda match: unicodedata.normalize("NFC", match.group(1) + "̧"),
        text,
    )
