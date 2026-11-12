"""Document loading and chunking.

A document is split into ordered passages. Each passage has a stable id and,
for PDFs, a page number, so a concept graph can cite the exact passage that
supports a concept or a prerequisite relation.

Only plain text, Markdown and PDF are supported. PDF extraction uses pypdf,
which is an optional dependency: `pip install pypdf`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

DEFAULT_MAX_PASSAGE_CHARS = 2000

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
_BLANK_LINE_PATTERN = re.compile(r"\n\s*\n")

_EXTENSION_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
}


@dataclass(frozen=True)
class Passage:
    """A chunk of a document, citable as evidence."""

    passage_id: str
    document_id: str
    ordinal: int
    text: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class Document:
    """A loaded document and its ordered passages."""

    document_id: str
    title: str
    source: str
    media_type: str
    passages: tuple[Passage, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _passage_id(document_id: str, ordinal: int) -> str:
    return f"{document_id}:{ordinal:04d}"


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most ``max_chars``, breaking on spaces."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        break_at = window.rfind(" ")
        if break_at <= 0:
            break_at = max_chars
        chunks.append(remaining[:break_at].strip())
        remaining = remaining[break_at:].strip()
    if remaining:
        chunks.append(remaining)
    return [chunk for chunk in chunks if chunk]


def _split_markdown_blocks(text: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.split("\n"):
        match = _HEADING_PATTERN.match(line)
        if match:
            if lines:
                blocks.append((heading, "\n".join(lines).strip()))
            heading = match.group(2)
            lines = [line]
        else:
            lines.append(line)
    if lines:
        blocks.append((heading, "\n".join(lines).strip()))
    return [(block_heading, block) for block_heading, block in blocks if block]


def _split_paragraphs(text: str) -> list[tuple[str | None, str]]:
    return [
        (None, paragraph.strip())
        for paragraph in _BLANK_LINE_PATTERN.split(text)
        if paragraph.strip()
    ]


def load_text(
    text: str,
    *,
    document_id: str,
    title: str | None = None,
    source: str | None = None,
    media_type: str = "text/plain",
    max_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    metadata: Mapping[str, Any] | None = None,
) -> Document:
    """Load plain text or Markdown and split it into passages."""
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")

    cleaned = _clean(text)
    if media_type == "text/markdown":
        blocks = _split_markdown_blocks(cleaned)
    else:
        blocks = _split_paragraphs(cleaned)

    passages: list[Passage] = []
    ordinal = 0
    for heading, block in blocks:
        for chunk in _chunk_text(block, max_chars):
            passages.append(
                Passage(
                    passage_id=_passage_id(document_id, ordinal),
                    document_id=document_id,
                    ordinal=ordinal,
                    text=chunk,
                    heading=heading,
                )
            )
            ordinal += 1
    if not passages:
        raise ValueError("text produced no passages")

    return Document(
        document_id=document_id,
        title=title if title is not None else document_id,
        source=source if source is not None else "",
        media_type=media_type,
        passages=tuple(passages),
        metadata=dict(metadata) if metadata is not None else {},
    )


def load_file(
    path: str | Path,
    *,
    document_id: str | None = None,
    title: str | None = None,
    max_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
) -> Document:
    """Load a .txt, .md, .markdown or .pdf file, dispatching on the extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _EXTENSION_MEDIA_TYPES:
        raise ValueError(f"unsupported document extension: {suffix!r}")
    if document_id is None:
        document_id = path.stem
    if title is None:
        title = path.stem
    if suffix == ".pdf":
        return load_pdf(
            path, document_id=document_id, title=title, max_chars=max_chars
        )
    text = path.read_text(encoding="utf-8")
    return load_text(
        text,
        document_id=document_id,
        title=title,
        source=str(path),
        media_type=_EXTENSION_MEDIA_TYPES[suffix],
        max_chars=max_chars,
    )


def load_pdf(
    path: str | Path,
    *,
    document_id: str | None = None,
    title: str | None = None,
    max_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
) -> Document:
    """Load a PDF and create one or more passages per non-empty page."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError(
            "PDF support requires pypdf; install it with `pip install pypdf`"
        ) from error

    path = Path(path)
    if document_id is None:
        document_id = path.stem
    reader = PdfReader(str(path))

    passages: list[Passage] = []
    ordinal = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        cleaned = _clean(page_text)
        if not cleaned:
            continue
        for chunk in _chunk_text(cleaned, max_chars):
            passages.append(
                Passage(
                    passage_id=_passage_id(document_id, ordinal),
                    document_id=document_id,
                    ordinal=ordinal,
                    text=chunk,
                    page=page_number,
                )
            )
            ordinal += 1
    if not passages:
        raise ValueError(
            "PDF has no extractable text layer; scanned documents require OCR"
        )

    return Document(
        document_id=document_id,
        title=title if title is not None else document_id,
        source=str(path),
        media_type="application/pdf",
        passages=tuple(passages),
    )
