"""PDF text extraction for the media pipeline.

A PDF the user sends in chat, or one the agent opens from Gmail, reaches the
agent as its extracted text layer. Scanned PDFs have no text layer; for those
the caller reports that nothing could be extracted and the bytes stay staged
so the agent can still save the file.
"""

from __future__ import annotations

import asyncio
import io
import logging

from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)

# Matches the Gmail body cap: enough for a multi-page inspection report or
# invoice without letting a 300-page manual flood the context window.
MAX_PDF_TEXT_CHARS = 16_000

# Bound the CPU spent on huge documents. Text past this page would be cut by
# the character cap anyway for any realistically dense PDF.
MAX_PDF_PAGES = 50

TRUNCATION_MARKER = "\n[...truncated]"


class PdfExtractionError(Exception):
    """The bytes could not be parsed as a PDF (corrupt or encrypted)."""


def _extract_sync(content: bytes, max_chars: int, max_pages: int) -> tuple[str, int]:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfExtractionError("PDF is password protected")
        total_pages = len(reader.pages)
        chunks: list[str] = []
        length = 0
        for index, page in enumerate(reader.pages):
            if index >= max_pages or length > max_chars:
                break
            text = (page.extract_text() or "").strip()
            if text:
                chunks.append(text)
                length += len(text)
    except PdfExtractionError:
        raise
    except (PyPdfError, ValueError, KeyError, TypeError, OSError) as exc:
        raise PdfExtractionError(str(exc) or type(exc).__name__) from exc

    text = "\n\n".join(chunks)
    if len(text) > max_chars or (text and total_pages > max_pages):
        text = text[:max_chars] + TRUNCATION_MARKER
    return text, total_pages


async def extract_pdf_text(
    content: bytes,
    max_chars: int = MAX_PDF_TEXT_CHARS,
    max_pages: int = MAX_PDF_PAGES,
) -> tuple[str, int]:
    """Return ``(text, page_count)`` for a PDF.

    ``text`` is empty when the PDF has no text layer (typically a scan).
    Raises :class:`PdfExtractionError` when the bytes are not a readable PDF.
    Parsing is CPU-bound, so it runs off the event loop.
    """
    return await asyncio.to_thread(_extract_sync, content, max_chars, max_pages)
