"""PDF text extraction for the media pipeline.

A PDF the user sends in chat, or one the agent opens from Gmail, reaches the
agent as its extracted text layer. Scanned PDFs have no text layer; for those
the caller reports that nothing could be extracted and the bytes stay staged
so the agent can still save the file.

Parsing runs in a child process (``_pdf_worker.py``) under a memory cap, a
CPU cap, and a wall-clock timeout. The bytes come from anyone who can send the
user a message or an email. A PDF of a few kilobytes can decompress into a
content stream that takes minutes and gigabytes to parse, and pypdf keeps
shipping fixes for inputs of that kind. A thread cannot be stopped once
started; a process can be killed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches the Gmail body cap: enough for a multi-page inspection report or
# invoice without letting a 300-page manual flood the context window.
MAX_PDF_TEXT_CHARS = 16_000

# Bound the CPU spent on huge documents. Text past this page would be cut by
# the character cap anyway for any realistically dense PDF.
MAX_PDF_PAGES = 50

# Limits on the child process. Ordinary PDFs finish in well under a second
# and a few tens of megabytes.
PDF_TIMEOUT_SECONDS = 20.0
PDF_MAX_MEMORY_BYTES = 512 * 1024 * 1024
PDF_MAX_CPU_SECONDS = 30

# At most this many parses at once, so a burst of PDFs cannot multiply the
# memory cap across the host.
_MAX_CONCURRENT = 2
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

_WORKER = Path(__file__).with_name("_pdf_worker.py")

TRUNCATION_MARKER = "\n[...truncated]"


class PdfExtractionError(Exception):
    """The bytes could not be parsed as a PDF (corrupt, encrypted, or too costly)."""


async def _run_worker(content: bytes, max_chars: int, max_pages: int) -> dict[str, object]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_WORKER),
        str(max_chars),
        str(max_pages),
        str(PDF_MAX_MEMORY_BYTES),
        str(PDF_MAX_CPU_SECONDS),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(content), timeout=PDF_TIMEOUT_SECONDS
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise PdfExtractionError("PDF took too long to read") from None
    if proc.returncode != 0:
        logger.info(
            "PDF worker exited %s: %s",
            proc.returncode,
            stderr.decode(errors="replace")[-300:],
        )
        raise PdfExtractionError("PDF is too large or complex to read")
    try:
        result = json.loads(stdout)
    except ValueError as exc:
        raise PdfExtractionError("PDF reader returned no result") from exc
    if not isinstance(result, dict):
        raise PdfExtractionError("PDF reader returned no result")
    return result


async def extract_pdf_text(
    content: bytes,
    max_chars: int = MAX_PDF_TEXT_CHARS,
    max_pages: int = MAX_PDF_PAGES,
) -> tuple[str, int]:
    """Return ``(text, page_count)`` for a PDF.

    ``text`` is empty when the PDF has no text layer (typically a scan).
    Raises :class:`PdfExtractionError` when the bytes are not a readable PDF
    or reading them exceeds the child process limits.
    """
    async with _semaphore:
        result = await _run_worker(content, max_chars, max_pages)
    error = result.get("error")
    if error is not None:
        raise PdfExtractionError(str(error))
    text = str(result.get("text") or "")
    pages = result.get("pages")
    total_pages = pages if isinstance(pages, int) else 0
    if len(text) > max_chars or (text and total_pages > max_pages):
        text = text[:max_chars] + TRUNCATION_MARKER
    return text, total_pages
