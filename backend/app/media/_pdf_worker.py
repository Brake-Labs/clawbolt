"""Standalone PDF text extractor, run in a child process by ``pdf.py``.

Reads PDF bytes on stdin and writes ``{"text", "pages"}`` or
``{"error"}`` as JSON on stdout. It imports nothing from ``backend`` so it
starts fast and runs under the resource limits it sets on itself. pypdf
regularly ships fixes for inputs that make it spin or allocate without
bound; the limits here, and the parent's timeout, cap what an unfixed one
can cost.
"""

import contextlib
import io
import json
import resource
import sys

from pypdf import PdfReader
from pypdf.errors import PyPdfError


def _limit(kind: int, value: int) -> None:
    # Not every platform enforces every limit (macOS rejects RLIMIT_AS).
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(kind, (value, value))


def _extract(content: bytes, max_chars: int, max_pages: int) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(content))
    if reader.is_encrypted and not reader.decrypt(""):
        raise PyPdfError("PDF is password protected")
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
    return "\n\n".join(chunks), total_pages


def main() -> None:
    max_chars, max_pages, max_memory, max_cpu = (int(a) for a in sys.argv[1:5])
    _limit(resource.RLIMIT_AS, max_memory)
    _limit(resource.RLIMIT_CPU, max_cpu)
    content = sys.stdin.buffer.read()
    try:
        text, pages = _extract(content, max_chars, max_pages)
        out: dict[str, object] = {"text": text, "pages": pages}
    except (PyPdfError, ValueError, KeyError, TypeError, OSError, RecursionError) as exc:
        out = {"error": str(exc) or type(exc).__name__}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
