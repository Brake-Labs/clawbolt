import sys
import time
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agent import media_staging
from backend.app.media import pdf as pdf_module
from backend.app.media.download import DownloadedMedia
from backend.app.media.pdf import PdfExtractionError, extract_pdf_text
from backend.app.media.pipeline import (
    VISION_FALLBACK,
    process_message_media,
    run_vision_on_media,
)
from backend.app.models import User
from tests.mocks.pdf import make_content_stream_bomb, make_text_pdf


def _make_media(
    mime_type: str = "image/jpeg", url: str = "https://example.com/media"
) -> DownloadedMedia:
    return DownloadedMedia(
        content=b"fake-bytes",
        mime_type=mime_type,
        original_url=url,
        filename="test_file",
    )


@patch("backend.app.media.pipeline.analyze_image", new_callable=AsyncMock)
async def test_process_single_image_stages_without_vision(
    mock_vision: AsyncMock, test_user: User
) -> None:
    """Pipeline classifies the image and leaves vision for the agent via
    analyze_photo. The combined context surfaces the staging handle."""
    await media_staging.clear_user(test_user.id)
    await media_staging.stage(
        test_user.id, "https://example.com/media", b"fake-bytes", "image/jpeg"
    )
    result = await process_message_media(
        "Check this deck", [_make_media("image/jpeg")], user_id=test_user.id
    )
    assert len(result.media_results) == 1
    assert result.media_results[0].category == "image"
    assert result.media_results[0].extracted_text == ""
    assert mock_vision.await_count == 0
    assert "Photo 1" in result.combined_context
    assert "Check this deck" in result.combined_context
    assert "call analyze_photo" in result.combined_context
    await media_staging.clear_user(test_user.id)


async def test_process_text_only() -> None:
    """Text-only message (no media) should produce a simple context."""
    result = await process_message_media("Just a text message", [])
    assert result.text_body == "Just a text message"
    assert len(result.media_results) == 0
    assert "Just a text message" in result.combined_context


async def test_text_only_message_has_no_wrapper() -> None:
    """Text-only messages pass through verbatim, without the "[Text message]:"
    delimiter. The wrapper only earns its keep next to media parts; alone it
    is formatting noise the model can parrot back as its own reply."""
    result = await process_message_media("Yes", [])
    assert result.combined_context == "Yes"


@patch("backend.app.media.pipeline.analyze_image", new_callable=AsyncMock)
async def test_text_with_media_keeps_wrapper(mock_vision: AsyncMock, test_user: User) -> None:
    """When media parts are present, the text keeps its delimiting wrapper."""
    await media_staging.clear_user(test_user.id)
    await media_staging.stage(
        test_user.id, "https://example.com/media", b"fake-bytes", "image/jpeg"
    )
    result = await process_message_media(
        "Check this deck", [_make_media("image/jpeg")], user_id=test_user.id
    )
    assert result.combined_context.startswith("[Text message]: 'Check this deck'")
    assert "Photo 1" in result.combined_context
    await media_staging.clear_user(test_user.id)


async def test_process_unknown_media_type() -> None:
    """Unknown media type should be skipped gracefully with a placeholder."""
    result = await process_message_media("", [_make_media("application/octet-stream")])
    assert len(result.media_results) == 1
    assert result.media_results[0].category == "unknown"


@patch(
    "backend.app.media.vision.analyze_image",
    new_callable=AsyncMock,
    side_effect=RuntimeError("Vision API rate limit exceeded"),
)
async def test_run_vision_on_media_failure_produces_fallback(mock_vision: AsyncMock) -> None:
    """run_vision_on_media (called by the analyze_photo tool) returns a fallback
    string when the vision API raises, instead of propagating the exception."""
    result = await run_vision_on_media(b"bytes", "image/jpeg", "Check this roof")
    assert result == VISION_FALLBACK


@patch(
    "backend.app.media.vision.analyze_image",
    new_callable=AsyncMock,
    side_effect=TimeoutError("Connection timed out"),
)
async def test_run_vision_on_media_timeout_produces_fallback(mock_vision: AsyncMock) -> None:
    """Timeouts from the vision API are caught and surface as the fallback."""
    result = await run_vision_on_media(b"bytes", "image/png", "")
    assert result == VISION_FALLBACK


@patch("backend.app.media.pipeline.analyze_image", new_callable=AsyncMock)
async def test_image_document_classified_as_image(mock_vision: AsyncMock) -> None:
    """Images sent as documents with image/* MIME type should be classified as
    images. Vision is never called from the pipeline."""
    result = await process_message_media("", [_make_media("image/png")])
    assert len(result.media_results) == 1
    assert result.media_results[0].category == "image"
    assert mock_vision.await_count == 0


async def test_pdf_text_is_extracted_into_context() -> None:
    """A PDF reaches the agent as its text layer, not a placeholder."""
    media = DownloadedMedia(
        content=make_text_pdf("Inspection summary: replace GFCI outlet"),
        mime_type="application/pdf",
        original_url="https://example.com/report.pdf",
        filename="report.pdf",
    )
    result = await process_message_media("", [media])
    assert result.media_results[0].category == "pdf"
    assert "replace GFCI outlet" in result.combined_context
    assert "(PDF text, 1 page(s))" in result.combined_context


async def test_scanned_pdf_reports_no_text_layer() -> None:
    media = DownloadedMedia(
        content=make_text_pdf(""),
        mime_type="application/pdf",
        original_url="https://example.com/scan.pdf",
        filename="scan.pdf",
    )
    result = await process_message_media("", [media])
    assert "has no text layer" in result.combined_context


async def test_corrupt_pdf_reports_unreadable() -> None:
    result = await process_message_media("", [_make_media("application/pdf")])
    assert "PDF could not be read" in result.combined_context


async def test_long_pdf_text_is_truncated() -> None:
    text, pages = await extract_pdf_text(make_text_pdf("word " * 200, "more"), max_chars=100)
    assert pages == 2
    assert text.endswith("[...truncated]")
    assert len(text) < 150


async def test_pdf_that_parses_too_long_is_cut_off() -> None:
    """A few-KB PDF whose pages share a large content stream would pin a CPU
    for minutes; the worker is killed at the timeout instead."""
    bomb = make_content_stream_bomb(2_000_000, pages=50)
    assert len(bomb) < 20_000
    started = time.monotonic()
    with (
        patch.object(pdf_module, "PDF_TIMEOUT_SECONDS", 1.0),
        pytest.raises(PdfExtractionError, match="too long"),
    ):
        await extract_pdf_text(bomb)
    assert time.monotonic() - started < 10


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is enforced on Linux")
async def test_pdf_that_needs_too_much_memory_is_refused() -> None:
    bomb = make_content_stream_bomb(20_000_000)
    with (
        patch.object(pdf_module, "PDF_MAX_MEMORY_BYTES", 256 * 1024 * 1024),
        pytest.raises(PdfExtractionError, match="too large or complex"),
    ):
        await extract_pdf_text(bomb)


async def test_pdf_worker_failure_reaches_context_as_unreadable() -> None:
    media = DownloadedMedia(
        content=make_content_stream_bomb(2_000_000, pages=50),
        mime_type="application/pdf",
        original_url="https://example.com/bomb.pdf",
        filename="bomb.pdf",
    )
    with patch.object(pdf_module, "PDF_TIMEOUT_SECONDS", 1.0):
        result = await process_message_media("", [media])
    assert "PDF could not be read" in result.combined_context
