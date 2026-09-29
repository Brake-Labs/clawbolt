"""Tests for trimming Gmail bodies and tightening search hits.

``gmail_get_message`` and ``gmail_search`` results stay in the session
history, so what they render is paid for on every later LLM call.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agent.tools.names import ToolName
from backend.app.config import settings
from backend.app.integrations.gmail.body_trim import trim_email_body
from backend.app.integrations.gmail.factory import create_gmail_tools
from backend.app.integrations.gmail.service import (
    MAX_FULL_BODY_CHARS,
    GmailAttachmentInfo,
    GmailMessage,
    GmailMessageSummary,
    GmailService,
)
from tests.mocks.gmail_bodies import (
    FIXTURES,
    GMAIL_REPLY,
    GMAIL_REPLY_DROP,
    GMAIL_REPLY_KEEP,
    PAYMENT_ALERT,
    PAYMENT_ALERT_KEEP,
)


def _message(body: str, subject: str = "Re: Bathroom remodel") -> GmailMessage:
    return GmailMessage(
        id="m1",
        thread_id="t1",
        sender="Bob Builder <bob@acmeplumbing.example.com>",
        recipients=["jane.doe@example.com"],
        cc=["accounts@example.com"],
        subject=subject,
        date="Mon, 28 Sep 2026 09:14:00 -0700",
        body=body,
        links=["https://acmeplumbing.example.com"],
        attachments=[
            GmailAttachmentInfo(
                part_id="2", filename="quote.pdf", mime_type="application/pdf", size=48_000
            )
        ],
    )


async def _get_message(msg: GmailMessage, full_body: bool = False) -> str:
    service = GmailService(access_token="test-access")
    with patch.object(service, "get_message", new_callable=AsyncMock, return_value=msg):
        tool = next(t for t in create_gmail_tools(service) if t.name == ToolName.GMAIL_GET_MESSAGE)
        result = await tool.function("m1", full_body=full_body)
    assert result.is_error is False
    return result.content


# ---------------------------------------------------------------------------
# trim_email_body on realistic fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_keeps_new_content_and_drops_boilerplate(name: str) -> None:
    body, keep, drop = FIXTURES[name]
    trimmed = trim_email_body(body)

    assert trimmed.text == keep
    assert drop not in trimmed.text
    assert trimmed.removed
    assert len(trimmed.text) < len(body) / 2


def test_gmail_reply_cuts_at_wrapped_header() -> None:
    trimmed = trim_email_body(GMAIL_REPLY)
    assert trimmed.removed == ("quoted replies",)
    assert "wrote:" not in trimmed.text
    assert "Bob Builder" not in trimmed.text


def test_payment_alert_keeps_amount_and_drops_every_footer_paragraph() -> None:
    trimmed = trim_email_body(PAYMENT_ALERT)
    assert trimmed.removed == ("footer",)
    assert "$1,250.00" in trimmed.text
    for gone in ("do not reply", "alert preferences", "Unsubscribe", "Privacy Policy", "©"):
        assert gone not in trimmed.text


def test_apple_mail_header_and_mobile_signature() -> None:
    body = (
        "Yes, go ahead with the extra outlet.\n"
        "\n"
        "Sent from my iPhone\n"
        "\n"
        "On Sep 28, 2026, at 7:41 AM, Bob Builder <bob@acmeplumbing.example.com> wrote:\n"
        "\n"
        "> Do you want a second outlet on the island? It adds $180.\n"
    )
    trimmed = trim_email_body(body)
    assert trimmed.text == "Yes, go ahead with the extra outlet."
    assert trimmed.removed == ("quoted replies", "signature")


def test_original_message_separator() -> None:
    body = (
        "Confirmed for Tuesday.\n"
        "\n"
        "-----Original Message-----\n"
        "From: Bob Builder\n"
        "Can you confirm Tuesday?\n"
    )
    assert trim_email_body(body).text == "Confirmed for Tuesday."


def test_trailing_quote_block_without_header_is_dropped() -> None:
    body = "Sounds good, see you then.\n\n> See you at 9?\n> Bob\n"
    trimmed = trim_email_body(body)
    assert trimmed.text == "Sounds good, see you then."
    assert trimmed.removed == ("quoted replies",)


def test_interleaved_reply_keeps_quotes_that_answers_refer_to() -> None:
    body = (
        "> Can you start Thursday?\n"
        "Yes.\n"
        "\n"
        "> And do you need the garage code?\n"
        "No, the side gate is fine.\n"
    )
    trimmed = trim_email_body(body)
    assert trimmed.text == body.strip()
    assert trimmed.removed == ()


def test_body_that_is_all_quote_is_kept() -> None:
    body = (
        "On Mon, Sep 28, 2026 at 9:14 AM Bob <bob@example.com> wrote:\n> The quote is attached.\n"
    )
    trimmed = trim_email_body(body)
    assert "The quote is attached." in trimmed.text
    assert trimmed.removed == ()


def test_gmail_forward_is_not_cut() -> None:
    body = (
        "FYI, see the inspector's notes below.\n"
        "\n"
        "-- \n"
        "Jane\n"
        "\n"
        "---------- Forwarded message ---------\n"
        "From: Sam Inspector <inspector@example.gov>\n"
        "Date: Mon, Sep 28, 2026 at 2:10 PM\n"
        "Subject: Rough-in inspection\n"
        "To: Jane Doe <jane.doe@example.com>\n"
        "\n"
        "Rough-in passed with two corrections.\n"
    )
    trimmed = trim_email_body(body, subject="Fwd: Rough-in inspection")
    assert "Rough-in passed with two corrections." in trimmed.text
    assert trimmed.removed == ()


def test_outlook_forward_is_recognised_by_subject() -> None:
    body = (
        "Please quote this.\n"
        "\n"
        "________________________________\n"
        "From: Pat Manager <pat.manager@example.com>\n"
        "Sent: Monday, September 28, 2026 8:02 AM\n"
        "To: Jane Doe <jane.doe@example.com>\n"
        "Subject: Water heater replacement\n"
        "\n"
        "Unit 4 needs a 50 gallon gas water heater.\n"
    )
    trimmed = trim_email_body(body, subject="FW: Water heater replacement")
    assert "Unit 4 needs a 50 gallon gas water heater." in trimmed.text
    assert trimmed.removed == ()


def test_long_text_after_dashes_is_not_a_signature() -> None:
    body = "Part one.\n--\n" + "\n".join(f"Line {i} of the actual report." for i in range(60))
    assert trim_email_body(body).removed == ()


def test_invisible_preheader_padding_is_removed() -> None:
    trimmed = trim_email_body(
        "Your invoice is ready\u034f \u200c\u034f \u200c\u00a0\n\n\n\nPay now."
    )
    assert trimmed.text == "Your invoice is ready\n\nPay now."


# ---------------------------------------------------------------------------
# gmail_get_message rendering
# ---------------------------------------------------------------------------


async def test_get_message_trims_and_says_how_to_get_the_rest() -> None:
    content = await _get_message(_message(GMAIL_REPLY))

    assert GMAIL_REPLY_KEEP in content
    assert GMAIL_REPLY_DROP not in content
    note = content.splitlines()[-1]
    assert note.startswith("[Removed quoted replies:")
    assert f"of {len(GMAIL_REPLY):,} chars" in note
    assert "full_body=true" in note


async def test_get_message_keeps_every_header_and_attachment() -> None:
    content = await _get_message(_message(GMAIL_REPLY))

    for expected in (
        "From: Bob Builder <bob@acmeplumbing.example.com>",
        "To: jane.doe@example.com",
        "Cc: accounts@example.com",
        "Subject: Re: Bathroom remodel",
        "Date: Mon, 28 Sep 2026 09:14:00 -0700",
        "Message ID: m1",
        "Thread ID: t1",
        "https://acmeplumbing.example.com",
        "quote.pdf (application/pdf, 47 KB) [attachment_id: 2]",
    ):
        assert expected in content


async def test_get_message_without_boilerplate_has_no_note() -> None:
    content = await _get_message(_message("Can you come by Thursday?"))
    assert content.endswith("Body:\nCan you come by Thursday?")


async def test_full_body_returns_the_untrimmed_body() -> None:
    content = await _get_message(_message(GMAIL_REPLY), full_body=True)

    assert GMAIL_REPLY.strip() in content
    assert "[Removed" not in content


async def test_payment_alert_keeps_links_list_after_footer_trim() -> None:
    msg = _message(PAYMENT_ALERT, subject="Deposit received")
    msg.links = [
        "https://bank.example.com/activity",
        "https://bank.example.com/unsubscribe?u=abc123",
    ]
    content = await _get_message(msg)

    assert PAYMENT_ALERT_KEEP in content
    assert "Member FDIC" not in content
    # The footer's unsubscribe URL stays reachable through the links list.
    assert "  - https://bank.example.com/unsubscribe?u=abc123" in content


async def test_long_body_is_capped_with_total_length() -> None:
    body = "Inspection notes.\n" + "x" * 3_000
    with patch.object(settings, "gmail_body_max_chars", 1_000):
        content = await _get_message(_message(body))

    rendered = content.split("Body:\n", 1)[1]
    text, note = rendered.rsplit("\n", 1)
    assert len(text) == 1_000
    assert note == (
        f"[Cut at 1,000 of {len(body):,} chars. "
        f"full_body=true returns up to {MAX_FULL_BODY_CHARS:,}.]"
    )


async def test_full_body_raises_the_cap() -> None:
    body = "y" * (MAX_FULL_BODY_CHARS + 500)
    with patch.object(settings, "gmail_body_max_chars", 1_000):
        content = await _get_message(_message(body), full_body=True)

    rendered = content.split("Body:\n", 1)[1]
    text, note = rendered.rsplit("\n", 1)
    assert len(text) == MAX_FULL_BODY_CHARS
    assert note == f"[Cut at {MAX_FULL_BODY_CHARS:,} of {len(body):,} chars.]"


async def test_service_returns_the_whole_body() -> None:
    """The tool layer needs the full length to report what it cut."""
    service = GmailService(access_token="test-access")
    body = "z" * (MAX_FULL_BODY_CHARS * 2)
    api_resp = {
        "id": "m1",
        "threadId": "t1",
        "payload": {
            "mimeType": "text/plain",
            "body": {"data": base64.urlsafe_b64encode(body.encode()).decode()},
            "headers": [],
        },
    }
    with patch.object(service, "_request", new_callable=AsyncMock, return_value=api_resp):
        msg = await service.get_message("m1")
    assert msg.body == body


# ---------------------------------------------------------------------------
# gmail_search hits
# ---------------------------------------------------------------------------


async def test_search_hit_is_compact() -> None:
    summary = GmailMessageSummary(
        id="18c2f0a1b2c3d4e5",
        thread_id="18c2f0a1b2c3d4e5",
        sender="Example Bank <alerts@bank.example.com>",
        subject="Deposit received",
        date="Mon, 28 Sep 2026 09:14:07 -0700 (PDT)",
        snippet=(
            "A payment of $1,250.00 was deposited to your account ending in 4821. "
            "We&#39;re letting you know \u034f \u200c \u034f \u200c because you enrolled in "
            "deposit alerts for this account and can change them any time."
        ),
    )
    service = GmailService(access_token="test-access")
    with patch.object(service, "search_messages", new_callable=AsyncMock, return_value=[summary]):
        tool = next(t for t in create_gmail_tools(service) if t.name == ToolName.GMAIL_SEARCH)
        result = await tool.function("from:alerts@bank.example.com", 5)

    line = result.content.splitlines()[1]
    sender, subject, date, snippet, ident = line[2:].split(" | ")
    assert sender == "Example Bank <alerts@bank.example.com>"
    assert subject == "Deposit received"
    assert date == "2026-09-28 09:14 -0700"
    assert snippet.startswith("A payment of $1,250.00")
    assert "We're letting you know because" in snippet
    assert "&#39;" not in snippet
    assert "\u034f" not in snippet
    assert snippet.endswith("...")
    assert len(snippet) <= 103
    assert ident == "[id: 18c2f0a1b2c3d4e5]"


async def test_search_hit_keeps_unparseable_date() -> None:
    summary = GmailMessageSummary(
        id="m1", thread_id="t1", sender="a@example.com", subject="Hi", date="sometime", snippet=""
    )
    service = GmailService(access_token="test-access")
    with patch.object(service, "search_messages", new_callable=AsyncMock, return_value=[summary]):
        tool = next(t for t in create_gmail_tools(service) if t.name == ToolName.GMAIL_SEARCH)
        result = await tool.function("q", 5)

    assert result.content.splitlines()[1] == "- a@example.com | Hi | sometime | [id: m1]"
