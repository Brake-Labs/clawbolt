"""Gmail REST API client using httpx.

Mirrors the shape of ``calendar/service.py``: an httpx-based client with a
reactive 401 retry that refreshes through the shared, locked OAuth refresh.

Why a hand-rolled client and not ``google-api-python-client``: the rest of
this codebase is httpx-based, the surface area we need from Gmail is tiny
(four reads + one write), and the official client drags in synchronous
discovery-document fetches we'd have to wrap to keep ``async``.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Any, NamedTuple

import httpx

logger = logging.getLogger(__name__)

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
# Cap the body slice we surface to the LLM so a marketing newsletter doesn't
# eat the context window. Callers asking for "the magic link" only need the
# first chunk; full retrieval of long bodies is intentionally out of scope.
_MAX_BODY_CHARS = 16_000

# A Gmail search returning thousands of message IDs would be useless to the
# LLM and expensive to fetch. Hard ceiling at the API's per-page max of 500.
_MAX_RESULTS_CEILING = 500

_URL_RE = re.compile(r"https?://[^\s<>\"')]+")


@dataclass
class GmailMessageSummary:
    """Lightweight summary returned by search / list-recent."""

    id: str
    thread_id: str
    sender: str
    subject: str
    date: str
    snippet: str


@dataclass
class GmailAttachmentInfo:
    """An attachment found on a received message.

    ``part_id`` is the MIME part path (``"1"``, ``"2.1"``) and is what the
    agent quotes back as the attachment id. Gmail's own ``attachmentId`` is
    not stable: every ``messages.get`` mints a fresh value for the same part,
    so it cannot survive a round trip through the conversation.

    Small parts carry their bytes inline in the payload (``inline_data``);
    larger ones only have ``gmail_attachment_id`` and are fetched with
    ``messages.attachments.get``.
    """

    part_id: str
    filename: str
    mime_type: str
    size: int
    inline: bool = False
    gmail_attachment_id: str = ""
    inline_data: str = field(default="", repr=False)


@dataclass
class GmailMessage:
    """Full message returned by ``get_message``.

    ``links`` is a deduplicated list of every URL the body contains, in
    first-seen order. The Gmail API returns the body either as plain text
    or as HTML depending on the part; we prefer ``text/plain`` and fall
    back to a stripped-tags rendering of ``text/html`` so the agent always
    has something readable to work with.
    """

    id: str
    thread_id: str
    sender: str
    recipients: list[str]
    cc: list[str]
    subject: str
    date: str
    body: str
    links: list[str] = field(default_factory=list)
    rfc822_message_id: str = ""
    attachments: list[GmailAttachmentInfo] = field(default_factory=list)


@dataclass
class GmailSendResult:
    """Outcome of a successful send."""

    id: str
    thread_id: str


class GmailAttachment(NamedTuple):
    """A single attachment to include on an outbound Gmail message.

    The Gmail API takes a fully assembled RFC 5322 message and base64-encodes
    it, so the only thing we need at this layer is bytes + MIME hints +
    filename. Resolving a storage path to bytes happens in the tool layer
    (see ``gmail_send`` in ``factory.py``), not here, so the service stays
    backend-agnostic and easy to unit test.
    """

    content: bytes
    maintype: str
    subtype: str
    filename: str


class GmailService:
    """Gmail API client bound to one user's tokens."""

    def __init__(
        self,
        access_token: str,
        refresh_access_token: Callable[[str], Awaitable[str | None]] | None = None,
        sender_email: str = "",
    ) -> None:
        """``refresh_access_token`` is called with the access token Google just
        answered 401 to and returns a fresh one, or None when no refresh could
        run. It owns the OAuth side (locking, persistence, retiring a dead
        grant) and raises ``ReconnectRequired`` when the grant is dead.
        """
        self._access_token = access_token
        self._refresh_access_token = refresh_access_token
        # Resolved lazily on first ``send_message`` call so we don't pay
        # the round-trip on read-only flows. ``getProfile`` is the only
        # Gmail endpoint that returns the authenticated user's address;
        # the OAuth ``id_token`` is not in scope here.
        self._sender_email = sender_email

    @property
    def provider_name(self) -> str:
        return "gmail"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: Mapping[str, str | list[str]] | None = None,
    ) -> dict[str, Any] | None:
        """Make an authenticated Gmail API request, refreshing once on a 401.

        Raises ``ReconnectRequired`` when the refresh finds the grant dead; a
        401 that persists after the refresh is raised as the
        ``HTTPStatusError`` it is.
        """
        url = f"{GMAIL_API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(method, url, headers=headers, json=json, params=params)
            if resp.status_code == 401 and self._refresh_access_token is not None:
                new_token = await self._refresh_access_token(self._access_token)
                if new_token:
                    self._access_token = new_token
                    headers["Authorization"] = f"Bearer {new_token}"
                    resp = await client.request(
                        method, url, headers=headers, json=json, params=params
                    )
            resp.raise_for_status()
            if resp.status_code == 204 or not resp.content:
                return None
            return resp.json()

    # -- Public API -----------------------------------------------------------

    async def get_profile(self) -> dict[str, Any]:
        """Return ``users.me.getProfile`` (caches ``emailAddress`` for sends)."""
        data = await self._request("GET", "/users/me/profile") or {}
        if not self._sender_email:
            self._sender_email = data.get("emailAddress", "")
        return data

    async def search_messages(
        self,
        query: str,
        max_results: int,
    ) -> list[GmailMessageSummary]:
        """Run a Gmail search; returns light summaries for each hit.

        Gmail's ``messages.list`` only returns IDs, so we follow up with a
        ``metadata`` fetch per ID. We cap *max_results* at
        ``_MAX_RESULTS_CEILING`` so the agent can't accidentally hammer the
        API with a 10000-row query.
        """
        capped = max(1, min(int(max_results), _MAX_RESULTS_CEILING))
        params: dict[str, str] = {"maxResults": str(capped)}
        if query:
            params["q"] = query
        listing = await self._request("GET", "/users/me/messages", params=params) or {}
        ids = [item.get("id", "") for item in listing.get("messages", []) if item.get("id")]

        summaries: list[GmailMessageSummary] = []
        for msg_id in ids:
            try:
                summaries.append(await self._get_message_summary(msg_id))
            except httpx.HTTPStatusError as exc:
                # A single 404 (message deleted between list and get) shouldn't
                # nuke the whole response. Log + skip; the LLM still sees the rest.
                logger.warning(
                    "Skipping Gmail message %s during search: %s",
                    msg_id,
                    exc.response.status_code,
                )
                continue
        return summaries

    async def _get_message_summary(self, message_id: str) -> GmailMessageSummary:
        # metadataHeaders is a repeated param: httpx encodes the list as
        # ?metadataHeaders=From&metadataHeaders=Subject&... A comma-joined
        # string matches no header, so Gmail would return none.
        params: dict[str, str | list[str]] = {
            "format": "metadata",
            "metadataHeaders": ["From", "Subject", "Date"],
        }
        data = await self._request("GET", f"/users/me/messages/{message_id}", params=params) or {}
        headers = _index_headers(data.get("payload", {}).get("headers", []))
        return GmailMessageSummary(
            id=data.get("id", ""),
            thread_id=data.get("threadId", ""),
            sender=headers.get("from", ""),
            subject=headers.get("subject", ""),
            date=headers.get("date", ""),
            snippet=data.get("snippet", ""),
        )

    async def get_message(self, message_id: str) -> GmailMessage:
        """Return the full message including a readable body and link list."""
        data = (
            await self._request(
                "GET", f"/users/me/messages/{message_id}", params={"format": "full"}
            )
            or {}
        )
        payload = data.get("payload", {})
        headers = _index_headers(payload.get("headers", []))
        body = _extract_body(payload)
        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + "\n[...truncated]"
        links = _extract_links(body)
        return GmailMessage(
            id=data.get("id", ""),
            thread_id=data.get("threadId", ""),
            sender=headers.get("from", ""),
            recipients=_split_addresses(headers.get("to", "")),
            cc=_split_addresses(headers.get("cc", "")),
            subject=headers.get("subject", ""),
            date=headers.get("date", ""),
            body=body,
            links=links,
            rfc822_message_id=headers.get("message-id", ""),
            attachments=_collect_attachments(payload),
        )

    async def get_attachment_bytes(self, message_id: str, attachment: GmailAttachmentInfo) -> bytes:
        """Return the decoded bytes of *attachment* on *message_id*.

        Uses the inline payload data when Gmail included it, otherwise
        fetches ``messages.attachments.get``.
        """
        data = attachment.inline_data
        if not data:
            if not attachment.gmail_attachment_id:
                raise ValueError(f"Attachment {attachment.part_id} has no retrievable content")
            resp = (
                await self._request(
                    "GET",
                    f"/users/me/messages/{message_id}/attachments/{attachment.gmail_attachment_id}",
                )
                or {}
            )
            data = resp.get("data") or ""
        return _b64url_decode(data)

    async def send_message(
        self,
        to: list[str],
        subject: str,
        body: str,
        reply_to_message_id: str = "",
        attachments: list[GmailAttachment] | None = None,
    ) -> GmailSendResult:
        """Send a new message, optionally threading onto an existing message.

        When *reply_to_message_id* is non-empty we fetch the original message
        to pull its ``Message-ID`` and ``References`` headers so the reply
        threads correctly in Gmail (and any other RFC 5322 client). The
        ``threadId`` field on the send body is what makes Gmail's UI bundle
        the messages; the headers cover everyone else.

        *attachments* is a list of ``GmailAttachment`` records; when present
        the message is built as ``multipart/mixed`` with the body as the
        first part and each attachment as a subsequent MIME part.
        """
        if not self._sender_email:
            await self.get_profile()
        if not to:
            raise ValueError("send_message requires at least one recipient")

        thread_id = ""
        in_reply_to = ""
        references = ""
        if reply_to_message_id:
            original = await self.get_message(reply_to_message_id)
            thread_id = original.thread_id
            in_reply_to = original.rfc822_message_id
            # Per RFC 5322 section 3.6.4, References is built by appending
            # the parent's Message-ID to the parent's References (or
            # In-Reply-To if References is absent). We don't have the
            # parent's References parsed out, so the simpler chain of
            # just the parent's ID is acceptable for a one-deep reply.
            references = in_reply_to

        rfc822 = _build_rfc822(
            sender=self._sender_email,
            to=to,
            subject=subject,
            body=body,
            in_reply_to=in_reply_to,
            references=references,
            attachments=attachments or [],
        )
        raw_b64 = base64.urlsafe_b64encode(rfc822).decode("ascii")
        send_body: dict[str, Any] = {"raw": raw_b64}
        if thread_id:
            send_body["threadId"] = thread_id

        data = await self._request("POST", "/users/me/messages/send", json=send_body) or {}
        return GmailSendResult(
            id=data.get("id", ""),
            thread_id=data.get("threadId", ""),
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _index_headers(headers: list[dict[str, Any]]) -> dict[str, str]:
    """Return a case-insensitive header lookup keyed by lowercase name."""
    out: dict[str, str] = {}
    for h in headers:
        name = (h.get("name") or "").lower()
        if name and name not in out:
            out[name] = h.get("value", "")
    return out


def _extract_body(payload: dict[str, Any]) -> str:
    """Walk a Gmail ``payload`` tree and return the best readable body.

    Preference: ``text/plain`` over ``text/html``. When only ``text/html``
    is present we strip tags with a tiny regex so the agent gets something
    legible without dragging in BeautifulSoup. Multipart bodies are walked
    depth-first.
    """
    text_part = _find_part(payload, "text/plain")
    if text_part is not None:
        decoded = _decode_part_data(text_part)
        if decoded:
            return decoded
    html_part = _find_part(payload, "text/html")
    if html_part is not None:
        decoded = _decode_part_data(html_part)
        if decoded:
            return _strip_tags(decoded)
    # Fall back to the top-level body (common for short single-part messages).
    decoded = _decode_part_data(payload)
    return decoded or ""


def _find_part(payload: dict[str, Any], mime_type: str) -> dict[str, Any] | None:
    # A named part is an attachment (a .txt or .html file), not the body.
    if (
        payload.get("mimeType") == mime_type
        and payload.get("body", {}).get("data")
        and not payload.get("filename")
    ):
        return payload
    for part in payload.get("parts", []) or []:
        found = _find_part(part, mime_type)
        if found is not None:
            return found
    return None


def _b64url_decode(data: str) -> bytes:
    """Decode Gmail's URL-safe base64, which comes without padding guarantees."""
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _collect_attachments(payload: dict[str, Any]) -> list[GmailAttachmentInfo]:
    """Walk a Gmail ``payload`` tree and return every attachment part.

    A part is an attachment when it has a filename, or when it is a
    non-text, non-container part with content (an unnamed image, say).
    Text parts without a filename are the message body, even when Gmail
    moved a long body behind an ``attachmentId``. An attachment's own
    subparts (a forwarded ``message/rfc822``) are not listed separately.
    """
    found: list[GmailAttachmentInfo] = []

    def walk(part: dict[str, Any]) -> None:
        mime = (part.get("mimeType") or "").lower()
        filename = part.get("filename") or ""
        body = part.get("body", {}) or {}
        has_content = bool(body.get("attachmentId") or body.get("data"))
        is_body_text = not filename and mime.startswith("text/")
        if has_content and not is_body_text and not mime.startswith("multipart/"):
            part_id = part.get("partId") or ""
            headers = _index_headers(part.get("headers", []) or [])
            disposition = headers.get("content-disposition", "").lower()
            inline = disposition.startswith("inline") or (
                not disposition.startswith("attachment") and bool(headers.get("content-id"))
            )
            found.append(
                GmailAttachmentInfo(
                    part_id=part_id,
                    filename=filename or f"part-{part_id or len(found) + 1}",
                    mime_type=mime or "application/octet-stream",
                    size=int(body.get("size") or 0),
                    inline=inline,
                    gmail_attachment_id=body.get("attachmentId") or "",
                    inline_data=body.get("data") or "",
                )
            )
            return
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)
    return found


def _decode_part_data(part: dict[str, Any]) -> str:
    body = part.get("body", {}) or {}
    data = body.get("data") or ""
    if not data:
        return ""
    try:
        raw = _b64url_decode(data)
    except (ValueError, TypeError) as exc:
        logger.warning("Could not decode Gmail body part: %s", exc)
        return ""
    return raw.decode("utf-8", errors="replace")


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")
# Match the whole opening anchor tag (not just the href attribute) so the URL
# can be placed OUTSIDE the tag boundaries before the generic tag stripper
# runs; otherwise the hoisted URL would land inside ``<a ... URL >`` and be
# eaten by ``_TAG_RE``.
_ANCHOR_OPEN_RE = re.compile(r"""<a\b[^>]*?href\s*=\s*['"]([^'"\s>]+)['"][^>]*>""", re.IGNORECASE)


def _strip_tags(html: str) -> str:
    """Best-effort HTML to text conversion.

    Anchor ``href`` URLs are surfaced inline before the tags get stripped, so
    a magic-link email rendered as ``<a href="https://x">click</a>`` still
    leaves the URL in the body for ``_extract_links`` to pick up.
    """
    hoisted = _ANCHOR_OPEN_RE.sub(r" \1 ", html)
    text = _TAG_RE.sub(" ", hoisted)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def _extract_links(body: str) -> list[str]:
    """Pull URLs out of a body in first-seen order, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _URL_RE.finditer(body):
        url = match.group(0).rstrip(".,);]>")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _split_addresses(value: str) -> list[str]:
    """Split a comma-separated address header into trimmed entries."""
    if not value:
        return []
    return [addr.strip() for addr in value.split(",") if addr.strip()]


def _build_rfc822(
    *,
    sender: str,
    to: list[str],
    subject: str,
    body: str,
    in_reply_to: str = "",
    references: str = "",
    attachments: list[GmailAttachment] | None = None,
) -> bytes:
    """Build an RFC 5322 message ready for base64url Gmail send.

    When *attachments* is non-empty, ``EmailMessage.add_attachment`` upgrades
    the message to ``multipart/mixed`` automatically: the text body becomes
    the first part, each attachment a sibling part with its own
    ``Content-Type`` and ``Content-Disposition: attachment`` header.
    """
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=False)
    msg["Message-ID"] = make_msgid()
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references
    msg.set_content(body)
    for att in attachments or []:
        msg.add_attachment(
            att.content,
            maintype=att.maintype,
            subtype=att.subtype,
            filename=att.filename,
        )
    return bytes(msg)


__all__ = [
    "GmailAttachment",
    "GmailAttachmentInfo",
    "GmailMessage",
    "GmailMessageSummary",
    "GmailSendResult",
    "GmailService",
]
