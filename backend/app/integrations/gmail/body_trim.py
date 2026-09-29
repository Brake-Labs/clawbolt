"""Trim a fetched email body down to the message's own new content.

A ``gmail_get_message`` result stays in the conversation for the rest of the
session and every later LLM call re-reads it. Most of a typical body is not
the message: it is the quoted thread below a reply header, the sender's
signature, and the legal or unsubscribe footer of an automated alert. This
module removes those three things and nothing else.

Every rule errs towards keeping text. A cut that would leave no new content
(a bare forward, a body that is all quote) is skipped, forwarded messages are
never cut, and the tool exposes ``full_body=True`` for when a trim was wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "On Mon, Jan 5, 2026 at 3:14 PM Jane Doe <jane@example.com> wrote:" (Gmail),
# "On Jan 5, 2026, at 3:14 PM, Jane Doe <jane@example.com> wrote:" (Apple
# Mail), plus the common non-English spellings Gmail emits. Clients wrap long
# headers, so this is matched against a line joined with the next one too.
_REPLY_HEADER_RE = re.compile(
    r"^\s*(?:On|Le|Am|El|Op|Il)\s.{0,300}?"
    r"(?:wrote|a écrit|schrieb|escribió|schreef|ha scritto)\s*:\s*$",
    re.IGNORECASE,
)
# Outlook / Exchange / Yahoo separators that open the quoted original.
_ORIGINAL_MESSAGE_RE = re.compile(
    r"^\s*-{2,}\s*(?:Original Message|Reply message|Mensaje original)\s*-{2,}\s*$",
    re.IGNORECASE,
)
# Outlook's header block: a "From:" line followed within a few lines by
# "Sent:" or "Date:" and by "To:" or "Subject:". Outlook bolds the labels in
# its plain-text part as ``*From:*``, so the asterisks are optional.
_OUTLOOK_FROM_RE = re.compile(r"^\s*\*?From:\*?\s+\S", re.IGNORECASE)
_OUTLOOK_SENT_RE = re.compile(r"^\s*\*?(?:Sent|Date):\*?\s+\S", re.IGNORECASE)
_OUTLOOK_TO_RE = re.compile(r"^\s*\*?(?:To|Subject):\*?\s", re.IGNORECASE)
_OUTLOOK_RULE_RE = re.compile(r"^\s*_{10,}\s*$")
_OUTLOOK_HEADER_WINDOW = 5

# A forwarded message is the content, not history, so these stop the cut.
_FORWARD_MARKER_RE = re.compile(
    r"^\s*(?:-{2,}\s*Forwarded message\s*-{2,}|Begin forwarded message:)\s*$",
    re.IGNORECASE,
)
_FORWARD_SUBJECT_RE = re.compile(r"^\s*(?:fwd?|fw)\s*:", re.IGNORECASE)

_QUOTED_LINE_RE = re.compile(r"^\s*>")

# RFC 3676 signature delimiter ("-- "), which many clients send as "--".
_SIG_DELIMITER_RE = re.compile(r"^--\s?$")
# Past this many lines after the delimiter it is probably not a signature.
_MAX_SIGNATURE_LINES = 40
_MOBILE_SIGNATURE_RE = re.compile(
    r"^\s*(?:Sent from my \w+|Sent from (?:Mail|Outlook|Yahoo Mail) for \w+|"
    r"Get Outlook for \w+)\b.{0,40}$",
    re.IGNORECASE,
)

# Trailing paragraphs of automated mail that carry nothing the agent acts on.
# Only paragraphs at the very end of the body are candidates, and the walk
# stops at the first paragraph that does not match.
_FOOTER_RE = re.compile(
    r"unsubscribe|opt[ -]out|manage (?:your )?(?:email |notification |alert )?"
    r"(?:preferences|settings|subscriptions)|please do not (?:reply|respond)|"
    r"(?:mailbox|address) is not monitored|automated (?:message|email|notification)|"
    r"you (?:are )?receiv(?:ed|ing) this (?:e-?mail|message|notification|alert|because)|"
    r"this (?:e-?mail|message) was sent to|"
    r"privacy (?:policy|notice|statement)|terms of (?:use|service)|all rights reserved|"
    r"copyright|©|member fdic|equal housing|\bnmls\b|confidentiality notice|"
    r"privileged (?:and|or) confidential|confidential (?:and|or) privileged|"
    r"intended (?:only |solely )?for the (?:use of the )?(?:named |intended )?"
    r"(?:recipient|addressee|individual)|received this (?:e-?mail|message) in error|"
    r"view (?:it |this email )?in (?:your |a )?(?:web )?browser|add us to your address book",
    re.IGNORECASE,
)
# A paragraph naming an amount is content (a payment alert's own sentence
# often says "automated"), so it is never read as footer.
_AMOUNT_RE = re.compile(r"[$€£]\s?\d")

# Zero-width and padding characters marketing mail stuffs into preheaders.
_INVISIBLE_RE = re.compile("[\u034f\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_BLANK_RUN_RE = re.compile(r"\n[ \t]*(?:\n[ \t]*){2,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass(frozen=True)
class TrimmedBody:
    """A body with its quoted history and boilerplate removed.

    ``removed`` names what was cut (``"quoted replies"``, ``"signature"``,
    ``"footer"``), in body order, and is empty when nothing was.
    """

    text: str
    removed: tuple[str, ...]


def clean_text(text: str) -> str:
    """Drop invisible padding, trailing spaces, and runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE_RE.sub("", text).replace("\u00a0", " ")
    text = _TRAILING_WS_RE.sub("", text)
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


def trim_email_body(body: str, subject: str = "") -> TrimmedBody:
    """Return *body* without quoted history, signature, or trailing footer."""
    lines = clean_text(body).split("\n")
    removed: list[str] = []

    # Outlook forwards carry no marker, only the same header block a reply
    # quotes under, so a forward is recognised by its subject.
    is_forward = bool(_FORWARD_SUBJECT_RE.match(subject))
    cut = None
    if not is_forward:
        cut = _quote_start(lines)
        if cut is None:
            cut = _trailing_quote_start(lines)
    if cut is not None and _has_content(lines[:cut]):
        lines = lines[:cut]
        removed.append("quoted replies")

    # A forwarded message is the content. Its sender's signature, or the
    # forwarder's own above the marker, is not worth risking the forward for.
    is_forward = is_forward or any(_FORWARD_MARKER_RE.match(line) for line in lines)
    if not is_forward:
        cut = _signature_start(lines)
        if cut is not None and _has_content(lines[:cut]):
            lines = lines[:cut]
            removed.append("signature")

    cut = _footer_start(lines)
    if cut is not None and _has_content(lines[:cut]):
        lines = lines[:cut]
        removed.append("footer")

    return TrimmedBody(text="\n".join(lines).strip(), removed=tuple(removed))


def _has_content(lines: list[str]) -> bool:
    return any(line.strip() for line in lines)


def _quote_start(lines: list[str]) -> int | None:
    """Index of the first line of quoted history, or ``None``.

    Stops at a forward marker: whatever follows it is the forwarded message,
    which the user shared on purpose.
    """
    for i, line in enumerate(lines):
        if _FORWARD_MARKER_RE.match(line):
            return None
        if _ORIGINAL_MESSAGE_RE.match(line):
            return i
        if _REPLY_HEADER_RE.match(line):
            return i
        if i + 1 < len(lines) and _REPLY_HEADER_RE.match(f"{line} {lines[i + 1].strip()}"):
            return i
        if _is_outlook_header(lines, i):
            # Outlook puts a horizontal rule above its header block.
            if i > 0 and _OUTLOOK_RULE_RE.match(lines[i - 1]):
                return i - 1
            return i
    return None


def _is_outlook_header(lines: list[str], i: int) -> bool:
    if not _OUTLOOK_FROM_RE.match(lines[i]):
        return False
    window = lines[i + 1 : i + 1 + _OUTLOOK_HEADER_WINDOW]
    return any(_OUTLOOK_SENT_RE.match(w) for w in window) and any(
        _OUTLOOK_TO_RE.match(w) for w in window
    )


def _trailing_quote_start(lines: list[str]) -> int | None:
    """Start of a ``>`` block that runs to the end of the body.

    Only the trailing block goes: ``>`` lines followed by more new text are
    an interleaved reply, and dropping them would orphan the answers.
    """
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    start = end
    while start > 0 and (_QUOTED_LINE_RE.match(lines[start - 1]) or not lines[start - 1].strip()):
        start -= 1
    if start == end or not any(_QUOTED_LINE_RE.match(line) for line in lines[start:end]):
        return None
    # A blank line may precede the block; keep the cut at the first quoted line.
    while start < end and not lines[start].strip():
        start += 1
    return start


def _signature_start(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if _SIG_DELIMITER_RE.match(line) and len(lines) - i - 1 <= _MAX_SIGNATURE_LINES:
            return i
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    if end > 0 and _MOBILE_SIGNATURE_RE.match(lines[end - 1]):
        return end - 1
    return None


def _footer_start(lines: list[str]) -> int | None:
    """Start of the run of footer paragraphs that ends the body, or ``None``."""
    paragraphs: list[tuple[int, int]] = []
    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = i
        elif start is not None:
            paragraphs.append((start, i))
            start = None
    if start is not None:
        paragraphs.append((start, len(lines)))

    cut: int | None = None
    for p_start, p_end in reversed(paragraphs):
        paragraph = " ".join(lines[p_start:p_end])
        if not _FOOTER_RE.search(paragraph) or _AMOUNT_RE.search(paragraph):
            break
        cut = p_start
    return cut
