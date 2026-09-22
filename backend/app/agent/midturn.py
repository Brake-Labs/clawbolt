"""Fold follow-up messages into an agent turn that is already running.

Messaging channels deliver a photo and its caption, or a message and its
correction, as separate inbounds. ``MessageBatcher`` only coalesces the ones
that land within its window, and an agent turn routinely runs for 20 to 150
seconds. A follow-up that arrives mid-turn used to wait on the per-user lock
and then run a second turn, so the user got two replies: the first answering
without the follow-up, the second contradicting it.

The fix is a per-user inbox of dispatches that are waiting on the lock:

- ``_dispatch_to_pipeline`` registers each waiting dispatch here, then waits
  for either the lock or for the running turn to claim it (see
  ``run_unless_folded``).
- The running turn drains the inbox before every LLM call, including the
  point where it would otherwise send its final reply. Claimed messages join
  the turn's context as user messages and the claimed dispatch exits without
  running a turn of its own.

Only dispatches for the same session and channel, without a webchat
``request_id``, are foldable. A webchat request owns an SSE response future
that expects its own reply, and a cross-channel follow-up would be answered
on the wrong channel. Heartbeat turns never drain the inbox: they are not
replies to the user and do not hold the per-user lock.

Approval replies never reach this inbox. Ingestion hands them to the pending
approval gate before the message is persisted or dispatched.

The inbox is process-local, like ``user_locks``: both assume one worker
serves a given user's inbound traffic.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from backend.app.agent.dto import StoredMessage
from backend.app.media.download import DownloadedMedia

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class PendingInbound:
    """One dispatch waiting on the per-user lock.

    ``messages`` holds every persisted inbound the dispatch covers, in seq
    order: one for a direct dispatch, several for a ``MessageBatcher``
    flush. As in the normal path, the batch's media is merged onto the
    last message.
    """

    user_id: str
    session_id: str
    channel: str
    messages: list[StoredMessage]
    media_urls: list[tuple[str, str]] = field(default_factory=list)
    downloaded_media: list[DownloadedMedia] = field(default_factory=list)
    download_media: Callable[[str], Awaitable[DownloadedMedia]] | None = None
    consumed: asyncio.Event = field(default_factory=asyncio.Event)


class MidTurnInbox:
    """Per-user registry of dispatches a running turn may fold in."""

    def __init__(self) -> None:
        self._pending: dict[str, list[PendingInbound]] = {}

    def register(self, entry: PendingInbound) -> None:
        self._pending.setdefault(entry.user_id, []).append(entry)

    def unregister(self, entry: PendingInbound) -> None:
        entries = self._pending.get(entry.user_id)
        if not entries:
            return
        if entry in entries:
            entries.remove(entry)
        if not entries:
            del self._pending[entry.user_id]

    def claim(self, user_id: str, session_id: str, channel: str) -> list[PendingInbound]:
        """Remove and return the matching entries, marking each consumed.

        Synchronous on purpose: no other task can observe an entry between
        its removal and its ``consumed`` flag being set.
        """
        entries = self._pending.get(user_id)
        if not entries:
            return []
        claimed = [e for e in entries if e.session_id == session_id and e.channel == channel]
        if not claimed:
            return []
        remaining = [e for e in entries if e not in claimed]
        if remaining:
            self._pending[user_id] = remaining
        else:
            del self._pending[user_id]
        for entry in claimed:
            entry.consumed.set()
        return claimed

    def pending_count(self, user_id: str) -> int:
        return len(self._pending.get(user_id, []))

    def clear(self) -> None:
        self._pending.clear()


midturn_inbox = MidTurnInbox()


async def run_unless_folded(coro: Coroutine[Any, Any, None], consumed: asyncio.Event) -> bool:
    """Run *coro* unless a running turn claims its message first.

    *coro* is expected to wait for the per-user lock before doing any work.
    Returns True when it ran to completion, False when ``consumed`` fired
    first and it was cancelled while still waiting. The claim can only
    happen while another task holds the lock, and the entry is unregistered
    as soon as *coro* acquires it, so a cancelled *coro* has not started the
    pipeline. Cancelling this function (e.g. the dispatch timeout) cancels
    *coro* too.
    """
    run = asyncio.ensure_future(coro)
    folded = asyncio.ensure_future(consumed.wait())
    try:
        await asyncio.wait({run, folded}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        folded.cancel()
        if not run.done():
            run.cancel()
            await asyncio.wait({run})
    if run.cancelled():
        return False
    run.result()
    return True
