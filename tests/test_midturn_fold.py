"""Messages that arrive while a turn is running fold into that turn.

Messaging channels deliver a photo and its caption as separate inbounds, and
a turn routinely outlives ``MessageBatcher``'s window. Without folding, the
caption waited on the user lock and ran a second turn, so the user got two
replies that contradicted each other. See ``backend/app/agent/midturn.py``.
"""

import asyncio
import copy
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import MessageResponse
from pydantic import BaseModel

from backend.app.agent.approval import ApprovalDecision, get_approval_gate
from backend.app.agent.concurrency import user_locks
from backend.app.agent.core import ClawboltAgent
from backend.app.agent.dto import SessionState, StoredMessage
from backend.app.agent.ingestion import (
    InboundMessage,
    _dispatch_to_pipeline,
    process_inbound_from_bus,
)
from backend.app.agent.messages import UserMessage
from backend.app.agent.midturn import midturn_inbox
from backend.app.agent.session_db import get_session_store
from backend.app.agent.tools.base import Tool, ToolResult
from backend.app.bus import OutboundMessage, message_bus
from backend.app.media.download import DownloadedMedia
from backend.app.models import User
from tests.mocks.llm import make_text_response, make_tool_call_response

PHOTO_TEXT = "Here are the fixtures in the hallway."
CAPTION_TEXT = "This photo is the 2 you were unsure of, plus an 8th you missed."
STALE_REPLY = "I count 6 fixtures. Which ones were you unsure about?"
COMBINED_REPLY = "With your note that makes 8 fixtures in total."


def _user_texts(messages: list[dict[str, Any]]) -> list[str]:
    """Plain-string user turns in an ``amessages`` payload."""
    texts: list[str] = []
    for m in messages:
        if m["role"] != "user":
            continue
        content = m["content"]
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.extend(
                b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
    return texts


def _drain_replies() -> list[OutboundMessage]:
    replies: list[OutboundMessage] = []
    while not message_bus.outbound.empty():
        msg = message_bus.outbound.get_nowait()
        if not msg.is_typing_indicator:
            replies.append(msg)
    return replies


async def _await_inbox(user_id: str, count: int, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while midturn_inbox.pending_count(user_id) != count:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"inbox for {user_id} never reached {count}")
        await asyncio.sleep(0.005)


class _BlockingLLM:
    """``amessages`` stand-in whose first call blocks until released."""

    def __init__(self, responses: list[MessageResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []
        self.first_call_started = asyncio.Event()
        self.release_first_call = asyncio.Event()

    async def __call__(self, **kwargs: Any) -> MessageResponse:
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        if len(self.calls) == 1:
            self.first_call_started.set()
            await self.release_first_call.wait()
        return self.responses[min(len(self.calls), len(self.responses)) - 1]


@pytest.fixture(autouse=True)
def _no_batching() -> Any:
    with patch("backend.app.agent.ingestion.settings.message_batch_window_ms", 0):
        yield
    midturn_inbox.clear()


async def _stored_messages(user_id: str) -> list[StoredMessage]:
    session, _ = await get_session_store(user_id).get_or_create_session()
    return sorted(session.messages, key=lambda m: m.seq)


# ---------------------------------------------------------------------------
# End to end through ingestion
# ---------------------------------------------------------------------------


async def test_message_during_turn_yields_one_reply_covering_both(test_user: User) -> None:
    """The production scenario: a follow-up lands while the first turn's LLM
    call is in flight. One reply goes out, and the LLM saw both messages."""
    llm = _BlockingLLM([make_text_response(STALE_REPLY), make_text_response(COMBINED_REPLY)])
    first = InboundMessage(
        channel="telegram", sender_id=test_user.channel_identifier, text=PHOTO_TEXT
    )
    second = InboundMessage(
        channel="telegram", sender_id=test_user.channel_identifier, text=CAPTION_TEXT
    )

    with patch("backend.app.agent.core.amessages", new=llm):
        first_task = asyncio.create_task(process_inbound_from_bus(first))
        await asyncio.wait_for(llm.first_call_started.wait(), timeout=10)

        second_task = asyncio.create_task(process_inbound_from_bus(second))
        await _await_inbox(test_user.id, 1)
        llm.release_first_call.set()

        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=15)

    replies = _drain_replies()
    assert [r.content for r in replies] == [COMBINED_REPLY]

    # The draft was discarded and one more round ran; no second turn.
    assert len(llm.calls) == 2
    final_user_texts = _user_texts(llm.calls[-1])
    assert any(PHOTO_TEXT in t for t in final_user_texts)
    assert any(CAPTION_TEXT in t for t in final_user_texts)
    assert not any(STALE_REPLY in str(m) for m in llm.calls[-1])

    stored = await _stored_messages(test_user.id)
    assert [(m.direction, m.body) for m in stored] == [
        ("inbound", PHOTO_TEXT),
        ("inbound", CAPTION_TEXT),
        ("outbound", COMBINED_REPLY),
    ]
    assert midturn_inbox.pending_count(test_user.id) == 0


async def test_message_during_tool_round_is_folded_before_next_llm_call(
    test_user: User,
) -> None:
    """A follow-up that lands while tools run joins before the next LLM call."""

    class _NoParams(BaseModel):
        pass

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def slow_lookup() -> ToolResult:
        tool_started.set()
        await release_tool.wait()
        return ToolResult(content="6 fixtures on file")

    calls: list[list[dict[str, Any]]] = []

    async def fake_llm(**kwargs: Any) -> MessageResponse:
        calls.append(copy.deepcopy(kwargs["messages"]))
        if len(calls) == 1:
            return make_tool_call_response([{"name": "slow_lookup", "arguments": {}}])
        return make_text_response(COMBINED_REPLY)

    async def fake_assemble(*_args: Any, **_kwargs: Any) -> tuple[list[Tool], list[Any]]:
        return [
            Tool(
                name="slow_lookup",
                description="Look up fixtures",
                function=slow_lookup,
                params_model=_NoParams,
            )
        ], []

    first = InboundMessage(
        channel="telegram", sender_id=test_user.channel_identifier, text=PHOTO_TEXT
    )
    second = InboundMessage(
        channel="telegram", sender_id=test_user.channel_identifier, text=CAPTION_TEXT
    )
    with (
        patch("backend.app.agent.core.amessages", side_effect=fake_llm),
        patch("backend.app.agent.router.assemble_turn_tools", side_effect=fake_assemble),
    ):
        first_task = asyncio.create_task(process_inbound_from_bus(first))
        await asyncio.wait_for(tool_started.wait(), timeout=10)
        second_task = asyncio.create_task(process_inbound_from_bus(second))
        await _await_inbox(test_user.id, 1)
        release_tool.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=15)

    assert [r.content for r in _drain_replies()] == [COMBINED_REPLY]
    assert len(calls) == 2
    # The folded message follows the tool result it arrived after.
    last = calls[1]
    assert last[-1]["role"] == "user"
    assert CAPTION_TEXT in last[-1]["content"]
    assert last[-2]["content"][0]["type"] == "tool_result"

    stored = await _stored_messages(test_user.id)
    assert [m.direction for m in stored] == ["inbound", "inbound", "outbound"]
    assert stored[1].seq < stored[2].seq
    assert "slow_lookup" in stored[2].tool_interactions_json


async def test_folded_message_media_goes_through_media_pipeline(test_user: User) -> None:
    """A folded photo is downloaded, staged, and described like any inbound."""
    llm = _BlockingLLM([make_text_response(STALE_REPLY), make_text_response(COMBINED_REPLY)])
    download = AsyncMock(
        return_value=DownloadedMedia(
            content=b"fake-jpeg",
            mime_type="image/jpeg",
            original_url="file-photo-2",
            filename="photo.jpg",
        )
    )
    first = InboundMessage(
        channel="telegram", sender_id=test_user.channel_identifier, text=PHOTO_TEXT
    )
    second = InboundMessage(
        channel="telegram",
        sender_id=test_user.channel_identifier,
        text=CAPTION_TEXT,
        media_refs=[("file-photo-2", "image/jpeg")],
    )

    with patch("backend.app.agent.core.amessages", new=llm):
        first_task = asyncio.create_task(process_inbound_from_bus(first, download_media=download))
        await asyncio.wait_for(llm.first_call_started.wait(), timeout=10)
        second_task = asyncio.create_task(process_inbound_from_bus(second, download_media=download))
        await _await_inbox(test_user.id, 1)
        llm.release_first_call.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=15)

    download.assert_awaited_once_with("file-photo-2")
    stored = await _stored_messages(test_user.id)
    caption_row = stored[1]
    assert caption_row.body == CAPTION_TEXT
    # The media pipeline's context (with the staged photo's handle) was
    # persisted on the folded row and is what the LLM saw.
    assert caption_row.processed_context
    assert caption_row.processed_context != CAPTION_TEXT
    assert any(caption_row.processed_context in t for t in _user_texts(llm.calls[-1]))
    assert [r.content for r in _drain_replies()] == [COMBINED_REPLY]


async def test_no_folding_when_idle(test_user: User) -> None:
    """Messages that arrive after the previous turn finished get their own turn."""
    replies_iter = iter([make_text_response("first reply"), make_text_response("second reply")])
    calls: list[list[dict[str, Any]]] = []

    async def fake_llm(**kwargs: Any) -> MessageResponse:
        calls.append(copy.deepcopy(kwargs["messages"]))
        return next(replies_iter)

    with patch("backend.app.agent.core.amessages", side_effect=fake_llm):
        for text in (PHOTO_TEXT, CAPTION_TEXT):
            await process_inbound_from_bus(
                InboundMessage(
                    channel="telegram", sender_id=test_user.channel_identifier, text=text
                )
            )

    assert [r.content for r in _drain_replies()] == ["first reply", "second reply"]
    assert len(calls) == 2
    assert not any(CAPTION_TEXT in t for t in _user_texts(calls[0]))
    stored = await _stored_messages(test_user.id)
    assert [m.direction for m in stored] == ["inbound", "outbound", "inbound", "outbound"]


# ---------------------------------------------------------------------------
# Approval replies and dispatch plumbing
# ---------------------------------------------------------------------------


async def test_approval_reply_during_turn_goes_to_gate_not_the_turn(test_user: User) -> None:
    """While a turn is blocked on approval, "yes" resolves the gate. It is not
    persisted, not queued for folding, and starts no turn."""
    gate = get_approval_gate()
    lock = user_locks.acquire(test_user.id)
    await lock.acquire()
    try:
        approval = asyncio.create_task(
            gate.request_approval(
                user_id=test_user.id,
                tool_name="test_tool",
                description="test",
                publish_outbound=AsyncMock(),
                channel="telegram",
                chat_id=test_user.channel_identifier,
                timeout=5.0,
            )
        )
        deadline = asyncio.get_running_loop().time() + 2
        while not gate.has_pending(test_user.id):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.005)

        with patch("backend.app.agent.ingestion.handle_inbound_message") as mock_handle:
            await asyncio.wait_for(
                process_inbound_from_bus(
                    InboundMessage(
                        channel="telegram",
                        sender_id=test_user.channel_identifier,
                        text="Yes",
                    )
                ),
                timeout=5,
            )
        assert await approval == ApprovalDecision.APPROVED
        mock_handle.assert_not_called()
        assert midturn_inbox.pending_count(test_user.id) == 0
        assert await _stored_messages(test_user.id) == []
    finally:
        lock.release()


async def test_claimed_dispatch_exits_without_running_a_turn(test_user: User) -> None:
    """A waiting dispatch returns as soon as the running turn claims it,
    without a pipeline run or an error fallback, while the lock is still held."""
    lock = user_locks.acquire(test_user.id)
    await lock.acquire()
    session = SessionState(session_id="sess-fold", user_id=test_user.id)
    message = StoredMessage(direction="inbound", body=CAPTION_TEXT, seq=2)
    try:
        with (
            patch("backend.app.agent.ingestion.handle_inbound_message") as mock_handle,
            patch("backend.app.agent.ingestion._send_error_fallback") as mock_fallback,
        ):
            task = asyncio.create_task(
                _dispatch_to_pipeline(
                    user=test_user,
                    session=session,
                    message=message,
                    media_urls=[],
                    channel="telegram",
                )
            )
            await _await_inbox(test_user.id, 1)
            claimed = midturn_inbox.claim(test_user.id, "sess-fold", "telegram")
            assert [e.messages for e in claimed] == [[message]]
            await asyncio.wait_for(task, timeout=2)
        mock_handle.assert_not_called()
        mock_fallback.assert_not_called()
        assert lock.locked()
    finally:
        lock.release()
    assert midturn_inbox.pending_count(test_user.id) == 0


async def test_unclaimed_dispatch_runs_its_own_turn_after_lock(test_user: User) -> None:
    """If the running turn never drains (e.g. it ended first), the waiting
    dispatch unregisters itself and runs normally once the lock frees."""
    lock = user_locks.acquire(test_user.id)
    await lock.acquire()
    session = SessionState(session_id="sess-fold", user_id=test_user.id)
    message = StoredMessage(direction="inbound", body=CAPTION_TEXT, seq=2)
    with patch("backend.app.agent.ingestion.handle_inbound_message") as mock_handle:
        task = asyncio.create_task(
            _dispatch_to_pipeline(
                user=test_user,
                session=session,
                message=message,
                media_urls=[],
                channel="telegram",
            )
        )
        await _await_inbox(test_user.id, 1)
        lock.release()
        await asyncio.wait_for(task, timeout=2)
    mock_handle.assert_awaited_once()
    assert midturn_inbox.pending_count(test_user.id) == 0


async def test_webchat_dispatch_is_never_foldable(test_user: User) -> None:
    """A webchat request owns an SSE future expecting its own reply."""
    lock = user_locks.acquire(test_user.id)
    await lock.acquire()
    session = SessionState(session_id="sess-fold", user_id=test_user.id)
    message = StoredMessage(direction="inbound", body=CAPTION_TEXT, seq=2)
    with patch("backend.app.agent.ingestion.handle_inbound_message") as mock_handle:
        task = asyncio.create_task(
            _dispatch_to_pipeline(
                user=test_user,
                session=session,
                message=message,
                media_urls=[],
                channel="webchat",
                request_id="req-1",
            )
        )
        await asyncio.sleep(0.05)
        assert midturn_inbox.pending_count(test_user.id) == 0
        lock.release()
        await asyncio.wait_for(task, timeout=2)
    mock_handle.assert_awaited_once()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def test_agent_without_drain_hook_is_unchanged(test_user: User) -> None:
    """Heartbeats and webchat build the agent without a drain hook."""
    with patch(
        "backend.app.agent.core.amessages",
        new_callable=AsyncMock,
        return_value=make_text_response("only reply"),
    ) as mock_llm:
        response = await ClawboltAgent(user=test_user).process_message("hello")
    assert response.reply_text == "only reply"
    assert mock_llm.await_count == 1


async def test_agent_sends_draft_when_last_round_has_no_room(test_user: User) -> None:
    """On the final round the draft goes out and the new message stays
    queued for its own turn rather than being consumed unanswered."""
    # Nothing pending before the only LLM call; a message would be pending
    # at the final reply if the loop asked.
    drain = AsyncMock(side_effect=[[], [UserMessage(content=CAPTION_TEXT)]])
    with (
        patch("backend.app.agent.core.MAX_TOOL_ROUNDS", 1),
        patch(
            "backend.app.agent.core.amessages",
            new_callable=AsyncMock,
            return_value=make_text_response(STALE_REPLY),
        ),
    ):
        response = await ClawboltAgent(user=test_user, drain_inbound=drain).process_message(
            PHOTO_TEXT
        )
    assert response.reply_text == STALE_REPLY
    assert drain.await_count == 1


async def test_agent_survives_drain_failure(test_user: User) -> None:
    drain = AsyncMock(side_effect=RuntimeError("boom"))
    with patch(
        "backend.app.agent.core.amessages",
        new_callable=AsyncMock,
        return_value=make_text_response("still replies"),
    ):
        response = await ClawboltAgent(user=test_user, drain_inbound=drain).process_message(
            PHOTO_TEXT
        )
    assert response.reply_text == "still replies"
    assert not response.is_error_fallback
