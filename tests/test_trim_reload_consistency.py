"""A trim must not rewrite the cached history prefix twice.

The turn that trims builds its history in memory. The next turn reloads it
from ``messages`` above the advanced ``sessions.last_trim_seq`` watermark. If
the two differ anywhere before the new tail, the history cache written on the
trim turn is never read, and the next turn pays a second full cache write of
the whole history. Two things used to make them differ: the trim turn put a
``[Summary of earlier conversation: ...]`` message at the head of history
that the reload never had, and the reload stamped a fresh timestamp marker on
whichever message happened to be first after the watermark.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from typing import Any
from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from backend.app.agent import context as context_module
from backend.app.agent.context import (
    get_or_create_conversation,
    load_conversation_history,
    trigger_compaction_for_dropped,
)
from backend.app.agent.core import ClawboltAgent
from backend.app.agent.messages import UserMessage, messages_to_messages_api
from backend.app.database import db_session_async
from backend.app.models import ChatSession, Message, User

_TZ = "America/New_York"
_BASE = datetime.datetime(2026, 6, 1, 13, 0, tzinfo=datetime.UTC)
_SYSTEM = "Stable system prompt"
_TURNS = 30
_TARGET_TURNS = 4


def _tool_json(turn: int) -> str:
    return json.dumps(
        [
            {
                "tool_call_id": f"call_{turn}",
                "name": "calculate",
                "args": {"expression": f"{turn} * 2"},
                "result": str(turn * 2),
                "is_error": False,
            }
        ]
    )


async def _insert_rows(user_id: str, rows: list[dict[str, Any]]) -> None:
    async with db_session_async() as db:
        cs = (await db.execute(select(ChatSession).filter_by(user_id=user_id))).scalar_one()
        max_seq = (
            await db.execute(
                select(Message.seq)
                .where(Message.session_id == cs.id)
                .order_by(Message.seq.desc())
                .limit(1)
            )
        ).scalar() or 0
        for offset, row in enumerate(rows, start=1):
            db.add(Message(session_id=cs.id, seq=max_seq + offset, **row))
        await db.commit()


def _turn_rows(turn: int) -> list[dict[str, Any]]:
    """One user turn ten minutes after the last, with a reply a minute later.

    Every third reply used a tool, so history carries tool_use/tool_result
    blocks and the trimmer's atomic tool blocks are exercised. A two-hour
    gap before turn 12 puts a genuine gap marker mid-history.
    """
    start = _BASE + datetime.timedelta(minutes=10 * turn)
    if turn >= 12:
        start += datetime.timedelta(hours=2)
    return [
        {
            "direction": "inbound",
            "body": f"Question {turn}",
            "timestamp": start,
        },
        {
            "direction": "outbound",
            "body": f"Answer {turn}",
            "llm_reply_text": f"Answer {turn}",
            "tool_interactions_json": _tool_json(turn) if turn % 3 == 0 else "",
            "timestamp": start + datetime.timedelta(minutes=1),
        },
    ]


async def _assemble(user: User, message: str, now: datetime.datetime) -> Any:
    session, _ = await get_or_create_conversation(user.id)
    history = await load_conversation_history(session, tz_name=_TZ)
    agent = ClawboltAgent(user=user)
    with patch("backend.app.agent.core.settings.context_trim_target_turns", _TARGET_TURNS):
        return await agent.assemble_prompt(
            message,
            history,
            system_prompt_override=_SYSTEM,
            deterministic_trim=True,
            now=now,
        )


async def test_trim_turn_history_matches_next_turn_reload(test_user: User) -> None:
    test_user.timezone = _TZ
    await get_or_create_conversation(test_user.id)
    rows: list[dict[str, Any]] = []
    for turn in range(_TURNS):
        rows.extend(_turn_rows(turn))
    trim_turn_at = _BASE + datetime.timedelta(minutes=10 * _TURNS, hours=2)
    rows.append({"direction": "inbound", "body": "Trim turn", "timestamp": trim_turn_at})
    await _insert_rows(test_user.id, rows)

    trim_turn = await _assemble(test_user, "Trim turn", trim_turn_at)
    assert trim_turn.dropped, "the fixture must make this turn trim"

    # The trim turn advances the watermark; the LLM compaction is not under test.
    with patch.object(context_module, "compact_session", AsyncMock(return_value=(False, None))):
        await trigger_compaction_for_dropped(test_user.id, trim_turn.dropped)
        await asyncio.gather(*context_module._background_tasks)

    next_turn_at = trim_turn_at + datetime.timedelta(minutes=2)
    await _insert_rows(
        test_user.id,
        [
            {
                "direction": "outbound",
                "body": "Trim reply",
                "llm_reply_text": "Trim reply",
                "timestamp": trim_turn_at + datetime.timedelta(minutes=1),
            },
            {"direction": "inbound", "body": "Next turn", "timestamp": next_turn_at},
        ],
    )
    next_turn = await _assemble(test_user, "Next turn", next_turn_at)
    assert not next_turn.dropped, "the next turn must not trim again"

    trim_system, trim_dicts = messages_to_messages_api(trim_turn.messages)
    next_system, next_dicts = messages_to_messages_api(next_turn.messages)
    assert trim_system == next_system

    # Everything the trim turn sent before its own current turn is the
    # history prefix it cache-wrote; the next turn must send it unchanged.
    trim_history = trim_dicts[:-1]
    assert trim_history, "the trim turn kept some history"
    assert next_dicts[: len(trim_history)] == trim_history


async def test_trim_summary_rides_the_current_turn(test_user: User) -> None:
    """The dropped-history summary still reaches the model on the trim turn.

    It moved off the head of history (where it made the prefix differ from
    the reload) onto the current turn, which is uncached anyway.
    """
    await get_or_create_conversation(test_user.id)
    rows: list[dict[str, Any]] = []
    for turn in range(_TURNS):
        rows.extend(_turn_rows(turn))
    at = _BASE + datetime.timedelta(minutes=10 * _TURNS, hours=2)
    rows.append({"direction": "inbound", "body": "Trim turn", "timestamp": at})
    await _insert_rows(test_user.id, rows)

    assembled = await _assemble(test_user, "Trim turn", at)
    assert assembled.dropped
    current = assembled.messages[-1]
    assert isinstance(current, UserMessage)
    assert "[Summary of earlier conversation:" in current.content
    assert current.content.rstrip().endswith("Trim turn")
    history = assembled.messages[1:-1]
    assert not any(
        isinstance(m, UserMessage) and "[Summary of earlier conversation:" in m.content
        for m in history
    )
