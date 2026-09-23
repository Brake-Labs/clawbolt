"""A trim must not rewrite the cached history prefix twice.

The turn that trims builds its history in memory. The next turn reloads it
from ``messages`` above the advanced ``sessions.last_trim_seq`` watermark. If
the two differ anywhere before the new tail, the history cache written on the
trim turn is never read, and the next turn pays a second full cache write of
the whole history. Two things used to make them differ: the trim turn put a
``[Summary of earlier conversation: ...]`` message at the head of history
that the reload never had, and the reload stamped a fresh timestamp marker on
whichever message happened to be first after the watermark.

The live loop renders history through ``prompt_epoch.EpochHistoryRenderer``
whenever the stable prefix is on (the default), so the same property is
checked through that renderer too.
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
from backend.app.agent.memory_db import write_memory
from backend.app.agent.messages import UserMessage, messages_to_messages_api
from backend.app.agent.prompt_epoch import EpochHistoryRenderer
from backend.app.config import settings
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


async def _assemble(
    user: User, message: str, now: datetime.datetime, *, epoch_renderer: bool = False
) -> Any:
    """Load history and assemble one turn's prompt.

    With *epoch_renderer* the history goes through the live loop's renderer
    under the stable prefix, and the system block is the real one, built from
    the epoch's workspace snapshot, rather than a fixed override.
    """
    session, _ = await get_or_create_conversation(user.id)
    renderer = EpochHistoryRenderer(user.id, compact=False) if epoch_renderer else None
    history = await load_conversation_history(session, tz_name=_TZ, render=renderer)
    agent = ClawboltAgent(user=user)
    with patch("backend.app.agent.core.settings.context_trim_target_turns", _TARGET_TURNS):
        return await agent.assemble_prompt(
            message,
            history,
            system_prompt_override=None if epoch_renderer else _SYSTEM,
            deterministic_trim=True,
            now=now,
            epoch=renderer.epoch if renderer is not None else None,
        )


async def _trim_then_next_turn(
    user: User, *, epoch_renderer: bool = False, memory_edit: str = ""
) -> tuple[Any, Any]:
    """Assemble a turn that trims, advance the watermark, assemble the next.

    The trim turn is warm (nine minutes after the last reply) and the trim
    drops turn 12, which opened the epoch. *memory_edit* is written to
    MEMORY.md between the two turns, as the trim's compaction does.
    """
    user.timezone = _TZ
    await get_or_create_conversation(user.id)
    rows: list[dict[str, Any]] = []
    for turn in range(_TURNS):
        rows.extend(_turn_rows(turn))
    trim_turn_at = _BASE + datetime.timedelta(minutes=10 * _TURNS, hours=2)
    rows.append({"direction": "inbound", "body": "Trim turn", "timestamp": trim_turn_at})
    await _insert_rows(user.id, rows)

    trim_turn = await _assemble(user, "Trim turn", trim_turn_at, epoch_renderer=epoch_renderer)
    assert trim_turn.dropped, "the fixture must make this turn trim"

    # The trim turn advances the watermark; the LLM compaction is not under test.
    with patch.object(context_module, "compact_session", AsyncMock(return_value=(False, None))):
        await trigger_compaction_for_dropped(user.id, trim_turn.dropped)
        await asyncio.gather(*context_module._background_tasks)
    if memory_edit:
        await write_memory(user.id, memory_edit)

    next_turn_at = trim_turn_at + datetime.timedelta(minutes=2)
    await _insert_rows(
        user.id,
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
    next_turn = await _assemble(user, "Next turn", next_turn_at, epoch_renderer=epoch_renderer)
    assert not next_turn.dropped, "the next turn must not trim again"
    return trim_turn, next_turn


def _assert_history_prefix_kept(trim_turn: Any, next_turn: Any) -> None:
    trim_system, trim_dicts = messages_to_messages_api(trim_turn.messages)
    next_system, next_dicts = messages_to_messages_api(next_turn.messages)
    assert trim_system == next_system

    # Everything the trim turn sent before its own current turn is the
    # history prefix it cache-wrote; the next turn must send it unchanged.
    trim_history = trim_dicts[:-1]
    assert trim_history, "the trim turn kept some history"
    assert next_dicts[: len(trim_history)] == trim_history


async def test_trim_turn_history_matches_next_turn_reload(test_user: User) -> None:
    trim_turn, next_turn = await _trim_then_next_turn(test_user)
    _assert_history_prefix_kept(trim_turn, next_turn)


async def test_trim_reload_matches_through_the_epoch_renderer(test_user: User) -> None:
    """The same property with the stable prefix on, as the live loop runs.

    The renderer must seed the first row after the watermark from the rows
    below it, as the plain loader does; otherwise that row gains a time
    marker on the next turn and the whole history is written to cache twice.
    """
    with patch.object(settings, "prompt_stable_prefix_enabled", True):
        trim_turn, next_turn = await _trim_then_next_turn(test_user, epoch_renderer=True)
    _assert_history_prefix_kept(trim_turn, next_turn)
    # The summary and the stable-prefix sections each appear once, the
    # summary on the trim turn's uncached current turn only.
    trim_current = trim_turn.messages[-1]
    assert isinstance(trim_current, UserMessage)
    assert trim_current.content.startswith("[Summary of earlier conversation:")
    assert trim_current.content.count("[Summary of earlier conversation:") == 1
    assert "## Your Memory" in trim_turn.stable_system
    assert "## Your Memory" not in trim_current.content
    assert all(
        "[Summary of earlier conversation:" not in (m.content or "") for m in next_turn.messages
    )


async def test_warm_trim_keeps_the_system_block(test_user: User) -> None:
    """A warm trim past the epoch's first row keeps the workspace snapshot.

    The next turn keys its epoch on the first row left above the watermark.
    Retaking the snapshot there put the compaction's MEMORY.md edit into the
    system block, which rewrote the system block and the whole history after
    it on the turn after the trim.
    """
    await write_memory(test_user.id, "- Customer prefers mornings")
    with patch.object(settings, "prompt_stable_prefix_enabled", True):
        trim_turn, next_turn = await _trim_then_next_turn(
            test_user,
            epoch_renderer=True,
            memory_edit="- Customer prefers mornings\n- Roof job on Elm St",
        )
    _assert_history_prefix_kept(trim_turn, next_turn)
    assert next_turn.stable_system == trim_turn.stable_system
    assert "Roof job on Elm St" in next_turn.messages[-1].content


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
