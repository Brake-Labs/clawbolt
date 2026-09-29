"""Prompt-cache epochs: a stable cached prefix, and history rebuilt at cold starts.

The properties that matter, each of which saves money only if it holds:

- Inside an epoch the system block is byte-identical on every call, and the
  history only grows by appending, so every call reads the cached prefix.
- A workspace edit made mid-epoch still reaches the model on the next turn.
- A cold start rebuilds the history to a budget with every ``tool_use`` still
  paired with a result, deterministically, so the next turn renders the same
  bytes.
- With both settings off, nothing changes.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm import ContextLengthExceededError
from pydantic import BaseModel
from sqlalchemy import select

from backend.app.agent import context as context_module
from backend.app.agent.context import (
    _stored_messages_to_agent_messages,
    load_conversation_history,
)
from backend.app.agent.core import AssembledPrompt, ClawboltAgent
from backend.app.agent.dto import StoredMessage
from backend.app.agent.memory_db import write_memory
from backend.app.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
    messages_to_messages_api,
)
from backend.app.agent.prompt_epoch import (
    EpochHistoryRenderer,
    PromptEpoch,
    build_history_view,
    cache_idle_threshold,
    elided_result_stub,
    find_epoch,
    is_cold_gap,
    shed_history_view,
)
from backend.app.agent.router import PipelineContext, load_history_step, run_agent_step
from backend.app.agent.session_db import get_session_store
from backend.app.agent.skills import loader
from backend.app.agent.skills.loader import (
    extract_delivered_skills,
    skill_delivery_marker,
    skill_guidance_block,
)
from backend.app.agent.system_prompt import WorkspaceSnapshot, render_workspace_updates
from backend.app.agent.tools.base import Tool, ToolResult, ToolTags
from backend.app.agent.tools.registry import SubToolInfo, ToolRegistry
from backend.app.config import settings
from backend.app.database import db_session_async
from backend.app.models import ChatSession, Message, User
from backend.app.services.model_comparison.sampling import ReplayFixture, assemble_for_sample
from backend.app.services.model_comparison.types import HistoryMode, ReplaySample
from tests.conftest import create_test_session
from tests.mocks.llm import extract_system_text, make_text_response, make_tool_call_response

T0 = _dt.datetime(2026, 9, 1, 8, 0, tzinfo=_dt.UTC)
BIG = "x" * 4_000  # a tool result worth eliding: ~1k tokens


def _at(minutes: float) -> str:
    return (T0 + _dt.timedelta(minutes=minutes)).isoformat()


class _Rows:
    """Builds a transcript: user turns, each answered with optional tool calls."""

    def __init__(self) -> None:
        self.rows: list[StoredMessage] = []

    def turn(self, minute: float, ask: str, reply: str, results: list[str] | None = None) -> None:
        seq = len(self.rows) + 1
        self.rows.append(
            StoredMessage(
                direction="inbound", body=ask, processed_context=ask, timestamp=_at(minute), seq=seq
            )
        )
        tools = [
            {
                "tool_call_id": f"call_{seq}_{i}",
                "name": "web_search",
                "args": {"query": f"{ask} {i}"},
                "result": result,
            }
            for i, result in enumerate(results or [])
        ]
        self.rows.append(
            StoredMessage(
                direction="outbound",
                body=reply,
                llm_reply_text=reply,
                tool_interactions_json=json.dumps(tools) if tools else "",
                timestamp=_at(minute + 1),
                seq=seq + 1,
            )
        )

    def ask(self, minute: float, text: str) -> StoredMessage:
        """The current inbound row, not yet answered."""
        row = StoredMessage(
            direction="inbound",
            body=text,
            processed_context=text,
            timestamp=_at(minute),
            seq=len(self.rows) + 1,
        )
        self.rows.append(row)
        return row


class _Query(BaseModel):
    query: str = ""


class _Action(BaseModel):
    action: str = ""


async def _ok(**_: object) -> ToolResult:
    return ToolResult(content="ok")


# The schema the rebuild classifies calls against: a lookup, a write, and a
# multi-action tool that only reads for ``status`` (the shape of
# ``manage_integration``, the repo's ``read_only_when`` tool).
TOOLS: dict[str, Tool] = {
    "web_search": Tool(
        name="web_search",
        description="Search",
        function=_ok,
        params_model=_Query,
        tags={ToolTags.READ_ONLY},
    ),
    "qb_create": Tool(name="qb_create", description="Create", function=_ok, params_model=_Query),
    "manage_integration": Tool(
        name="manage_integration",
        description="Manage",
        function=_ok,
        params_model=_Action,
        read_only_when=lambda args: args.get("action") == "status",
    ),
}


def _calls_turn(
    t: _Rows, minute: float, ask: str, calls: list[tuple[str, dict[str, Any], str, bool]]
) -> None:
    """A turn answered with *calls*: (tool name, args, result, is_error)."""
    seq = len(t.rows) + 1
    t.rows.append(
        StoredMessage(
            direction="inbound", body=ask, processed_context=ask, timestamp=_at(minute), seq=seq
        )
    )
    t.rows.append(
        StoredMessage(
            direction="outbound",
            body=f"done: {ask}",
            llm_reply_text=f"done: {ask}",
            tool_interactions_json=json.dumps(
                [
                    {
                        "tool_call_id": f"call_{seq}_{i}",
                        "name": name,
                        "args": args,
                        "result": result,
                        "is_error": is_error,
                    }
                    for i, (name, args, result, is_error) in enumerate(calls)
                ]
            ),
            timestamp=_at(minute + 1),
            seq=seq + 1,
        )
    )


def _results(messages: list[AgentMessage]) -> dict[str, ToolResultMessage]:
    return {m.tool_call_id: m for m in messages if isinstance(m, ToolResultMessage)}


def _api(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    return messages_to_messages_api(messages)[1]


def _assert_pairs_intact(messages: list[AgentMessage]) -> None:
    """Every tool_use has a result right after it, and every result a tool_use."""
    wanted: list[str] = []
    for m in messages:
        if isinstance(m, AssistantMessage) and m.tool_calls:
            assert not wanted, "a tool_use was left without its result"
            wanted = [tc.id for tc in m.tool_calls]
        elif isinstance(m, ToolResultMessage):
            assert m.tool_call_id in wanted, "a result with no tool_use before it"
            wanted.remove(m.tool_call_id)
        else:
            assert not wanted, "a tool_use was left without its result"
    assert not wanted


@pytest.fixture()
def one_hour_cache() -> Iterator[None]:
    with (
        patch.object(settings, "prompt_cache_idle_seconds", None),
        patch.object(settings, "llm_cache_extended_ttl", True),
        patch.object(settings, "llm_cache_history_ttl", "1h"),
    ):
        yield


# -- the definition of a cold start ------------------------------------------


def test_cold_gap_follows_the_history_cache_ttl() -> None:
    with (
        patch.object(settings, "prompt_cache_idle_seconds", None),
        patch.object(settings, "llm_cache_extended_ttl", True),
    ):
        with patch.object(settings, "llm_cache_history_ttl", "1h"):
            assert cache_idle_threshold() == _dt.timedelta(hours=1)
        with patch.object(settings, "llm_cache_history_ttl", "5m"):
            assert cache_idle_threshold() == _dt.timedelta(minutes=5)
    # A 1h history behind a 5m system block is sent as 5m, so it goes cold at 5m.
    with (
        patch.object(settings, "prompt_cache_idle_seconds", None),
        patch.object(settings, "llm_cache_extended_ttl", False),
        patch.object(settings, "llm_cache_history_ttl", "1h"),
    ):
        assert cache_idle_threshold() == _dt.timedelta(minutes=5)
    with patch.object(settings, "prompt_cache_idle_seconds", 90):
        assert cache_idle_threshold() == _dt.timedelta(seconds=90)
        assert is_cold_gap(T0, T0 + _dt.timedelta(seconds=91))
        assert not is_cold_gap(T0, T0 + _dt.timedelta(seconds=90))
    assert is_cold_gap(None, T0)


def test_epoch_opens_at_the_first_message_after_an_idle_gap(one_hour_cache: None) -> None:
    t = _Rows()
    t.turn(0, "morning ask", "morning reply")
    t.turn(10, "follow-up", "done")
    t.turn(200, "afternoon ask", "afternoon reply")  # opens an epoch
    t.turn(220, "another", "ok")
    current = t.ask(240, "and this")
    epoch, boundary = find_epoch(t.rows[:-1], current)
    assert epoch == PromptEpoch(key=5, cold_start=False)
    assert boundary == 4

    cold = t.ask(400, "evening")
    epoch, boundary = find_epoch(t.rows[:-1], cold)
    assert epoch == PromptEpoch(key=cold.seq, cold_start=True)
    assert boundary == len(t.rows) - 1


def test_a_heartbeat_reply_does_not_count_as_a_warm_cache(one_hour_cache: None) -> None:
    """A heartbeat's call carries no history, so it leaves the history cold."""
    t = _Rows()
    t.turn(0, "ask", "reply")
    t.rows.append(
        StoredMessage(direction="outbound", body="Reminder: invoice due", timestamp=_at(170), seq=3)
    )
    current = t.ask(180, "thanks")
    epoch, _ = find_epoch(t.rows[:-1], current)
    assert epoch.cold_start


def test_a_batch_opens_one_epoch_at_its_first_row(one_hour_cache: None) -> None:
    t = _Rows()
    t.turn(0, "ask", "reply")
    t.ask(120, "photo")
    current = t.ask(120.1, "caption")
    epoch, boundary = find_epoch(t.rows[:-1], current)
    assert epoch == PromptEpoch(key=3, cold_start=True)
    assert boundary == 2


# -- task B: the history rebuilt at cold starts -------------------------------


def _long_history(turns: int = 10) -> _Rows:
    t = _Rows()
    for i in range(turns):
        t.turn(i * 5, f"ask {i}", f"reply {i}", results=[BIG, "short result"])
    return t


def test_cold_start_elides_old_results_and_keeps_pairs(one_hour_cache: None) -> None:
    t = _long_history(10)
    current = t.ask(500, "back after lunch")
    with (
        patch.object(settings, "cold_start_verbatim_turns", 3),
        patch.object(settings, "cold_start_history_budget_tokens", 30_000),
    ):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)

    assert view.epoch.cold_start
    assert not view.dropped_rows
    _assert_pairs_intact(view.messages)
    results = [m for m in view.messages if isinstance(m, ToolResultMessage)]
    stubbed = [r for r in results if r.content.startswith("[tool result elided")]
    # 7 old turns x one big result; the short ones and the last 3 turns stay.
    assert len(stubbed) == 7 == view.elided_results
    assert stubbed[0].content == elided_result_stub("web_search", len(BIG))
    assert sum(r.content == BIG for r in results) == 3
    assert sum(r.content == "short result" for r in results) == 10
    # Every word of prose survives.
    prose = " ".join(m.content or "" for m in view.messages if not isinstance(m, ToolResultMessage))
    for i in range(10):
        assert f"ask {i}" in prose and f"reply {i}" in prose


def test_cold_start_shrinks_the_verbatim_window_before_dropping_prose(
    one_hour_cache: None,
) -> None:
    t = _long_history(10)
    current = t.ask(500, "back")
    with (
        patch.object(settings, "cold_start_verbatim_turns", 4),
        patch.object(settings, "cold_start_history_budget_tokens", 3_000),
    ):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    assert not view.dropped_rows
    assert sum(isinstance(m, ToolResultMessage) and m.content == BIG for m in view.messages) == 1


def test_cold_start_drops_the_oldest_turns_when_prose_is_over_budget(
    one_hour_cache: None,
) -> None:
    t = _Rows()
    for i in range(40):
        t.turn(i * 5, f"ask {i} " + "p" * 400, f"reply {i} " + "q" * 400)
    current = t.ask(900, "back")
    with patch.object(settings, "cold_start_history_budget_tokens", 4_000):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
        tokens = sum(len(m.content or "") for m in view.messages) // 2
        assert tokens <= 4_000
        assert view.dropped_rows
        assert view.dropped_rows[0].seq == 1
        # A fixed point: the rows that remain rebuild to the same bytes and
        # drop nothing more, so the rest of the epoch appends to this. The
        # dropped rows are below the watermark then, and the loader passes
        # them as ``preceding``.
        kept = [r for r in t.rows[:-1] if r.seq > view.dropped_rows[-1].seq]
        again = build_history_view(
            kept, current, "", compact=True, tools_by_name=TOOLS, preceding=view.dropped_rows
        )
    assert not again.dropped_rows
    assert _api(again.messages) == _api(view.messages)


def test_rebuild_is_deterministic_and_a_warm_turn_only_appends(one_hour_cache: None) -> None:
    t = _long_history(8)
    cold = t.ask(500, "back after lunch")
    with patch.object(settings, "cold_start_history_budget_tokens", 30_000):
        first = build_history_view(t.rows[:-1], cold, "", compact=True, tools_by_name=TOOLS)
        assert _api(
            build_history_view(t.rows[:-1], cold, "", compact=True, tools_by_name=TOOLS).messages
        ) == _api(first.messages)

        # The cold turn is answered with a large lookup, then the user writes
        # again inside the epoch.
        t.rows.append(
            StoredMessage(
                direction="outbound",
                body="found it",
                llm_reply_text="found it",
                tool_interactions_json=json.dumps(
                    [{"tool_call_id": "c", "name": "web_search", "args": {}, "result": BIG}]
                ),
                timestamp=_at(502),
                seq=cold.seq + 1,
            )
        )
        warm = t.ask(510, "and the other one?")
        second = build_history_view(t.rows[:-1], warm, "", compact=True, tools_by_name=TOOLS)

    assert second.epoch == PromptEpoch(key=cold.seq, cold_start=False)
    before, after = _api(first.messages), _api(second.messages)
    assert after[: len(before)] == before
    # What was appended is verbatim: the new turn's result is not elided.
    appended = second.messages[len(first.messages) :]
    assert any(isinstance(m, ToolResultMessage) and m.content == BIG for m in appended)


# -- write results survive the rebuild ----------------------------------------

# A write's result names the record it made. Long enough that its length
# alone would get it stubbed.
WRITE_RESULT = "Created invoice 643 for Acme Plumbing. " + "line item detail " * 40


def _mixed_history(turns: int = 8) -> _Rows:
    """Turns that each read and write, for a cold start to rebuild."""
    t = _Rows()
    for i in range(turns):
        _calls_turn(
            t,
            i * 5,
            f"ask {i}",
            [
                ("web_search", {"query": f"q{i}"}, BIG, False),
                ("qb_create", {"query": f"invoice {i}"}, WRITE_RESULT, False),
            ],
        )
    return t


def test_cold_start_keeps_old_write_results_and_stubs_old_reads(one_hour_cache: None) -> None:
    t = _mixed_history(8)
    current = t.ask(500, "send that invoice")
    with (
        patch.object(settings, "cold_start_verbatim_turns", 2),
        patch.object(settings, "cold_start_history_budget_tokens", 30_000),
    ):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)

    assert not view.dropped_rows
    _assert_pairs_intact(view.messages)
    results = _results(view.messages)
    # The oldest turn: the read is a stub, the write is word for word.
    assert results["call_1_0"].content == elided_result_stub("web_search", len(BIG))
    assert results["call_1_1"].content == WRITE_RESULT
    assert sum(r.content == WRITE_RESULT for r in results.values()) == 8
    assert view.elided_results == 6


def test_a_call_classified_per_call_follows_its_arguments(one_hour_cache: None) -> None:
    t = _Rows()
    _calls_turn(
        t,
        0,
        "check and disconnect",
        [
            ("manage_integration", {"action": "status"}, BIG, False),
            ("manage_integration", {"action": "disable"}, BIG, False),
        ],
    )
    for i in range(4):
        t.turn(5 + i * 5, f"ask {i}", f"reply {i}")
    current = t.ask(500, "back")
    with patch.object(settings, "cold_start_verbatim_turns", 1):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    results = _results(view.messages)
    assert results["call_1_0"].content.startswith("[tool result elided")
    assert results["call_1_1"].content == BIG


def test_a_tool_missing_from_the_schema_is_kept_verbatim(one_hour_cache: None) -> None:
    """A tool the schema no longer offers cannot be classified: it counts as a write."""
    t = _Rows()
    _calls_turn(t, 0, "old", [("retired_tool", {"query": "x"}, BIG, False)])
    for i in range(4):
        t.turn(5 + i * 5, f"ask {i}", f"reply {i}", results=[BIG])
    current = t.ask(500, "back")
    with patch.object(settings, "cold_start_verbatim_turns", 1):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    results = _results(view.messages)
    assert results["call_1_0"].content == BIG
    # A lookup of the same age is still stubbed.
    assert results["call_3_0"].content.startswith("[tool result elided")
    _assert_pairs_intact(view.messages)


def test_a_write_error_is_kept_verbatim(one_hour_cache: None) -> None:
    error = "QuickBooks rejected the invoice: customer 88 is inactive. " + "detail " * 60
    t = _Rows()
    _calls_turn(
        t,
        0,
        "invoice it",
        [
            ("qb_create", {"query": "invoice"}, error, True),
            ("web_search", {"query": "why"}, error, True),
        ],
    )
    for i in range(4):
        t.turn(5 + i * 5, f"ask {i}", f"reply {i}")
    current = t.ask(500, "back")
    with patch.object(settings, "cold_start_verbatim_turns", 1):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    results = _results(view.messages)
    # It explains why the record does not exist, so a retry is not a duplicate.
    assert results["call_1_0"].content == error
    # A failed lookup is still a lookup.
    assert results["call_1_1"].content.startswith("[tool result elided")


def test_writes_alone_over_budget_drop_the_oldest_turns(one_hour_cache: None) -> None:
    """Kept writes count towards the budget. When they alone exceed it the
    oldest turns go to compaction whole, write results included."""
    big_write = "Created estimate 77. " + "w" * 4_000
    t = _Rows()
    for i in range(12):
        _calls_turn(t, i * 5, f"ask {i}", [("qb_create", {"query": f"e{i}"}, big_write, False)])
    current = t.ask(500, "back")
    budget = 6_000
    with patch.object(settings, "cold_start_history_budget_tokens", budget):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
        assert view.dropped_rows
        # The rows handed to compaction carry the write results.
        dropped_results = [
            item["result"]
            for row in view.dropped_rows
            if row.tool_interactions_json
            for item in json.loads(row.tool_interactions_json)
        ]
        assert big_write in dropped_results
        assert sum(len(m.content or "") for m in view.messages) // 2 <= budget
        kept = [m for m in view.messages if isinstance(m, ToolResultMessage)]
        assert kept and all(m.content == big_write for m in kept)
        _assert_pairs_intact(view.messages)
        # A fixed point: the next turn loads the kept rows and drops nothing more.
        rest = [r for r in t.rows[:-1] if r.seq > view.dropped_rows[-1].seq]
        again = build_history_view(
            rest, current, "", compact=True, tools_by_name=TOOLS, preceding=view.dropped_rows
        )
    assert not again.dropped_rows
    assert _api(again.messages) == _api(view.messages)


def test_a_single_write_over_budget_is_kept_whole(one_hour_cache: None) -> None:
    """Correctness over budget: the newest turn stays, write result and all."""
    huge = "Created work order 5501. " + "z" * 40_000
    t = _Rows()
    _calls_turn(t, 0, "make it", [("qb_create", {"query": "wo"}, huge, False)])
    current = t.ask(500, "back")
    with patch.object(settings, "cold_start_history_budget_tokens", 1_000):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    assert not view.dropped_rows
    assert _results(view.messages)["call_1_0"].content == huge


def test_rebuild_with_writes_is_deterministic_and_a_warm_turn_only_appends(
    one_hour_cache: None,
) -> None:
    t = _mixed_history(8)
    cold = t.ask(500, "back after lunch")
    with (
        patch.object(settings, "cold_start_verbatim_turns", 2),
        patch.object(settings, "cold_start_history_budget_tokens", 30_000),
    ):
        first = build_history_view(t.rows[:-1], cold, "", compact=True, tools_by_name=TOOLS)
        repeat = build_history_view(t.rows[:-1], cold, "", compact=True, tools_by_name=TOOLS)
        assert _api(repeat.messages) == _api(first.messages)

        # The cold turn writes, then the user writes again inside the epoch.
        t.rows.append(
            StoredMessage(
                direction="outbound",
                body="sent",
                llm_reply_text="sent",
                tool_interactions_json=json.dumps(
                    [
                        {
                            "tool_call_id": "c",
                            "name": "qb_create",
                            "args": {"query": "send 643"},
                            "result": WRITE_RESULT,
                        }
                    ]
                ),
                timestamp=_at(502),
                seq=cold.seq + 1,
            )
        )
        warm = t.ask(510, "and the other one?")
        second = build_history_view(t.rows[:-1], warm, "", compact=True, tools_by_name=TOOLS)

    assert second.epoch == PromptEpoch(key=cold.seq, cold_start=False)
    before, after = _api(first.messages), _api(second.messages)
    assert after[: len(before)] == before
    _assert_pairs_intact(second.messages)


def test_a_compacting_renderer_needs_the_tools() -> None:
    with pytest.raises(ValueError, match="tools_by_name"):
        EpochHistoryRenderer("u", compact=True)


def test_no_cold_gap_means_nothing_is_rebuilt(one_hour_cache: None) -> None:
    t = _long_history(10)
    current = t.ask(60, "still here")
    view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    assert view.elided_results == 0
    assert _api(view.messages) == _api(_stored_messages_to_agent_messages(t.rows[:-1]))


def test_compaction_off_renders_exactly_the_old_history(one_hour_cache: None) -> None:
    t = _long_history(10)
    current = t.ask(900, "back")
    view = build_history_view(t.rows[:-1], current, "", compact=False, tools_by_name=TOOLS)
    assert view.epoch.cold_start
    assert _api(view.messages) == _api(_stored_messages_to_agent_messages(t.rows[:-1]))


async def test_cold_start_drop_compacts_and_the_next_load_matches(
    test_user: User, one_hour_cache: None
) -> None:
    """Through the real loader: the drop advances the watermark, compaction is
    fired for the dropped rows, and the next turn in the epoch renders the
    rebuilt history byte for byte."""
    t = _Rows()
    for i in range(30):
        t.turn(i * 5, f"ask {i} " + "p" * 400, f"reply {i} " + "q" * 400)
    t.ask(900, "back")
    await create_test_session(test_user.id, messages=t.rows)

    with (
        patch.object(settings, "cold_start_compaction_enabled", True),
        patch.object(settings, "cold_start_history_budget_tokens", 3_000),
        patch("backend.app.agent.context.compact_session", new_callable=AsyncMock) as compact,
    ):
        compact.return_value = ("", False)
        state, _ = await get_session_store(test_user.id).get_or_create_session()
        renderer = EpochHistoryRenderer(test_user.id, compact=True, tools_by_name=TOOLS)
        first = await load_conversation_history(state, render=renderer)
        assert renderer.epoch is not None and renderer.epoch.cold_start

        state, _ = await get_session_store(test_user.id).get_or_create_session()
        assert state.last_trim_seq is not None and state.last_trim_seq > 0
        again = await load_conversation_history(
            state, render=EpochHistoryRenderer(test_user.id, compact=True, tools_by_name=TOOLS)
        )

    assert compact.await_count == 1
    assert _api(again) == _api(first)


async def test_live_turn_classifies_history_by_its_own_tools(
    test_user: User, one_hour_cache: None
) -> None:
    """The pipeline builds the turn's tools once, before the history, and the
    rebuild reads them: the model is sent the old write whole and the old
    lookup as a stub."""
    t = _mixed_history(6)
    t.ask(500, "send that invoice")
    await create_test_session(test_user.id, messages=t.rows)
    state, _ = await get_session_store(test_user.id).get_or_create_session()
    ctx = PipelineContext(user=test_user, session=state, message=state.messages[-1], media_urls=[])
    assemble = AsyncMock(return_value=(list(TOOLS.values()), {}))
    with (
        patch.object(settings, "cold_start_compaction_enabled", True),
        patch.object(settings, "cold_start_verbatim_turns", 1),
        patch("backend.app.agent.router.assemble_turn_tools", assemble),
        patch(
            "backend.app.agent.core.amessages_streamed",
            new_callable=AsyncMock,
            return_value=make_text_response("Sent."),
        ) as llm,
    ):
        ctx = await load_history_step(ctx)
        ctx = await run_agent_step(ctx)

    assert assemble.await_count == 1
    assert ctx.response is not None and ctx.response.reply_text == "Sent."
    sent = json.dumps(llm.call_args.kwargs["messages"])
    assert sent.count(json.dumps(WRITE_RESULT)[1:-1]) == 6
    assert json.dumps(elided_result_stub("web_search", len(BIG)))[1:-1] in sent


# -- task A: the stable prefix ------------------------------------------------


class _NoArgs(BaseModel):
    pass


async def _noop(**_: object) -> ToolResult:
    return ToolResult(content="ok")


def _tools() -> list[Tool]:
    return [
        Tool(
            name="save_fact",
            description="Save",
            function=_noop,
            params_model=_NoArgs,
            usage_hint="Save durable facts to MEMORY.md.",
        )
    ]


async def _turn(
    user: User, mock: AsyncMock, epoch: PromptEpoch | None, history: list[AgentMessage]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Run one turn; return every call's system text and the last call's messages."""
    mock.reset_mock()
    agent = ClawboltAgent(user=user)
    agent.register_tools(_tools())
    await agent.process_message("next", history, prompt_epoch=epoch)
    systems = [extract_system_text(c.kwargs["system"]) for c in mock.call_args_list]
    return systems, mock.call_args_list[-1].kwargs["messages"]


@patch("backend.app.agent.core.amessages_streamed")
async def test_system_block_is_byte_identical_within_an_epoch(
    mock_amessages: AsyncMock, test_user: User
) -> None:
    """Across rounds and turns of one epoch, with a memory edit in between."""
    facts = [f"- Fact number {i} about the customer's job site" for i in range(12)]
    await write_memory(test_user.id, "\n".join(["- Customer prefers mornings", *facts]))
    epoch = PromptEpoch(key=11, cold_start=True)
    mock_amessages.side_effect = [
        make_tool_call_response([{"name": "save_fact", "arguments": {}, "id": "t1"}]),
        make_text_response("Saved."),
    ]
    first_systems, _ = await _turn(test_user, mock_amessages, epoch, [])

    await write_memory(test_user.id, "\n".join(["- Customer prefers afternoons", *facts]))
    mock_amessages.side_effect = [make_text_response("Noted.")]
    warm = PromptEpoch(key=11, cold_start=False)
    second_systems, messages = await _turn(test_user, mock_amessages, warm, [])

    assert len(first_systems) == 2
    assert len({*first_systems, *second_systems}) == 1
    system = first_systems[0]
    assert "## Tool Guidelines\n- Save durable facts to MEMORY.md." in system
    assert "## Your Memory\n- Customer prefers mornings" in system

    # The edit is not lost: it rides the current turn, marked as newer.
    current_turn = messages[-1]["content"]
    assert "## Workspace Updates" in current_turn
    assert "MEMORY.md changed" in current_turn
    assert "\n-- Customer prefers mornings\n+- Customer prefers afternoons\n" in current_turn
    # Only the change rides the turn, not the unchanged facts.
    assert "Fact number 5" not in current_turn


def test_a_change_larger_than_the_file_sends_the_file() -> None:
    old = WorkspaceSnapshot(soul="s", user="u", memory="- one")
    new = WorkspaceSnapshot(soul="s", user="u", memory="- two")
    updates = render_workspace_updates(old, new)
    assert updates == "MEMORY.md changed since it was last shown above. It now reads:\n- two"
    assert render_workspace_updates(old, old) == ""


def test_a_repeated_line_is_placed_by_its_context() -> None:
    """The edit names which customer paid, not just that a "paid" line changed."""
    customers = [f"Test Customer {c}" for c in "ABCDEFGH"]
    old_memory = "\n".join(
        f"## {name}\n- deposit paid: no\n- notes: " + "x" * 60 for name in customers
    )
    new_memory = old_memory.replace(
        "## Test Customer C\n- deposit paid: no", "## Test Customer C\n- deposit paid: yes"
    )
    updates = render_workspace_updates(
        WorkspaceSnapshot(soul="", user="", memory=old_memory),
        WorkspaceSnapshot(soul="", user="", memory=new_memory),
    )
    assert "It now reads" not in updates
    assert " ## Test Customer C\n-- deposit paid: no\n+- deposit paid: yes" in updates


@patch("backend.app.agent.core.amessages_streamed")
async def test_the_next_cold_start_folds_the_edit_into_the_snapshot(
    mock_amessages: AsyncMock, test_user: User
) -> None:
    await write_memory(test_user.id, "- Old fact")
    mock_amessages.return_value = make_text_response("ok")
    await _turn(test_user, mock_amessages, PromptEpoch(key=1, cold_start=True), [])
    await write_memory(test_user.id, "- New fact")

    systems, messages = await _turn(
        test_user, mock_amessages, PromptEpoch(key=9, cold_start=True), []
    )
    assert "## Your Memory\n- New fact" in systems[0]
    assert "Old fact" not in systems[0]
    assert "Workspace Updates" not in messages[-1]["content"]


@patch("backend.app.agent.core.amessages_streamed")
async def test_an_unchanged_workspace_adds_nothing_to_the_current_turn(
    mock_amessages: AsyncMock, test_user: User
) -> None:
    await write_memory(test_user.id, "- A fact")
    mock_amessages.return_value = make_text_response("ok")
    epoch = PromptEpoch(key=3, cold_start=False)
    await _turn(test_user, mock_amessages, epoch, [])
    _, messages = await _turn(test_user, mock_amessages, epoch, [])
    current_turn = messages[-1]["content"]
    assert "A fact" not in current_turn
    assert "Tool Guidelines" not in current_turn
    assert "Workspace Updates" not in current_turn


@patch("backend.app.agent.core.amessages_streamed")
async def test_heartbeat_turns_render_the_live_workspace(
    mock_amessages: AsyncMock, test_user: User
) -> None:
    """A turn with no epoch (a heartbeat, a replay) never reads a snapshot."""
    await write_memory(test_user.id, "- Snapshot fact")
    mock_amessages.return_value = make_text_response("ok")
    await _turn(test_user, mock_amessages, PromptEpoch(key=5, cold_start=True), [])
    await write_memory(test_user.id, "- Live fact")

    agent = ClawboltAgent(user=test_user)
    agent.register_tools(_tools())
    await agent.process_message("Scheduled task", wrap_up_on_max_rounds=False)
    call = mock_amessages.call_args
    system = extract_system_text(call.kwargs["system"])
    assert "## Your Memory\n- Live fact" in system
    assert "Workspace Updates" not in call.kwargs["messages"][-1]["content"]


@patch("backend.app.agent.core.amessages_streamed")
async def test_both_settings_off_keep_the_old_layout(
    mock_amessages: AsyncMock, test_user: User
) -> None:
    await write_memory(test_user.id, "- A fact")
    mock_amessages.return_value = make_text_response("ok")
    with patch.object(settings, "prompt_stable_prefix_enabled", False):
        _, messages = await _turn(test_user, mock_amessages, None, [])
        system = extract_system_text(mock_amessages.call_args.kwargs["system"])
    current_turn = messages[-1]["content"]
    assert "## Your Memory\n- A fact" in current_turn
    assert "## Tool Guidelines" in current_turn
    assert "A fact" not in system
    assert "Tool Guidelines" not in system


# -- a workspace update is delivered once -------------------------------------

# A MEMORY.md a compaction has just rewritten: every line changed, so the
# update is the whole file rather than a diff.
_COMPACTED = "\n".join(
    f"- Compacted fact {i}: the job at site {i} is scheduled" for i in range(400)
)


def _reply(t: _Rows, minute: float, text: str = "ok") -> None:
    t.rows.append(
        StoredMessage(
            direction="outbound",
            body=text,
            llm_reply_text=text,
            timestamp=_at(minute),
            seq=len(t.rows) + 1,
        )
    )


class _Session:
    """Assembles one turn after another over a growing transcript."""

    def __init__(self, user: User) -> None:
        self.user = user
        self.t = _Rows()

    async def turn(self, minute: float, ask: str) -> AssembledPrompt:
        current = self.t.ask(minute, ask)
        view = build_history_view(self.t.rows[:-1], current, "", compact=False, tools_by_name={})
        agent = ClawboltAgent(user=self.user)
        agent.register_tools(_tools())
        assembled = await agent.assemble_prompt(
            ask, view.messages, epoch=view.epoch, deterministic_trim=True
        )
        _reply(self.t, minute + 1)
        return assembled


def _current(assembled: AssembledPrompt) -> str:
    last = assembled.messages[-1]
    assert isinstance(last, UserMessage)
    return last.content


def _prior(assembled: AssembledPrompt) -> list[dict[str, Any]]:
    """Everything before the current turn: the part the cache can hold."""
    return _api(assembled.messages[:-1])


async def test_a_workspace_update_is_sent_once_then_read_from_the_cache(
    test_user: User, one_hour_cache: None
) -> None:
    await write_memory(test_user.id, "- Customer prefers mornings")
    s = _Session(test_user)
    cold = await s.turn(0, "hi")
    assert "Workspace Updates" not in _current(cold)

    await write_memory(test_user.id, _COMPACTED)
    first = await s.turn(5, "what is on today?")
    assert "## Workspace Updates\nMEMORY.md changed" in _current(first)
    assert _COMPACTED in _current(first)

    second = await s.turn(10, "and tomorrow?")
    third = await s.turn(15, "thanks")
    for later in (second, third):
        assert "Workspace Updates" not in _current(later)
        assert _COMPACTED not in _current(later)
        assert later.stable_system == cold.stable_system
    # The model still has the file: on the turn it was delivered, in history.
    delivered = [
        m for m in second.messages if isinstance(m, UserMessage) and _COMPACTED in m.content
    ]
    assert len(delivered) == 1
    assert "what is on today?\n\n## Workspace Updates\nMEMORY.md changed" in delivered[0].content
    # The prefix only grows by appending, so the next turn reads it from cache.
    assert _prior(third)[: len(_prior(second))] == _prior(second)


async def test_a_second_change_is_a_diff_against_the_first(
    test_user: User, one_hour_cache: None
) -> None:
    await write_memory(test_user.id, "- Customer prefers mornings")
    s = _Session(test_user)
    await s.turn(0, "hi")
    await write_memory(test_user.id, _COMPACTED)
    await s.turn(5, "what is on today?")
    before = await s.turn(10, "ok")

    await write_memory(test_user.id, _COMPACTED + "\n- Gate code is 4321")
    changed = await s.turn(15, "remember the gate code")
    update = _current(changed)
    assert "MEMORY.md changed since it was last shown above. Apply this diff" in update
    assert "+- Gate code is 4321" in update
    assert "It now reads" not in update
    assert len(update) < 1_000

    after = await s.turn(20, "thanks")
    assert "Workspace Updates" not in _current(after)
    assert _prior(changed)[: len(_prior(before))] == _prior(before)
    assert _prior(after)[: len(_prior(changed))] == _prior(changed)
    carried = [m.content for m in after.messages[:-1] if isinstance(m, UserMessage)]
    assert sum(_COMPACTED in c for c in carried) == 1
    assert sum("+- Gate code is 4321" in c for c in carried) == 1


async def test_the_next_cold_start_drops_the_delivered_updates(
    test_user: User, one_hour_cache: None
) -> None:
    await write_memory(test_user.id, "- Customer prefers mornings")
    s = _Session(test_user)
    await s.turn(0, "hi")
    await write_memory(test_user.id, _COMPACTED)
    await s.turn(5, "what is on today?")
    await s.turn(10, "ok")

    cold = await s.turn(500, "back again")
    assert f"## Your Memory\n{_COMPACTED}" in cold.stable_system
    assert not any(
        "Workspace Updates" in m.content for m in cold.messages if isinstance(m, UserMessage)
    )
    warm = await s.turn(505, "and now?")
    assert warm.stable_system == cold.stable_system
    assert _prior(warm)[: len(_prior(cold))] == _prior(cold)


async def test_an_update_whose_turn_left_the_history_is_sent_again(
    test_user: User, one_hour_cache: None
) -> None:
    """A trim that drops the delivering turn must not leave the model without the file."""
    await write_memory(test_user.id, "- Customer prefers mornings")
    s = _Session(test_user)
    await s.turn(0, "hi")
    await write_memory(test_user.id, _COMPACTED)
    first = await s.turn(5, "what is on today?")
    assert _COMPACTED in _current(first)

    # The trim watermark moved past the delivering turn: the next load starts after it.
    s.t.rows = s.t.rows[4:]
    again = await s.turn(10, "and tomorrow?")
    assert _COMPACTED in _current(again)
    assert "Customer prefers mornings" in again.stable_system


async def test_a_trim_in_the_same_turn_resends_the_update(
    test_user: User, one_hour_cache: None
) -> None:
    await write_memory(test_user.id, "- Customer prefers mornings")
    s = _Session(test_user)
    await s.turn(0, "hi")
    await write_memory(test_user.id, _COMPACTED)
    await s.turn(5, "what is on today?")
    for i in range(6):
        await s.turn(10 + i * 5, f"filler {i}")
    with (
        patch.object(settings, "context_trim_target_tokens", 3_000),
        patch.object(settings, "context_trim_trigger_tokens", 6_000),
    ):
        trimmed = await s.turn(60, "and now?")
    assert trimmed.trimmed_count > 0
    history = [m.content for m in trimmed.messages[1:-1] if isinstance(m, UserMessage)]
    assert not any(_COMPACTED in c for c in history)
    current = _current(trimmed)
    assert current.startswith("[Summary of earlier conversation:")
    assert f"MEMORY.md changed since it was last shown above. It now reads:\n{_COMPACTED}" in (
        current
    )
    assert _COMPACTED in trimmed.dynamic_context
    # The trim ran on the rows as stored: the delivering turn went to
    # compaction without the update the view had appended to it.
    dropped = [m.content for m in trimmed.dropped if isinstance(m, UserMessage)]
    assert "what is on today?" in dropped
    assert not any("Workspace Updates" in c for c in dropped)


def _user_texts(messages: list[dict[str, Any]]) -> list[str]:
    """The text of every user message sent, current turn included."""
    texts = []
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.extend(b["text"] for b in content if b.get("type") == "text")
    return texts


async def _delivered_then_grown(user: User, write_result: str) -> _Rows:
    """A memory change delivered on a small turn, then eight big turns and an ask.

    Returns the rows, all persisted, with the ask unanswered. The delivering
    turn is the ask ``"what is on today?"`` (seq 5).
    """
    await write_memory(user.id, "- Customer prefers mornings")
    t = _Rows()
    t.turn(0, "hi", "hello")
    t.ask(2, "first")
    await create_test_session(user.id, messages=t.rows)
    await _live_turn(user)

    await write_memory(user.id, _COMPACTED)
    reply = StoredMessage(direction="outbound", body="Done.", timestamp=_at(3), seq=4)
    t.rows.append(reply)
    ask = t.ask(4, "what is on today?")
    await _append_rows(user.id, [reply, ask])
    delivering = _user_texts(await _live_turn(user))[-1]
    assert f"It now reads:\n{_COMPACTED}" in delivering

    grown = _read_and_write_turns(write_result).rows[:-1]
    reply = StoredMessage(direction="outbound", body="Done.", timestamp=_at(5), seq=6)
    t.rows.append(reply)
    base = len(t.rows)
    for row in grown:
        t.rows.append(
            row.model_copy(
                update={
                    "seq": row.seq + base,
                    "timestamp": _at(6 + row.seq),
                }
            )
        )
    t.ask(6 + len(grown) + 1, "what now?")
    await _append_rows(user.id, t.rows[base - 1 :])
    return t


async def test_a_shed_keeps_a_delivered_update_and_the_next_turn_matches(
    test_user: User, trim_settings: AsyncMock
) -> None:
    """The shed re-renders the history from the rows; the update delivered on
    an earlier turn is laid onto the row it rode after that, so the model
    still has it, the current turn does not repeat it, and the next turn
    sends the shed turn's history byte for byte."""
    t = await _delivered_then_grown(test_user, WRITE_RESULT)

    first = await _live_turn(test_user)

    state, _ = await get_session_store(test_user.id).get_or_create_session()
    assert state.history_stub_seq is not None
    assert state.last_trim_seq is None
    stub = elided_result_stub("web_search", len(HUGE))
    assert stub in _results_by_id(first).values()
    texts = _user_texts(first)
    assert sum(_COMPACTED in c for c in texts) == 1
    carried = next(c for c in texts if _COMPACTED in c)
    assert "what is on today?\n\n## Workspace Updates\nMEMORY.md changed" in carried
    assert "Workspace Updates" not in texts[-1]

    await _next_turn(test_user, t, 60)
    second = await _live_turn(test_user)

    assert second[: len(first) - 1] == first[:-1]
    assert "Workspace Updates" not in _user_texts(second)[-1]


async def test_a_shed_that_drops_the_delivering_turn_sends_the_update_again(
    test_user: User, trim_settings: AsyncMock
) -> None:
    """Writes alone over target: the shed drops the oldest turns, the
    delivering one among them. The current turn carries the change again,
    and compaction reads the dropped turn as stored."""
    await _delivered_then_grown(test_user, HUGE_WRITE)
    trim_settings.reset_mock()

    sent = await _live_turn(test_user)

    state, _ = await get_session_store(test_user.id).get_or_create_session()
    assert state.last_trim_seq is not None and state.last_trim_seq >= 5
    texts = _user_texts(sent)
    assert not any(_COMPACTED in c for c in texts[:-1])
    assert f"It now reads:\n{_COMPACTED}" in texts[-1]
    assert trim_settings.await_args is not None
    compacted = [m.content for m in trim_settings.await_args.args[1] if isinstance(m, UserMessage)]
    assert "what is on today?" in compacted
    assert not any("Workspace Updates" in c for c in compacted)


async def test_the_overflow_retry_resends_an_update_it_trims_away(
    test_user: User, one_hour_cache: None
) -> None:
    """The provider refuses the prompt and the retry's trim drops the turn
    that carried the update. The retry sends the change on the current turn,
    and the dropped turn reaches compaction as stored."""
    await write_memory(test_user.id, "- Customer prefers mornings")
    s = _Session(test_user)
    await s.turn(0, "hi")
    await write_memory(test_user.id, _COMPACTED)
    await s.turn(5, "what is on today?")
    for i in range(3):
        await s.turn(10 + i * 5, f"filler {i}")

    current = s.t.ask(30, "and now?")
    view = build_history_view(s.t.rows[:-1], current, "", compact=False, tools_by_name={})
    agent = ClawboltAgent(user=test_user)
    agent.register_tools(_tools())
    with (
        patch(
            "backend.app.agent.core.amessages_streamed",
            new_callable=AsyncMock,
            side_effect=[ContextLengthExceededError("Input too long"), make_text_response("ok")],
        ) as llm,
        patch(
            "backend.app.agent.core.trigger_compaction_for_dropped", new_callable=AsyncMock
        ) as compact,
    ):
        await agent.process_message("and now?", view.messages, prompt_epoch=view.epoch)

    refused, retried = (_user_texts(c.kwargs["messages"]) for c in llm.call_args_list)
    assert sum(_COMPACTED in c for c in refused) == 1
    assert "Workspace Updates" not in refused[-1]
    assert not any(_COMPACTED in c for c in retried[:-1])
    assert f"It now reads:\n{_COMPACTED}" in retried[-1]
    assert retried[-1].startswith("[Summary of earlier conversation:")
    assert compact.await_args is not None
    dropped = [m.content for m in compact.await_args.args[1] if isinstance(m, UserMessage)]
    assert "what is on today?" in dropped
    assert not any("Workspace Updates" in c for c in dropped)


# -- the comparison harness ---------------------------------------------------


async def test_harness_replays_with_and_without_the_rebuild(
    test_user: User, one_hour_cache: None
) -> None:
    t = _long_history(8)
    current = t.ask(600, "back after lunch")
    fixture = ReplayFixture(user=test_user, rows=t.rows, tools_by_name=dict(TOOLS))
    sample = ReplaySample(
        seq=current.seq, timestamp=current.timestamp, message_context=current.body
    )

    full = await assemble_for_sample(fixture, sample, HistoryMode.FULL)
    compacted = await assemble_for_sample(fixture, sample, HistoryMode.COLD_START_COMPACTION)

    def results(messages: list[AgentMessage]) -> list[str]:
        return [m.content for m in messages if isinstance(m, ToolResultMessage)]

    assert all(r in (BIG, "short result") for r in results(full.messages))
    assert any(r.startswith("[tool result elided") for r in results(compacted.messages))
    _assert_pairs_intact(compacted.messages)
    # Same system block and same current turn: only the history differs.
    assert full.stable_system == compacted.stable_system
    assert full.messages[-1] == compacted.messages[-1]


# -- SKILL.md guidance on old results ------------------------------------------

QB_GUIDANCE = "## QuickBooks\nLook up the customer before creating an invoice.\n" + "g" * 2_000
QB_WRITE = "ok | Id: 643"


def _guided(result: str, *, legacy: bool = False) -> str:
    """*result* as stored when first-use injection rode on it."""
    if legacy:
        return f"{result}\n\n{skill_delivery_marker('quickbooks')}\n{QB_GUIDANCE}"
    return result + skill_guidance_block("quickbooks", QB_GUIDANCE)


def _guided_history(*, legacy: bool = False) -> _Rows:
    """An old turn whose write and read both carry guidance, then quiet turns."""
    t = _Rows()
    _calls_turn(
        t,
        0,
        "invoice acme",
        [
            ("qb_create", {"query": "invoice"}, _guided(QB_WRITE, legacy=legacy), False),
            ("web_search", {"query": "acme"}, _guided("short read", legacy=legacy), False),
        ],
    )
    for i in range(4):
        t.turn(5 + i * 5, f"ask {i}", f"reply {i}")
    return t


@pytest.mark.parametrize("legacy", [False, True])
def test_old_write_keeps_its_result_and_loses_its_guidance(
    one_hour_cache: None, legacy: bool
) -> None:
    t = _guided_history(legacy=legacy)
    current = t.ask(500, "back")
    with patch.object(settings, "cold_start_verbatim_turns", 1):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    results = _results(view.messages)
    assert results["call_1_0"].content == QB_WRITE
    # A short read keeps its result too, once the guidance is gone.
    assert results["call_1_1"].content == "short read"
    assert not any(extract_delivered_skills(r.content) for r in results.values())
    _assert_pairs_intact(view.messages)


def test_old_read_with_guidance_is_stubbed_as_before(one_hour_cache: None) -> None:
    t = _Rows()
    _calls_turn(t, 0, "look up", [("web_search", {"query": "x"}, _guided(BIG), False)])
    for i in range(4):
        t.turn(5 + i * 5, f"ask {i}", f"reply {i}")
    current = t.ask(500, "back")
    with patch.object(settings, "cold_start_verbatim_turns", 1):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    assert _results(view.messages)["call_1_0"].content == elided_result_stub("web_search", len(BIG))
    assert view.elided_results == 1


def test_guidance_inside_the_verbatim_window_stays(one_hour_cache: None) -> None:
    """Recent guidance may be in use: the verbatim window is left as stored."""
    t = _Rows()
    for i in range(4):
        t.turn(i * 5, f"ask {i}", f"reply {i}")
    _calls_turn(t, 30, "invoice acme", [("qb_create", {"query": "i"}, _guided(QB_WRITE), False)])
    current = t.ask(500, "back")
    with patch.object(settings, "cold_start_verbatim_turns", 2):
        view = build_history_view(t.rows[:-1], current, "", compact=True, tools_by_name=TOOLS)
    assert _results(view.messages)["call_9_0"].content == _guided(QB_WRITE)


def test_guidance_stripping_is_deterministic_and_a_warm_turn_only_appends(
    one_hour_cache: None,
) -> None:
    t = _guided_history()
    cold = t.ask(500, "back")
    with patch.object(settings, "cold_start_verbatim_turns", 1):
        first = build_history_view(t.rows[:-1], cold, "", compact=True, tools_by_name=TOOLS)
        repeat = build_history_view(t.rows[:-1], cold, "", compact=True, tools_by_name=TOOLS)
        assert _api(repeat.messages) == _api(first.messages)
        # The cold turn writes again and the guidance is re-delivered on it.
        t.rows.append(
            StoredMessage(
                direction="outbound",
                body="sent",
                llm_reply_text="sent",
                tool_interactions_json=json.dumps(
                    [
                        {
                            "tool_call_id": "c",
                            "name": "qb_create",
                            "args": {"query": "again"},
                            "result": _guided(QB_WRITE),
                        }
                    ]
                ),
                timestamp=_at(502),
                seq=cold.seq + 1,
            )
        )
        warm = t.ask(510, "and the other one?")
        second = build_history_view(t.rows[:-1], warm, "", compact=True, tools_by_name=TOOLS)
    assert second.epoch == PromptEpoch(key=cold.seq, cold_start=False)
    before, after = _api(first.messages), _api(second.messages)
    assert after[: len(before)] == before
    # The re-delivery was appended during the epoch, so it is not stripped.
    assert _results(second.messages)["c"].content == _guided(QB_WRITE)
    _assert_pairs_intact(second.messages)


def _qb_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "quickbooks",
        lambda ctx: [TOOLS["qb_create"]],
        core=False,
        summary="QuickBooks",
        sub_tools=[SubToolInfo("qb_create", "Create")],
    )
    return reg


def _sent_results(messages: list[dict[str, Any]]) -> list[str]:
    return [
        block["content"]
        for msg in messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if block.get("type") == "tool_result"
    ]


@patch("backend.app.agent.core.amessages_streamed")
async def test_stripped_guidance_is_delivered_again_once(
    mock_amessages: AsyncMock, test_user: User, one_hour_cache: None
) -> None:
    """The next use of the category after a cold start carries the guidance,
    and a later turn in the same epoch does not get a second copy."""
    t = _guided_history()
    cold = t.ask(500, "invoice acme again")
    with (
        patch.object(settings, "cold_start_verbatim_turns", 1),
        patch.dict(loader._skill_instructions, {"quickbooks": QB_GUIDANCE}),
    ):
        view = build_history_view(t.rows[:-1], cold, "", compact=True, tools_by_name=TOOLS)
        mock_amessages.side_effect = [
            make_tool_call_response([{"name": "qb_create", "arguments": {"query": "again"}}]),
            make_text_response("sent"),
        ]
        agent = ClawboltAgent(user=test_user, registry=_qb_registry())
        agent.register_tools([TOOLS["qb_create"]])
        response = await agent.process_message(
            "invoice acme again", view.messages, prompt_epoch=view.epoch
        )
        assert response.tool_calls[0].result == "ok" + skill_guidance_block(
            "quickbooks", QB_GUIDANCE
        )
        # The follow-up call carries one copy: the old one is gone.
        sent = _sent_results(mock_amessages.call_args_list[1].kwargs["messages"])
        assert sum(skill_delivery_marker("quickbooks") in r for r in sent) == 1

        # The next turn in the epoch finds the re-delivery in an appended row.
        t.rows.append(
            StoredMessage(
                direction="outbound",
                body="sent",
                llm_reply_text="sent",
                tool_interactions_json=json.dumps(
                    [r.model_dump(mode="json") for r in response.tool_calls]
                ),
                timestamp=_at(502),
                seq=cold.seq + 1,
            )
        )
        warm = t.ask(510, "one more")
        warm_view = build_history_view(t.rows[:-1], warm, "", compact=True, tools_by_name=TOOLS)
        assert _api(warm_view.messages)[: len(_api(view.messages))] == _api(view.messages)
        mock_amessages.reset_mock()
        mock_amessages.side_effect = [
            make_tool_call_response([{"name": "qb_create", "arguments": {"query": "more"}}]),
            make_text_response("sent"),
        ]
        agent = ClawboltAgent(user=test_user, registry=_qb_registry())
        agent.register_tools([TOOLS["qb_create"]])
        again = await agent.process_message(
            "one more", warm_view.messages, prompt_epoch=warm_view.epoch
        )
    assert again.tool_calls[0].result == "ok"


# -- the mid-session trim sheds stale reads first ------------------------------

# Sized so the history dominates the prompt whatever the system prompt weighs.
HUGE = "r" * 50_000
# Eight of these are over the trim target on their own.
HUGE_WRITE = "Created invoice 643. " + "w" * 30_000


async def _append_rows(user_id: str, rows: list[StoredMessage]) -> None:
    """Persist *rows* at their own timestamps (``add_message`` stamps now)."""
    async with db_session_async() as db:
        cs = (await db.execute(select(ChatSession).filter_by(user_id=user_id))).scalar_one()
        for row in rows:
            db.add(
                Message(
                    session_id=cs.id,
                    seq=row.seq,
                    direction=row.direction,
                    body=row.body,
                    processed_context=row.processed_context,
                    tool_interactions_json=row.tool_interactions_json,
                    timestamp=_dt.datetime.fromisoformat(row.timestamp),
                )
            )
        await db.commit()


def _without_cache_markers(value: Any) -> Any:
    """*value* with ``cache_control`` removed: the breakpoint moves every turn."""
    if isinstance(value, dict):
        return {k: _without_cache_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_without_cache_markers(v) for v in value]
    return value


async def _live_turn(user: User) -> list[dict[str, Any]]:
    """Run the live pipeline's history and agent steps; return what was sent."""
    state, _ = await get_session_store(user.id).get_or_create_session()
    ctx = PipelineContext(user=user, session=state, message=state.messages[-1], media_urls=[])
    with patch(
        "backend.app.agent.core.amessages_streamed",
        new_callable=AsyncMock,
        return_value=make_text_response("Done."),
    ) as llm:
        ctx = await load_history_step(ctx)
        ctx = await run_agent_step(ctx)
    assert ctx.response is not None and ctx.response.reply_text == "Done."
    await asyncio.gather(*context_module._background_tasks)
    return _without_cache_markers(llm.call_args.kwargs["messages"])


@pytest.fixture()
def trim_settings(one_hour_cache: None) -> Iterator[AsyncMock]:
    """Compaction on, a two-turn verbatim window, and a trim at 90k down to 50k.

    The trimmer estimates from characters (no recalled API count), so the
    thresholds read the bytes the test builds. Yields the compaction mock.
    """
    with (
        patch.object(settings, "cold_start_compaction_enabled", True),
        patch.object(settings, "compaction_enabled", True),
        patch.object(settings, "cold_start_verbatim_turns", 2),
        patch.object(settings, "context_trim_target_tokens", 50_000),
        patch.object(settings, "context_trim_trigger_tokens", 90_000),
        patch("backend.app.agent.core._recall_input_tokens", return_value=None),
        patch(
            "backend.app.agent.router.assemble_turn_tools",
            AsyncMock(return_value=(list(TOOLS.values()), {})),
        ),
        patch(
            "backend.app.agent.context.compact_session",
            new_callable=AsyncMock,
            return_value=("", False),
        ) as compact,
    ):
        yield compact


def _results_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    return {
        block["tool_use_id"]: block["content"]
        for msg in messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if block.get("type") == "tool_result"
    }


def _read_and_write_turns(write_result: str) -> _Rows:
    """Eight turns, each a big lookup and a write, then an unanswered ask."""
    t = _Rows()
    for i in range(8):
        _calls_turn(
            t,
            i * 5,
            f"ask {i}",
            [
                ("web_search", {"query": f"q{i}"}, HUGE, False),
                ("qb_create", {"query": f"invoice {i}"}, write_result, False),
            ],
        )
    t.ask(40, "what now?")
    return t


async def _next_turn(user: User, t: _Rows, minute: float) -> None:
    """Persist the reply to the last ask and a new ask, both inside the epoch."""
    reply = StoredMessage(
        direction="outbound", body="Done.", timestamp=_at(minute), seq=len(t.rows) + 1
    )
    t.rows.append(reply)
    ask = t.ask(minute + 4, "and next?")
    await _append_rows(user.id, [reply, ask])


async def test_trim_stubs_old_reads_and_the_next_turn_renders_the_same_bytes(
    test_user: User, trim_settings: AsyncMock
) -> None:
    """Over the trigger, stubbing old reads is enough: writes and the verbatim
    window stay whole, no turn is dropped, nothing is compacted, and the next
    turn sends the trimmed history byte for byte without trimming again."""
    t = _read_and_write_turns(WRITE_RESULT)
    await create_test_session(test_user.id, messages=t.rows)

    first = await _live_turn(test_user)

    results = _results_by_id(first)
    stub = elided_result_stub("web_search", len(HUGE))
    # Turn i is seqs 2i+1 and 2i+2; the last two turns are the verbatim window.
    assert [results[f"call_{2 * i + 1}_0"] == stub for i in range(8)] == [True] * 6 + [False] * 2
    assert results["call_15_0"] == HUGE
    assert all(results[f"call_{2 * i + 1}_1"] == WRITE_RESULT for i in range(8))
    # Every turn is still there, the oldest first, with no summary note.
    assert "ask 0" in json.dumps(first[0])
    assert "Summary of earlier conversation" not in json.dumps(first)
    state, _ = await get_session_store(test_user.id).get_or_create_session()
    assert state.last_trim_seq is None
    assert state.history_stub_seq == 13
    assert trim_settings.await_count == 0

    await _next_turn(test_user, t, 42)
    second = await _live_turn(test_user)

    assert second[: len(first) - 1] == first[:-1]
    state, _ = await get_session_store(test_user.id).get_or_create_session()
    assert state.history_stub_seq == 13
    assert state.last_trim_seq is None
    assert trim_settings.await_count == 0


async def test_trim_drops_the_oldest_turns_when_stubs_are_not_enough(
    test_user: User, trim_settings: AsyncMock
) -> None:
    """Write results are never stubbed, so when they alone are over target,
    with every read stubbed (the verbatim window included), the oldest turns
    go whole: compacted from their full rows, the watermark past them, and
    the next turn renders what this one sent."""
    t = _read_and_write_turns(HUGE_WRITE)
    await create_test_session(test_user.id, messages=t.rows)

    first = await _live_turn(test_user)

    state, _ = await get_session_store(test_user.id).get_or_create_session()
    assert state.last_trim_seq is not None
    dropped_turns = state.last_trim_seq // 2
    assert 1 <= dropped_turns < 8
    assert state.last_trim_seq == 2 * dropped_turns
    assert "ask 0" not in json.dumps(first[0])
    assert "Summary of earlier conversation" in json.dumps(first[-1])
    results = _results_by_id(first)
    # Kept writes are whole. Every kept read is a stub: the verbatim window
    # shrank to nothing before any turn was dropped, and the turns kept leave
    # no room to widen it again for a read this size.
    assert sum(r == HUGE_WRITE for r in results.values()) == 8 - dropped_turns
    stub = elided_result_stub("web_search", len(HUGE))
    assert all(results[f"call_{2 * i + 1}_0"] == stub for i in range(dropped_turns, 8))
    assert state.history_stub_seq == 17
    # One compaction, over the dropped rows as stored: reads unstubbed.
    assert trim_settings.await_count == 1
    assert trim_settings.await_args is not None
    compacted = trim_settings.await_args.args[1]
    assert HUGE in [m.content for m in compacted if isinstance(m, ToolResultMessage)]
    seqs = {m.seq for m in compacted if isinstance(m, UserMessage | AssistantMessage)}
    assert seqs == set(range(1, state.last_trim_seq + 1))

    await _next_turn(test_user, t, 42)
    second = await _live_turn(test_user)

    assert second[: len(first) - 1] == first[:-1]
    assert trim_settings.await_count == 1
    again, _ = await get_session_store(test_user.id).get_or_create_session()
    assert again.last_trim_seq == state.last_trim_seq
    assert again.history_stub_seq == state.history_stub_seq


def _chars(messages: list[AgentMessage]) -> int:
    return sum(len(m.content) for m in messages if isinstance(m.content, str))


def _writes_then_reads() -> tuple[_Rows, StoredMessage]:
    """Six turns that each write, then two that each read a huge result."""
    t = _Rows()
    for i in range(6):
        _calls_turn(t, i * 5, f"ask {i}", [("qb_create", {"query": f"inv {i}"}, HUGE_WRITE, False)])
    for i in range(6, 8):
        _calls_turn(t, i * 5, f"ask {i}", [("web_search", {"query": f"q{i}"}, HUGE, False)])
    return t, t.ask(40, "what now?")


@pytest.mark.parametrize(
    ("limit", "verbatim_reads"),
    [
        # Stubbing both reads is the only way under: the window goes to zero.
        (200_000, 0),
        # One read fits beside the writes: the window keeps the newest turn.
        (245_000, 1),
    ],
)
def test_shed_shrinks_the_verbatim_window_before_dropping_a_turn(
    limit: int, verbatim_reads: int
) -> None:
    """Recent reads can be fetched again; prose and write results cannot. So
    the shed stubs reads inside the verbatim window before it drops a turn."""
    t, current = _writes_then_reads()
    rows = t.rows[:-1]

    with patch.object(settings, "cold_start_verbatim_turns", 2):
        shed = shed_history_view(
            rows,
            current,
            "",
            tools_by_name=TOOLS,
            fits=lambda messages: _chars(messages) <= limit,
        )

    assert shed.view.dropped_rows == []
    results = [m.content for m in shed.view.messages if isinstance(m, ToolResultMessage)]
    assert results[:6] == [HUGE_WRITE] * 6
    stub = elided_result_stub("web_search", len(HUGE))
    assert results[6:] == [stub] * (2 - verbatim_reads) + [HUGE] * verbatim_reads
    # The stub seq is the first verbatim row, or the current turn with none.
    assert shed.stub_before == (rows[-2].seq if verbatim_reads else current.seq)


async def test_replay_over_the_trigger_sheds_as_live_does_and_writes_nothing(
    test_user: User, trim_settings: AsyncMock
) -> None:
    """The comparison replay of a turn over the trim trigger stubs the same
    reads the live trim does, and leaves the session as it found it."""
    t = _read_and_write_turns(WRITE_RESULT)
    await create_test_session(test_user.id, messages=t.rows)
    current = t.rows[-1]
    fixture = ReplayFixture(user=test_user, rows=t.rows, tools_by_name=dict(TOOLS))
    sample = ReplaySample(
        seq=current.seq, timestamp=current.timestamp, message_context=current.body
    )

    replayed = await assemble_for_sample(fixture, sample, HistoryMode.COLD_START_COMPACTION)

    state, _ = await get_session_store(test_user.id).get_or_create_session()
    assert state.history_stub_seq is None
    assert state.last_trim_seq is None
    assert trim_settings.await_count == 0

    live = _results_by_id(await _live_turn(test_user))
    replay = {
        m.tool_call_id: m.content for m in replayed.messages if isinstance(m, ToolResultMessage)
    }
    stub = elided_result_stub("web_search", len(HUGE))
    assert sum(r == stub for r in replay.values()) == 6
    assert replay == live
