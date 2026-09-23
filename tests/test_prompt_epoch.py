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

import datetime as _dt
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from backend.app.agent.context import (
    _stored_messages_to_agent_messages,
    load_conversation_history,
)
from backend.app.agent.core import ClawboltAgent
from backend.app.agent.dto import StoredMessage
from backend.app.agent.memory_db import write_memory
from backend.app.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
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
)
from backend.app.agent.session_db import get_session_store
from backend.app.agent.system_prompt import WorkspaceSnapshot, render_workspace_updates
from backend.app.agent.tools.base import Tool, ToolResult
from backend.app.config import settings
from backend.app.models import User
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
    ):
        yield


# -- the definition of a cold start ------------------------------------------


def test_cold_gap_follows_the_cache_ttl() -> None:
    with patch.object(settings, "prompt_cache_idle_seconds", None):
        with patch.object(settings, "llm_cache_extended_ttl", True):
            assert cache_idle_threshold() == _dt.timedelta(hours=1)
        with patch.object(settings, "llm_cache_extended_ttl", False):
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
        view = build_history_view(t.rows[:-1], current, "", compact=True)

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
        view = build_history_view(t.rows[:-1], current, "", compact=True)
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
        view = build_history_view(t.rows[:-1], current, "", compact=True)
        tokens = sum(len(m.content or "") for m in view.messages) // 2
        assert tokens <= 4_000
        assert view.dropped_rows
        assert view.dropped_rows[0].seq == 1
        # A fixed point: the rows that remain rebuild to the same bytes and
        # drop nothing more, so the rest of the epoch appends to this.
        kept = [r for r in t.rows[:-1] if r.seq > view.dropped_rows[-1].seq]
        again = build_history_view(kept, current, "", compact=True)
    assert not again.dropped_rows
    assert _api(again.messages) == _api(view.messages)


def test_rebuild_is_deterministic_and_a_warm_turn_only_appends(one_hour_cache: None) -> None:
    t = _long_history(8)
    cold = t.ask(500, "back after lunch")
    with patch.object(settings, "cold_start_history_budget_tokens", 30_000):
        first = build_history_view(t.rows[:-1], cold, "", compact=True)
        assert _api(build_history_view(t.rows[:-1], cold, "", compact=True).messages) == _api(
            first.messages
        )

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
        second = build_history_view(t.rows[:-1], warm, "", compact=True)

    assert second.epoch == PromptEpoch(key=cold.seq, cold_start=False)
    before, after = _api(first.messages), _api(second.messages)
    assert after[: len(before)] == before
    # What was appended is verbatim: the new turn's result is not elided.
    appended = second.messages[len(first.messages) :]
    assert any(isinstance(m, ToolResultMessage) and m.content == BIG for m in appended)


def test_no_cold_gap_means_nothing_is_rebuilt(one_hour_cache: None) -> None:
    t = _long_history(10)
    current = t.ask(60, "still here")
    view = build_history_view(t.rows[:-1], current, "", compact=True)
    assert view.elided_results == 0
    assert _api(view.messages) == _api(_stored_messages_to_agent_messages(t.rows[:-1]))


def test_compaction_off_renders_exactly_the_old_history(one_hour_cache: None) -> None:
    t = _long_history(10)
    current = t.ask(900, "back")
    view = build_history_view(t.rows[:-1], current, "", compact=False)
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
        renderer = EpochHistoryRenderer(test_user.id, compact=True)
        first = await load_conversation_history(state, render=renderer)
        assert renderer.epoch is not None and renderer.epoch.cold_start

        state, _ = await get_session_store(test_user.id).get_or_create_session()
        assert state.last_trim_seq is not None and state.last_trim_seq > 0
        again = await load_conversation_history(
            state, render=EpochHistoryRenderer(test_user.id, compact=True)
        )

    assert compact.await_count == 1
    assert _api(again) == _api(first)


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


@patch("backend.app.agent.core.amessages")
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
    assert "Removed:\n- - Customer prefers mornings" in current_turn
    assert "Added:\n+ - Customer prefers afternoons" in current_turn
    # Only the change rides the turn, not the unchanged facts.
    assert "Fact number 5" not in current_turn


def test_a_change_larger_than_the_file_sends_the_file() -> None:
    old = WorkspaceSnapshot(soul="s", user="u", memory="- one")
    new = WorkspaceSnapshot(soul="s", user="u", memory="- two")
    updates = render_workspace_updates(old, new)
    assert (
        updates == 'MEMORY.md changed after "Your Memory" above was captured. It now reads:\n- two'
    )
    assert render_workspace_updates(old, old) == ""


@patch("backend.app.agent.core.amessages")
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


@patch("backend.app.agent.core.amessages")
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


@patch("backend.app.agent.core.amessages")
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


@patch("backend.app.agent.core.amessages")
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


# -- the comparison harness ---------------------------------------------------


async def test_harness_replays_with_and_without_the_rebuild(
    test_user: User, one_hour_cache: None
) -> None:
    t = _long_history(8)
    current = t.ask(600, "back after lunch")
    fixture = ReplayFixture(user=test_user, rows=t.rows)
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
