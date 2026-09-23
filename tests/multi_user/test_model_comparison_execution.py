"""Model dispatch for the model comparison report.

The replay has to ask a model the way production asks it. Where the two
differ, the report charges the model for the harness: a reply production
would have retried is scored as a truncation, and a thinking budget that
cannot fit under ``max_tokens`` is scored as a provider refusal.
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import AsyncMock, patch

from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock, ToolUseBlock
from pydantic import BaseModel

from backend.app.agent.core import AssembledPrompt
from backend.app.agent.messages import SystemMessage, UserMessage
from backend.app.agent.tools.base import Tool, ToolResult, ToolTags
from backend.app.config import settings
from backend.app.services.llm_service import LLMTarget
from backend.app.services.model_comparison.execution import MAX_REPLAY_READ_ROUNDS, call_model
from backend.app.services.model_comparison.types import RecordedToolResult

TARGET = LLMTarget(provider="anthropic", model="m")


def _prompt() -> AssembledPrompt:
    return AssembledPrompt(
        messages=[SystemMessage(content="system"), UserMessage(content="book it for 9am")],
        stable_system="system",
        dynamic_context="",
        system_prompt="system",
    )


def _response(
    *,
    text: str = "",
    tool: dict[str, Any] | None = None,
    stop: Literal["end_turn", "max_tokens", "tool_use"] = "end_turn",
) -> MessageResponse:
    content: list[Any] = []
    if text:
        content.append(TextBlock(type="text", text=text))
    if tool is not None:
        content.append(
            ToolUseBlock(type="tool_use", id="toolu_1", name=tool["name"], input=tool["input"])
        )
    return MessageResponse(
        id="msg_1",
        content=content,
        model="m",
        role="assistant",
        stop_reason=stop,
        type="message",
        usage=MessageUsage(input_tokens=100, output_tokens=50),
    )


async def _call(mock: AsyncMock, *, effort: str = "", target: LLMTarget = TARGET) -> Any:
    with patch("backend.app.services.model_comparison.execution.amessages", mock):
        return await call_model(_prompt(), None, target=target, reasoning_effort=effort)


async def test_truncated_reply_with_no_tool_call_is_retried_like_production() -> None:
    """Regression: the replay kept a reply production would have discarded.

    The live loop retries a response cut off at ``max_tokens`` with no tool
    call at double the budget. The replay counted it as a truncation against
    the model instead.
    """
    mock = AsyncMock(side_effect=[_response(stop="max_tokens"), _response(text="Booked.")])
    result = await _call(mock)

    assert result.stop_reason == "end_turn"
    assert result.text == "Booked."
    assert result.truncation_retries == 1
    budgets = [call.kwargs["max_tokens"] for call in mock.await_args_list]
    assert budgets == [8192, 16384]
    # The spent attempt is paid for in production, so it is counted here.
    assert result.input_tokens == 200
    assert result.output_tokens == 100


async def test_retry_stops_at_the_production_ceiling() -> None:
    mock = AsyncMock(return_value=_response(stop="max_tokens"))
    result = await _call(mock)
    assert result.stop_reason == "max_tokens"
    assert [c.kwargs["max_tokens"] for c in mock.await_args_list] == [8192, 16384]


async def test_truncated_tool_call_is_not_retried() -> None:
    """Production only retries a truncation that carries no tool call."""
    mock = AsyncMock(
        return_value=_response(tool={"name": "lookup", "input": {"q": "x"}}, stop="max_tokens")
    )
    result = await _call(mock)
    assert mock.await_count == 1
    assert result.truncation_retries == 0


async def test_a_thinking_budget_always_fits_under_max_tokens() -> None:
    """Regression: ``high`` asks for 24576 thinking tokens under an 8192 ceiling.

    Anthropic refuses that request, and a gateway that maps the budget to an
    effort spends the whole ceiling reasoning and truncates.
    """
    for effort, budget in (("high", 24576), ("xhigh", 32768)):
        mock = AsyncMock(return_value=_response(text="ok"))
        await _call(mock, effort=effort)
        kwargs = mock.await_args_list[-1].kwargs
        assert kwargs["thinking"]["budget_tokens"] == budget
        assert kwargs["max_tokens"] > budget


async def test_a_budget_that_already_fits_is_left_alone() -> None:
    mock = AsyncMock(return_value=_response(text="ok"))
    await _call(mock, effort="low")
    assert mock.await_args_list[-1].kwargs["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# Continuing through lookups the live turn made
# ---------------------------------------------------------------------------


class _SearchParams(BaseModel):
    query: str
    limit: int = 10


class _NoteParams(BaseModel):
    work_order_id: str
    body: str


async def _never(**_kwargs: object) -> ToolResult:  # pragma: no cover - never invoked
    raise AssertionError("a replay must never execute a tool")


TOOLS = {
    "search": Tool(
        name="search",
        description="search",
        function=_never,
        params_model=_SearchParams,
        tags={ToolTags.READ_ONLY},
    ),
    "add_note": Tool(
        name="add_note", description="add_note", function=_never, params_model=_NoteParams
    ),
}

RECORDED = (
    RecordedToolResult(
        name="search", arguments={"query": "12 Oak St"}, result="work order 71002 at 12 Oak St"
    ),
    RecordedToolResult(
        name="add_note", arguments={"work_order_id": "71002", "body": "done"}, result="ok"
    ),
)

SEARCH = {"name": "search", "input": {"query": "12 Oak St", "limit": 10}}
NOTE = {"name": "add_note", "input": {"work_order_id": "71002", "body": "done"}}


async def _replay(mock: AsyncMock, recorded: tuple[RecordedToolResult, ...] = RECORDED) -> Any:
    with patch("backend.app.services.model_comparison.execution.amessages", mock):
        return await call_model(
            _prompt(),
            None,
            target=TARGET,
            reasoning_effort="",
            tools_by_name=TOOLS,
            recorded=recorded,
        )


async def test_a_lookup_the_live_turn_made_is_answered_and_the_decision_scored() -> None:
    """Regression: only the first call was scored.

    A model that looked the work order up first, as production did, was
    scored on the search and lost to a model that guessed the ID and wrote.
    """
    mock = AsyncMock(side_effect=[_response(tool=SEARCH), _response(tool=NOTE)])
    result = await _replay(mock)

    assert [c.name for c in result.tool_calls] == ["add_note"]
    assert [lookup.name for lookup in result.replayed_lookups] == ["search"]
    assert result.replayed_lookups[0].result == "work order 71002 at 12 Oak St"
    # Usage covers both rounds.
    assert result.input_tokens == 200

    second_round = mock.await_args_list[1].kwargs["messages"]
    fed = second_round[-1]["content"][0]
    assert fed["type"] == "tool_result"
    assert fed["content"] == "work order 71002 at 12 Oak St"
    assert second_round[-2]["content"][-1]["name"] == "search"


async def test_a_default_spelled_out_still_matches_the_recorded_call() -> None:
    mock = AsyncMock(side_effect=[_response(tool=SEARCH), _response(text="Found it.")])
    result = await _replay(mock)
    assert mock.await_count == 2
    assert result.text == "Found it."


async def test_a_lookup_the_live_turn_did_not_make_ends_the_replay() -> None:
    """Answering it would need a live call, and a replay never makes one."""
    other = {"name": "search", "input": {"query": "14 Elm St"}}
    mock = AsyncMock(return_value=_response(tool=other))
    result = await _replay(mock)
    assert mock.await_count == 1
    assert result.tool_calls[0].arguments == {"query": "14 Elm St"}
    assert result.replayed_lookups == []


async def test_a_write_is_never_answered_even_when_the_live_turn_made_it() -> None:
    mock = AsyncMock(return_value=_response(tool=NOTE))
    result = await _replay(mock)
    assert mock.await_count == 1
    assert [c.name for c in result.tool_calls] == ["add_note"]


async def test_the_replay_gives_up_after_its_round_budget_and_says_so() -> None:
    """The cap is the measurement's limit, so the result carries that it hit.

    Without the flag the decision on record is a read, ``compare_writes``
    finds no matching write, and the card renders the red "Did not make the
    write" for a candidate that was cut off mid-lookup.
    """
    mock = AsyncMock(return_value=_response(tool=SEARCH))
    result = await _replay(mock)
    assert mock.await_count == MAX_REPLAY_READ_ROUNDS + 1
    assert [c.name for c in result.tool_calls] == ["search"]
    assert len(result.replayed_lookups) == MAX_REPLAY_READ_ROUNDS
    assert result.hit_read_round_cap is True


async def test_a_replay_that_decided_within_the_budget_is_not_marked_capped() -> None:
    """The flag means "was still asking", not "used every round". A model
    that answers on its last allowed round decided; it was not cut off."""
    rounds = [_response(tool=SEARCH)] * MAX_REPLAY_READ_ROUNDS + [_response(text="Found it.")]
    mock = AsyncMock(side_effect=rounds)
    result = await _replay(mock)
    assert mock.await_count == MAX_REPLAY_READ_ROUNDS + 1
    assert result.hit_read_round_cap is False
    assert result.text == "Found it."


async def test_a_single_round_decision_is_not_marked_capped() -> None:
    mock = AsyncMock(return_value=_response(tool=NOTE))
    result = await _replay(mock)
    assert result.hit_read_round_cap is False


async def test_the_round_cap_leaves_headroom_over_a_lookup_chain() -> None:
    """Six rather than three, and far under production's fifteen.

    Disambiguate the customer, find their job, pull the invoice, then write
    is four rounds. Three cut those turns off and reported them as missed
    writes; production's own ``max_tool_rounds`` is the other bound, and a
    replay must not be allowed to spend a whole agent loop per turn.
    """
    assert MAX_REPLAY_READ_ROUNDS == 6
    assert settings.max_tool_rounds > MAX_REPLAY_READ_ROUNDS


async def test_without_a_tool_set_the_replay_is_single_round() -> None:
    mock = AsyncMock(return_value=_response(tool=SEARCH))
    with patch("backend.app.services.model_comparison.execution.amessages", mock):
        await call_model(_prompt(), None, target=TARGET, reasoning_effort="", recorded=RECORDED)
    assert mock.await_count == 1
