"""Model dispatch for the model-swap evaluator.

The replay has to ask a model the way production asks it. Where the two
differ, the report charges the model for the harness: a reply production
would have retried is scored as a truncation, and a thinking budget that
cannot fit under ``max_tokens`` is scored as a provider refusal.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock, ToolUseBlock

from backend.app.agent.core import AssembledPrompt
from backend.app.agent.messages import SystemMessage, UserMessage
from backend.app.services.llm_eval.execution import call_model
from backend.app.services.llm_service import LLMTarget

TARGET = LLMTarget(provider="anthropic", model="m")


def _prompt() -> AssembledPrompt:
    return AssembledPrompt(
        messages=[SystemMessage(content="system"), UserMessage(content="book it for 9am")],
        stable_system="system",
        dynamic_context="",
        system_prompt="system",
    )


def _response(
    *, text: str = "", tool: dict[str, Any] | None = None, stop: str = "end_turn"
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
    with patch("backend.app.services.llm_eval.execution.amessages", mock):
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
        kwargs = mock.await_args.kwargs
        assert kwargs["thinking"]["budget_tokens"] == budget
        assert kwargs["max_tokens"] > budget


async def test_a_budget_that_already_fits_is_left_alone() -> None:
    mock = AsyncMock(return_value=_response(text="ok"))
    await _call(mock, effort="low")
    assert mock.await_args.kwargs["max_tokens"] == 8192
