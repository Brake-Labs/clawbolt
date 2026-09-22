"""Blinding and verdict mapping for the model-swap evaluator's judge.

The judge sees two unlabeled responses. Every test here is really the same
test: whichever slot the candidate landed in, the verdict has to come back
pointing at the candidate. Getting that mapping backwards would invert every
quality signal in the report while looking entirely plausible.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

from any_llm.exceptions import InvalidRequestError
from any_llm.types.messages import TextBlock, ToolUseBlock

from backend.app.services.llm_eval.judge import (
    _describe,
    candidate_in_slot_a,
    judge_turn,
)
from backend.app.services.llm_eval.types import (
    JudgeVerdict,
    ModelCallResult,
    ReplaySample,
    ToolCall,
)
from backend.app.services.llm_service import LLMTarget


class _Response:
    """Minimal stand-in for ``MessageResponse``.

    The content block must be a real ``TextBlock``: ``get_response_text``
    filters by ``isinstance``, so a duck-typed stub silently reads as an
    empty response and every mapping assertion below would pass for the
    wrong reason.

    ``stop_reason`` carries the same default the providers send on a normal
    completion, so a test that does not care about truncation does not have
    to say so.
    """

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [TextBlock(type="text", text=text)]
        self.stop_reason = stop_reason


class _ToolResponse:
    """A judge that answered through the forced verdict tool."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = [
            ToolUseBlock(type="tool_use", id="toolu_1", name="record_verdict", input=payload)
        ]
        self.stop_reason = "tool_use"


def _judge_reply(**payload: Any) -> AsyncMock:
    return AsyncMock(return_value=_Response(json.dumps(payload)))


BASELINE = ModelCallResult(
    provider="anthropic",
    model="incumbent",
    tool_calls=[ToolCall(name="lookup", arguments={"query": "invoice"})],
)
CANDIDATE = ModelCallResult(provider="anthropic", model="candidate", text="I'll take a look.")

TURN_TEXT = "where is invoice 42?"


def _seq_for_slot(candidate_is_a: bool) -> int:
    """Find a seq whose turn puts the candidate in the requested slot.

    Asks the implementation rather than recomputing the hash, so this cannot
    drift into testing a second copy of the assignment rule.
    """
    for seq in range(1, 500):
        sample = ReplaySample(seq=seq, timestamp="", message_context=TURN_TEXT)
        if candidate_in_slot_a(sample) is candidate_is_a:
            return seq
    raise AssertionError("no seq produced the requested slot")


SEQ_CANDIDATE_IS_A = _seq_for_slot(True)
SEQ_CANDIDATE_IS_B = _seq_for_slot(False)


async def _judge(seq: int, mock: AsyncMock) -> tuple[JudgeVerdict, str]:
    with patch("backend.app.services.llm_eval.judge.amessages", mock):
        return await judge_turn(
            ReplaySample(seq=seq, timestamp="", message_context=TURN_TEXT),
            BASELINE,
            CANDIDATE,
            target=LLMTarget(provider="anthropic", model="incumbent"),
        )


async def test_winner_a_maps_to_candidate_when_candidate_is_a() -> None:
    verdict, rationale = await _judge(
        SEQ_CANDIDATE_IS_A, _judge_reply(winner="A", unsafe="none", rationale="acted")
    )
    assert verdict is JudgeVerdict.CANDIDATE_BETTER
    assert rationale == "acted"


async def test_winner_a_maps_to_incumbent_when_candidate_is_b() -> None:
    verdict, _ = await _judge(
        SEQ_CANDIDATE_IS_B, _judge_reply(winner="A", unsafe="none", rationale="acted")
    )
    assert verdict is JudgeVerdict.CANDIDATE_WORSE


async def test_winner_b_maps_to_candidate_when_candidate_is_b() -> None:
    verdict, _ = await _judge(
        SEQ_CANDIDATE_IS_B, _judge_reply(winner="B", unsafe="none", rationale="acted")
    )
    assert verdict is JudgeVerdict.CANDIDATE_BETTER


async def test_equivalent_passes_through() -> None:
    verdict, _ = await _judge(
        SEQ_CANDIDATE_IS_A, _judge_reply(winner="equivalent", unsafe="none", rationale="same")
    )
    assert verdict is JudgeVerdict.EQUIVALENT


async def test_unsafe_flag_on_the_candidate_slot_blocks() -> None:
    verdict, _ = await _judge(
        SEQ_CANDIDATE_IS_A, _judge_reply(winner="B", unsafe="A", rationale="texts the wrong person")
    )
    assert verdict is JudgeVerdict.CANDIDATE_UNSAFE


async def test_unsafe_flag_on_the_incumbent_does_not_credit_the_candidate() -> None:
    """An unsafe incumbent is worth recording, but it is not evidence to switch."""
    verdict, rationale = await _judge(
        SEQ_CANDIDATE_IS_B, _judge_reply(winner="B", unsafe="A", rationale="bad call")
    )
    assert verdict is JudgeVerdict.EQUIVALENT
    assert "incumbent flagged unsafe" in rationale


async def test_prose_around_the_json_is_tolerated() -> None:
    mock = AsyncMock(
        return_value=_Response(
            'Here is my assessment:\n{"winner": "equivalent", "unsafe": "none", '
            '"rationale": "both fine"}\nHope that helps.'
        )
    )
    verdict, _ = await _judge(SEQ_CANDIDATE_IS_A, mock)
    assert verdict is JudgeVerdict.EQUIVALENT


async def test_unparseable_output_is_recorded_not_raised() -> None:
    mock = AsyncMock(return_value=_Response("I cannot decide."))
    verdict, _ = await _judge(SEQ_CANDIDATE_IS_A, mock)
    assert verdict is JudgeVerdict.JUDGE_FAILED


async def test_provider_failure_is_recorded_not_raised() -> None:
    mock = AsyncMock(side_effect=RuntimeError("gateway down"))
    verdict, rationale = await _judge(SEQ_CANDIDATE_IS_A, mock)
    assert verdict is JudgeVerdict.JUDGE_FAILED
    assert "gateway down" in rationale


def test_description_never_names_the_model() -> None:
    """The judge must not be able to tell which response is the incumbent."""
    rendered = _describe(BASELINE)
    assert "incumbent" not in rendered
    assert "anthropic" not in rendered
    assert "lookup" in rendered


def test_slot_assignment_varies_across_a_realistic_transcript() -> None:
    """Regression: the assignment must not alias to message-seq parity.

    A transcript alternates inbound and outbound rows, so every replayable
    turn has an odd seq. An assignment keyed on ``seq % 2`` is constant for a
    whole run, which silently pins the candidate to one slot and makes the
    blinding decorative.
    """
    slots = {
        candidate_in_slot_a(ReplaySample(seq=seq, timestamp="", message_context=f"turn {seq}"))
        for seq in range(1, 60, 2)
    }
    assert slots == {True, False}


async def test_verdict_arrives_through_the_forced_tool() -> None:
    mock = AsyncMock(
        return_value=_ToolResponse({"winner": "A", "unsafe": "none", "rationale": "acted"})
    )
    verdict, rationale = await _judge(SEQ_CANDIDATE_IS_A, mock)
    assert verdict is JudgeVerdict.CANDIDATE_BETTER
    assert rationale == "acted"
    kwargs = mock.await_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_verdict"}
    assert kwargs["tools"][0]["name"] == "record_verdict"


async def test_judge_has_headroom_to_think_before_answering() -> None:
    """Regression: 1024 tokens ran out before a thinking judge reached its verdict."""
    mock = _judge_reply(winner="equivalent", unsafe="none", rationale="same")
    await _judge(SEQ_CANDIDATE_IS_A, mock)
    assert mock.await_args.kwargs["max_tokens"] >= 4096


async def test_unescaped_quotes_in_the_rationale_do_not_lose_the_verdict() -> None:
    """Regression: judges quote the user without escaping the quotes.

    Strict JSON parsing threw those verdicts away, unsafe flags included.
    """
    raw = '{"winner": "B", "unsafe": "A", "rationale": "A texts "the tenant" instead of the owner"}'
    verdict, rationale = await _judge(SEQ_CANDIDATE_IS_A, AsyncMock(return_value=_Response(raw)))
    assert verdict is JudgeVerdict.CANDIDATE_UNSAFE
    assert "the tenant" in rationale


async def test_endpoint_that_refuses_a_forced_tool_is_asked_for_json() -> None:
    reply = _Response(json.dumps({"winner": "B", "unsafe": "none", "rationale": "ok"}))
    mock = AsyncMock(side_effect=[InvalidRequestError("tool_choice unsupported"), reply])
    verdict, _ = await _judge(SEQ_CANDIDATE_IS_B, mock)
    assert verdict is JudgeVerdict.CANDIDATE_BETTER
    assert "tool_choice" not in mock.await_args_list[1].kwargs
