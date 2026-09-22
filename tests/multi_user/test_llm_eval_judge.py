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

from backend.app.agent.core import AssembledPrompt
from backend.app.agent.messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolCallRequest,
    ToolResultMessage,
    UserMessage,
)
from backend.app.services.llm_eval.judge import (
    _SYSTEM_PROMPT,
    JudgeOutcome,
    _describe,
    _judge_prompt,
    build_judge_context,
    candidate_in_slot_a,
    judge_turn,
)
from backend.app.services.llm_eval.types import (
    JudgeVerdict,
    ModelCallResult,
    RecordedToolResult,
    ReplaySample,
    Side,
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


async def _outcome(seq: int, mock: AsyncMock) -> JudgeOutcome:
    with patch("backend.app.services.llm_eval.judge.amessages", mock):
        return await judge_turn(
            ReplaySample(seq=seq, timestamp="", message_context=TURN_TEXT),
            BASELINE,
            CANDIDATE,
            target=LLMTarget(provider="anthropic", model="incumbent"),
        )


async def _judge(seq: int, mock: AsyncMock) -> tuple[JudgeVerdict, str]:
    outcome = await _outcome(seq, mock)
    return outcome.verdict, outcome.rationale


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


async def test_unsafe_flag_on_the_candidate_slot_is_a_candidate_finding() -> None:
    outcome = await _outcome(
        SEQ_CANDIDATE_IS_A, _judge_reply(winner="B", unsafe="A", rationale="texts the wrong person")
    )
    assert outcome.unsafe == {Side.CANDIDATE}
    # The preference is kept alongside the flag, not replaced by it.
    assert outcome.verdict is JudgeVerdict.CANDIDATE_WORSE


async def test_unsafe_flag_on_the_incumbent_is_recorded_against_the_incumbent() -> None:
    """Regression: an unsafe incumbent was folded into an "equivalent" verdict.

    The flag was lost to the run, so the comparison could never see that the
    incumbent does the unsafe thing too, and the candidate's actual win on
    the turn was thrown away with it.
    """
    outcome = await _outcome(
        SEQ_CANDIDATE_IS_B, _judge_reply(winner="B", unsafe="A", rationale="bad call")
    )
    assert outcome.unsafe == {Side.BASELINE}
    assert outcome.verdict is JudgeVerdict.CANDIDATE_BETTER


async def test_both_unsafe_flags_both_sides() -> None:
    outcome = await _outcome(
        SEQ_CANDIDATE_IS_A, _judge_reply(winner="equivalent", unsafe="both", rationale="both")
    )
    assert outcome.unsafe == {Side.BASELINE, Side.CANDIDATE}


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
    kwargs = mock.await_args_list[-1].kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_verdict"}
    assert kwargs["tools"][0]["name"] == "record_verdict"


async def test_judge_has_headroom_to_think_before_answering() -> None:
    """Regression: 1024 tokens ran out before a thinking judge reached its verdict."""
    mock = _judge_reply(winner="equivalent", unsafe="none", rationale="same")
    await _judge(SEQ_CANDIDATE_IS_A, mock)
    assert mock.await_args_list[-1].kwargs["max_tokens"] >= 4096


async def test_unescaped_quotes_in_the_rationale_do_not_lose_the_verdict() -> None:
    """Regression: judges quote the user without escaping the quotes.

    Strict JSON parsing threw those verdicts away, unsafe flags included.
    """
    raw = '{"winner": "B", "unsafe": "A", "rationale": "A texts "the tenant" instead of the owner"}'
    outcome = await _outcome(SEQ_CANDIDATE_IS_A, AsyncMock(return_value=_Response(raw)))
    assert outcome.unsafe == {Side.CANDIDATE}
    assert outcome.verdict is JudgeVerdict.CANDIDATE_WORSE
    assert "the tenant" in outcome.rationale


async def test_endpoint_that_refuses_a_forced_tool_is_asked_for_json() -> None:
    reply = _Response(json.dumps({"winner": "B", "unsafe": "none", "rationale": "ok"}))
    mock = AsyncMock(side_effect=[InvalidRequestError("tool_choice unsupported"), reply])
    verdict, _ = await _judge(SEQ_CANDIDATE_IS_B, mock)
    assert verdict is JudgeVerdict.CANDIDATE_BETTER
    assert "tool_choice" not in mock.await_args_list[1].kwargs


# ---------------------------------------------------------------------------
# What the judge is shown
# ---------------------------------------------------------------------------


def _assembled(*history: AgentMessage) -> AssembledPrompt:
    return AssembledPrompt(
        messages=[SystemMessage(content="system rules"), *history, UserMessage(content="now")],
        stable_system="system rules",
        dynamic_context="",
        system_prompt="system rules",
    )


async def test_the_judge_sees_history_tool_results_and_the_clock() -> None:
    """Regression: the judge saw the user message and nothing else.

    It could not check an ID against the lookup it came from, called facts
    from earlier turns made up, and had no date to resolve "tomorrow" by.
    """
    context = build_judge_context(
        _assembled(
            UserMessage(content="the unit at 12 Oak St needs a note"),
            AssistantMessage(
                content=None,
                tool_calls=[ToolCallRequest(id="t1", name="search", arguments={"q": "12 Oak"})],
            ),
            ToolResultMessage(tool_call_id="t1", content="work order 71002"),
            AssistantMessage(content="Found work order 71002."),
        ),
        "[Current time: Friday, 2026-05-01 12:00 PM (UTC)]",
    )
    mock = _judge_reply(winner="equivalent", unsafe="none", rationale="same")
    with patch("backend.app.services.llm_eval.judge.amessages", mock):
        await judge_turn(
            ReplaySample(
                seq=SEQ_CANDIDATE_IS_A,
                timestamp="",
                message_context=TURN_TEXT,
                historic_tool_names=["search", "add_note"],
            ),
            BASELINE,
            CANDIDATE,
            target=LLMTarget(provider="anthropic", model="incumbent"),
            context=context,
        )

    prompt = mock.await_args_list[-1].kwargs["messages"][0]["content"]
    assert "2026-05-01" in prompt
    assert "the unit at 12 Oak St needs a note" in prompt
    assert "Tool result: work order 71002" in prompt
    assert "search, add_note" in prompt
    assert "system rules" not in prompt


def test_the_rubric_does_not_penalize_looking_up_before_writing() -> None:
    """Regression: "did it take the action" scored a correct lookup as a failure."""
    assert "before writing is a correct step" in _SYSTEM_PROMPT


def test_a_response_shows_the_lookups_it_made_first() -> None:
    call = ModelCallResult(
        provider="anthropic",
        model="m",
        tool_calls=[ToolCall(name="add_note", arguments={"work_order_id": "71002"})],
        replayed_lookups=[
            RecordedToolResult(name="search", arguments={"q": "12 Oak"}, result="work order 71002")
        ],
    )
    rendered = _describe(call)
    assert "Lookups made first" in rendered
    assert "work order 71002" in rendered


def test_the_transcript_keeps_the_newest_messages_within_budget() -> None:
    old = [UserMessage(content=f"old {i} " + "x" * 1400) for i in range(40)]
    context = build_judge_context(_assembled(*old, UserMessage(content="the latest ask")), "")
    assert "the latest ask" in context.transcript
    assert "old 0 " not in context.transcript
    assert context.transcript.startswith("[earlier conversation omitted]")
    assert len(context.transcript) < 30000


# ---------------------------------------------------------------------------
# A decision read out of the transcript must not be identifiable
# ---------------------------------------------------------------------------
#
# The blinding is what stops the judge flattering the model it is: the
# default judge is the incumbent itself. In historic mode three things used
# to mark the incumbent's side on sight, none of them about its content.

_BLOCK_LABELS = ("Lookups made first", "Tool calls", "Reply text")


def _blocks(section: str) -> list[str]:
    """Which labelled blocks a rendered response carries, in order."""
    return [
        label
        for label in _BLOCK_LABELS
        if any(line.startswith(label) for line in section.splitlines())
    ]


def _responses(prompt: str) -> tuple[str, str]:
    after_a = prompt.split("--- Response A ---")[1]
    first, second = after_a.split("--- Response B ---")
    return first, second


def test_a_decision_with_no_prose_renders_no_reply_line() -> None:
    """Regression: an empty reply printed ``(empty)``.

    A decision read out of the transcript never carries prose alongside a
    tool call, because the row holds the turn's final reply and its calls in
    one flat list. Printing a placeholder made that side say ``(empty)`` on
    every acting turn while the candidate's said whatever it said.
    """
    rendered = _describe(BASELINE)
    assert "(empty)" not in rendered
    assert "Reply text" not in rendered
    assert _blocks(rendered) == ["Tool calls"]


def test_a_historic_turn_shows_two_structurally_identical_responses() -> None:
    """Same blocks, in the same order, for a side that was read and one that ran."""
    lookup = RecordedToolResult(
        name="qb_find", arguments={"q": "Acme Plumbing"}, result="invoice 1186"
    )
    historic = ModelCallResult(
        provider="anthropic",
        model="incumbent",
        tool_calls=[ToolCall(name="qb_send", arguments={"invoice_id": "1186"})],
        replayed_lookups=[lookup],
    )
    candidate = ModelCallResult(
        provider="anthropic",
        model="candidate",
        tool_calls=[ToolCall(name="qb_send", arguments={"invoice_id": "1187"})],
        replayed_lookups=[lookup],
        # The candidate routinely says something alongside its call. A
        # decision read out of the transcript never can: the outbound row
        # holds the turn's final reply and all of its calls in one flat
        # list, so a decision that opened with a call carries no prose of
        # its own. Both blocks together would mark the candidate on every
        # acting turn, and the judge defaults to the incumbent model.
        text="Sending that one now.",
    )
    sample = ReplaySample(
        seq=SEQ_CANDIDATE_IS_A,
        timestamp="",
        message_context=TURN_TEXT,
        historic_tool_names=["qb_find", "qb_send"],
    )
    prompt = _judge_prompt(sample, candidate, historic, None, historic_side_shown=True)
    first, second = _responses(prompt)
    assert _blocks(first) == _blocks(second) == ["Lookups made first", "Tool calls"]
    assert "Sending that one now." not in prompt
    # And the live turn's own calls are withheld: the historic side is
    # literally the head of that list, so printing it names which is which.
    assert "The live assistant's tool calls" not in prompt
    assert "incumbent" not in prompt


def test_a_decision_that_only_replied_keeps_its_prose_in_historic_mode() -> None:
    """Dropping it there would leave a response with nothing in it.

    A turn both sides answered in prose has no tool call to identify either
    of them, and the prose is the whole of what the judge is being asked to
    compare.
    """
    historic = ModelCallResult(
        provider="anthropic", model="incumbent", text="They owe 1186 on that job."
    )
    candidate = ModelCallResult(
        provider="anthropic", model="candidate", text="The balance is 1186."
    )
    sample = ReplaySample(seq=SEQ_CANDIDATE_IS_A, timestamp="", message_context=TURN_TEXT)
    prompt = _judge_prompt(sample, candidate, historic, None, historic_side_shown=True)
    first, second = _responses(prompt)
    assert _blocks(first) == _blocks(second) == ["Tool calls", "Reply text"]
    assert "The balance is 1186." in prompt


def test_a_replayed_turn_keeps_the_prose_beside_the_call() -> None:
    """Both sides were elicited, so neither is marked by carrying prose."""
    candidate = ModelCallResult(
        provider="anthropic",
        model="candidate",
        tool_calls=[ToolCall(name="qb_send", arguments={"invoice_id": "1187"})],
        text="Sending that one now.",
    )
    sample = ReplaySample(seq=SEQ_CANDIDATE_IS_A, timestamp="", message_context=TURN_TEXT)
    prompt = _judge_prompt(sample, candidate, BASELINE, None)
    assert "Sending that one now." in prompt


def test_a_replayed_turn_still_gets_the_live_turns_calls_as_context() -> None:
    """Both sides ran, so the list identifies neither and is worth having."""
    sample = ReplaySample(
        seq=SEQ_CANDIDATE_IS_A,
        timestamp="",
        message_context=TURN_TEXT,
        historic_tool_names=["qb_find", "qb_send"],
    )
    prompt = _judge_prompt(sample, CANDIDATE, BASELINE, None)
    assert "The live assistant's tool calls" in prompt


def test_a_multi_call_candidate_response_cannot_be_blinded() -> None:
    """The tell that rendering cannot close, pinned where the rendering lives.

    ``sampling._historic_first_decision`` returns exactly one call, so a
    response listing two is necessarily the candidate's. Nothing in
    ``_describe`` can hide that: the record has no second call to show, and
    trimming the candidate's would score a decision it did not make. This
    asserts the tell is real, which is why
    ``runner._judge_skip_reason`` withholds the whole turn instead
    (``JudgeSkipReason.UNBLINDABLE_SHAPE``); a change that made the two
    shapes match would land here first.
    """
    historic = ModelCallResult(
        provider="anthropic",
        model="incumbent",
        tool_calls=[ToolCall(name="qb_send", arguments={"invoice_id": "1186"})],
    )
    candidate = ModelCallResult(
        provider="anthropic",
        model="candidate",
        tool_calls=[
            ToolCall(name="qb_find", arguments={"q": "Acme Plumbing"}),
            ToolCall(name="qb_send", arguments={"invoice_id": "1187"}),
        ],
    )
    sample = ReplaySample(seq=SEQ_CANDIDATE_IS_A, timestamp="", message_context=TURN_TEXT)
    prompt = _judge_prompt(sample, candidate, historic, None, historic_side_shown=True)
    first, second = _responses(prompt)
    # Same labelled blocks, so the blinding that can be done is done.
    assert _blocks(first) == _blocks(second) == ["Tool calls"]
    # And the candidate is still the only side that can list two calls.
    assert first.count("\n- ") == 2
    assert second.count("\n- ") == 1


def test_a_single_call_candidate_is_indistinguishable_after_the_withholding() -> None:
    """What is left once the batched turns are withheld: one call each side."""
    candidate = ModelCallResult(
        provider="anthropic",
        model="candidate",
        tool_calls=[ToolCall(name="qb_send", arguments={"invoice_id": "1187"})],
        text="Sending that one now.",
    )
    historic = ModelCallResult(
        provider="anthropic",
        model="incumbent",
        tool_calls=[ToolCall(name="qb_send", arguments={"invoice_id": "1186"})],
    )
    sample = ReplaySample(seq=SEQ_CANDIDATE_IS_A, timestamp="", message_context=TURN_TEXT)
    prompt = _judge_prompt(sample, candidate, historic, None, historic_side_shown=True)
    first, second = _responses(prompt)
    assert _blocks(first) == _blocks(second) == ["Tool calls"]
    assert first.count("\n- ") == second.count("\n- ") == 1
