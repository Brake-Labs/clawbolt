"""The two walks of ``replayable_lookup``, driven over one record.

``execution.call_model`` walks the candidate's rounds; ``sampling.
_historic_first_decision`` walks the recorded calls of the same turn. Both
apply ``metrics.replayable_lookup`` under ``MAX_REPLAY_READ_ROUNDS``, and the
whole historic mode rests on them landing on the same decision: the bug it
shipped with scored the candidate on the write it made after a lookup and the
incumbent on the lookup itself, and an identical candidate came back
``do_not_switch`` at a 100% silent-no-op rate.

Nothing tested that against the real code paths. Each side had its own tests,
each with its own idea of what the other does, which is how they drifted the
first time.

The two paths are not identical, and cannot be. ``call_model`` spends the
budget in *rounds* and sees a response's calls together; the flat
``tool_interactions_json`` record has no round boundaries, so the historic
walk spends it in *calls*. On a turn where production asked for several
things at once the two land on different decisions. Those cases are listed
here as known divergences with their exact outcomes pinned, so a change to
either path fails here rather than quietly widening the gap.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock, ToolUseBlock
from pydantic import BaseModel

from backend.app.agent.core import AssembledPrompt
from backend.app.agent.dto import StoredMessage
from backend.app.agent.messages import SystemMessage, UserMessage
from backend.app.agent.tools.base import Tool, ToolResult, ToolTags
from backend.app.services.llm_eval.execution import call_model
from backend.app.services.llm_eval.sampling import _historic_first_decision
from backend.app.services.llm_eval.types import ModelCallResult, RecordedToolResult
from backend.app.services.llm_service import LLMTarget

BASE_TIME = _dt.datetime(2026, 5, 1, 12, 0, tzinfo=_dt.UTC)
TURN_TEXT = "invoice Acme for the stalls"
TARGET = LLMTarget(provider="anthropic", model="m")


class _FindParams(BaseModel):
    q: str


class _SendParams(BaseModel):
    invoice_id: str


async def _never(**_kwargs: object) -> ToolResult:  # pragma: no cover - never invoked
    raise AssertionError("a replay must never execute a tool")


TOOLS = {
    "qb_find": Tool(
        name="qb_find",
        description="find",
        function=_never,
        params_model=_FindParams,
        tags={ToolTags.READ_ONLY},
    ),
    # Untagged, so mutating: never fed back, whichever side asked for it.
    "qb_send": Tool(name="qb_send", description="send", function=_never, params_model=_SendParams),
}

_Call = tuple[str, dict[str, Any]]


def _find(which: str) -> _Call:
    return ("qb_find", {"q": which})


def _send(invoice: str) -> _Call:
    return ("qb_send", {"invoice_id": invoice})


@dataclass(frozen=True)
class _Turn:
    """One turn, described once and fed to both paths.

    *rounds* is what the candidate emits, one entry per model response. The
    record the historic side reads is those same calls flattened in order,
    which is exactly what production writes to ``tool_interactions_json``.
    """

    name: str
    rounds: tuple[tuple[_Call, ...], ...]
    reply: str = "sent it"
    diverges: str = ""
    """Why the two paths cannot land on the same decision here. Empty when
    they must, which is the case this file exists to hold."""

    expected_candidate: tuple[_Call, ...] = ()
    """Pinned only for a divergent turn, where "they differ" is too weak an
    assertion to catch a change in either walk."""

    expected_historic: tuple[_Call, ...] = ()

    @property
    def recorded_calls(self) -> list[_Call]:
        return [call for round_calls in self.rounds for call in round_calls]


TURNS = [
    _Turn(name="a write with nothing before it", rounds=((_send("1186"),),)),
    _Turn(name="a lookup then a write", rounds=((_find("Acme"),), (_send("1186"),))),
    _Turn(
        name="two lookups then a write",
        rounds=((_find("Acme"),), (_find("Acme stalls"),), (_send("1186"),)),
    ),
    _Turn(
        name="lookups until the round budget runs out",
        rounds=tuple((_find(f"customer {i}"),) for i in range(4)),
    ),
    _Turn(name="a turn that only replied", rounds=((),), reply="All good."),
    _Turn(
        name="a lookup then a reply",
        rounds=((_find("Acme"),), ()),
        reply="They owe 1186.",
    ),
    _Turn(
        name="a lookup and a write in one response",
        rounds=((_find("Acme"), _send("1186")),),
        diverges=(
            "the round holds a write, so the replay cannot answer any of it and scores "
            "the whole response; the record has no round boundary, so the walk skips the "
            "lookup and scores the write alone"
        ),
        expected_candidate=(_find("Acme"), _send("1186")),
        expected_historic=(_send("1186"),),
    ),
    _Turn(
        name="four lookups, two to a round, then a write",
        rounds=((_find("a"), _find("b")), (_find("c"), _find("d")), (_send("1186"),)),
        diverges=(
            "the replay spends two of its three rounds on four lookups and reaches the "
            "write; the walk spends one of its three on each lookup, runs out, and scores "
            "the fourth lookup"
        ),
        expected_candidate=(_send("1186"),),
        expected_historic=(_find("d"),),
    ),
]


def _prompt() -> AssembledPrompt:
    return AssembledPrompt(
        messages=[SystemMessage(content="system"), UserMessage(content=TURN_TEXT)],
        stable_system="system",
        dynamic_context="",
        system_prompt="system",
    )


def _response(calls: tuple[_Call, ...], text: str) -> MessageResponse:
    content: list[Any] = []
    if text:
        content.append(TextBlock(type="text", text=text))
    content.extend(
        ToolUseBlock(type="tool_use", id=f"toolu_{index}", name=name, input=args)
        for index, (name, args) in enumerate(calls)
    )
    return MessageResponse(
        id="msg_1",
        content=content,
        model="m",
        role="assistant",
        stop_reason="tool_use" if calls else "end_turn",
        type="message",
        usage=MessageUsage(input_tokens=100, output_tokens=50),
    )


def _recorded(turn: _Turn) -> tuple[RecordedToolResult, ...]:
    """The tool results the live turn wrote down, in order."""
    return tuple(
        RecordedToolResult(name=name, arguments=args, result=f"{name} ok")
        for name, args in turn.recorded_calls
    )


def _rows(turn: _Turn) -> list[StoredMessage]:
    """The transcript rows production would have persisted for this turn."""
    interactions = json.dumps(
        [
            {"tool_call_id": f"t{index}", "name": name, "args": args, "result": f"{name} ok"}
            for index, (name, args) in enumerate(turn.recorded_calls)
        ]
    )
    return [
        StoredMessage(
            seq=1,
            direction="inbound",
            body=TURN_TEXT,
            processed_context=TURN_TEXT,
            timestamp=BASE_TIME.isoformat(),
        ),
        StoredMessage(
            seq=2,
            direction="outbound",
            body=turn.reply,
            llm_reply_text=turn.reply,
            tool_interactions_json=interactions if turn.recorded_calls else "",
            timestamp=(BASE_TIME + _dt.timedelta(seconds=9)).isoformat(),
        ),
    ]


async def _candidate_decision(turn: _Turn) -> ModelCallResult:
    """Drive the real ``call_model`` over this turn's rounds."""
    # Prose rides on the round that carries no call, which is how a turn that
    # ends in an answer looks. A round with calls says nothing, so the two
    # sides' prose is comparable where either has any.
    responses = [_response(calls, "" if calls else turn.reply) for calls in turn.rounds]
    mock = AsyncMock(side_effect=responses)
    with patch("backend.app.services.llm_eval.execution.amessages", mock):
        return await call_model(
            _prompt(),
            None,
            target=TARGET,
            reasoning_effort="",
            tools_by_name=TOOLS,
            recorded=_recorded(turn),
        )


def _pairs(calls: Any) -> list[_Call]:
    return [(call.name, dict(call.arguments)) for call in calls]


@dataclass
class _Decisions:
    candidate_calls: list[_Call] = field(default_factory=list)
    candidate_lookups: list[_Call] = field(default_factory=list)
    candidate_text: str = ""
    historic_calls: list[_Call] = field(default_factory=list)
    historic_lookups: list[_Call] = field(default_factory=list)
    historic_text: str = ""


async def _both(turn: _Turn) -> _Decisions:
    candidate = await _candidate_decision(turn)
    historic = _historic_first_decision(_rows(turn), 0, TOOLS)
    assert historic.available, turn.name
    return _Decisions(
        candidate_calls=_pairs(candidate.tool_calls),
        candidate_lookups=_pairs(candidate.replayed_lookups),
        candidate_text=candidate.text,
        historic_calls=_pairs(historic.calls),
        historic_lookups=_pairs(historic.lookups),
        historic_text=historic.text,
    )


async def test_both_paths_score_the_same_decision_on_the_same_record() -> None:
    """The invariant the historic mode rests on, against the real code.

    For every turn shape that has no batched round, the candidate's replay
    and the walk of the record land on the same call, having skipped the same
    lookups to get there. A turn that ended in prose lands on the same prose.
    """
    for turn in TURNS:
        if turn.diverges:
            continue
        decisions = await _both(turn)
        assert decisions.candidate_calls == decisions.historic_calls, turn.name
        assert decisions.candidate_lookups == decisions.historic_lookups, turn.name
        if not decisions.candidate_calls:
            assert decisions.candidate_text == decisions.historic_text, turn.name


async def test_the_batched_round_divergences_are_the_only_ones_and_have_not_moved() -> None:
    """A flat record cannot give back what production asked for at once.

    Both of these are real: a lookup and a write in one response, and four
    lookups batched two to a round. They are pinned rather than fixed because
    the record has no round boundaries to recover, and named in
    ``metrics.replayable_lookup`` and AGENTS.md so the invariant is not read
    as unconditional. A change to either walk that alters them lands here.
    """
    for turn in TURNS:
        if not turn.diverges:
            continue
        decisions = await _both(turn)
        assert decisions.candidate_calls != decisions.historic_calls, turn.name
        assert decisions.candidate_calls == list(turn.expected_candidate), turn.name
        assert decisions.historic_calls == list(turn.expected_historic), turn.name
