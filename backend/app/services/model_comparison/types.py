"""Value types for the model comparison report.

The report answers one question for an operator: if this user's agent loop
were pointed at a different model, what would change? It answers it by
replaying the user's own recent turns through the candidate and laying the
candidate's decision next to what production actually did, which is already
in the record.

Nothing here executes a tool. A replay continues through a lookup only when
the live turn made that same lookup, by feeding back the result it recorded,
and stops at the first decision that would need a live tool. That decision is
the thing a model swap actually changes and the only thing that can be shown
without re-running the user's real side effects.

There is no verdict here, and deliberately so. Four rounds of review on the
evaluator this replaced kept finding artifacts in the machinery that turned a
noisy comparison into a recommendation: identical candidates blocked, bad
candidates approved, judge blinding that leaked. The deployment has three
users whose transcripts the operator reads anyway, so the output is evidence
laid out for reading, not a green light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Side(StrEnum):
    """Whose behaviour a finding describes."""

    PRODUCTION = "production"
    """What the live turn actually did, read back from the record."""

    CANDIDATE = "candidate"
    """What the candidate model decided when the turn was replayed."""


class Finding(StrEnum):
    """Something a decision did that production would have acted on badly.

    Deterministic: every one of these is decided by code reading the tool
    schema and the transcript, with no model in the loop. Counted per side
    and never averaged into a score.
    """

    UNKNOWN_TOOL = "unknown_tool"
    """Called a tool name that is in neither the schema nor this turn's record."""

    INVALID_ARGS = "invalid_args"
    """Emitted arguments the tool would reject: its params model or its own precheck."""

    UNREQUESTED_WRITE = "unrequested_write"
    """Made a write the live turn did not make.

    Candidate-only by construction: production's own writes are the standard
    this is measured against, so the production side of this check is
    vacuous rather than clean. See ``PRODUCTION_CHECKED``.
    """

    FABRICATED_ID = "fabricated_id"
    """Wrote to a record ID that appears nowhere in what the model was shown.

    A model that calls a search and a write in one response has to guess the
    ID the search would have returned, and a note filed against the
    neighbouring work order reads as decisive action. Checked against the
    prompt, the user's message and every tool result the model saw, on both
    sides. See ``checks.fabricated_ids``.
    """

    TRUNCATED = "truncated"
    """Hit the output token ceiling even after production's retry.

    Candidate-only: the record holds a delivered reply, not a budget the
    live call ran out of.
    """

    TOOL_NOT_IN_SCHEMA = "tool_not_in_schema"
    """Named a tool this turn's record carries but today's schema does not.

    A property of the fixture, not of a model: the name is in the
    conversation history and either side can read it out. Surfaced so the
    operator knows the replay is describing a tool surface the user no
    longer has, and excluded from the violation counts.
    """

    CALL_FAILED = "call_failed"
    """The provider raised. Recorded per turn rather than failing the run."""


HARD_VIOLATIONS = frozenset(
    {
        Finding.UNKNOWN_TOOL,
        Finding.INVALID_ARGS,
        Finding.UNREQUESTED_WRITE,
        Finding.FABRICATED_ID,
        Finding.TRUNCATED,
    }
)
"""Findings the summary counts as violations, per side.

``TOOL_NOT_IN_SCHEMA`` and ``CALL_FAILED`` are deliberately absent. Neither
is something a model did: the first is a property of the replayed fixture and
the second is a failure to measure. Both are still recorded on the turn.
"""


PRODUCTION_CHECKED = frozenset(
    {
        Finding.INVALID_ARGS,
        Finding.FABRICATED_ID,
        Finding.TOOL_NOT_IN_SCHEMA,
    }
)
"""The checks that also run against the record, so "the incumbent does this
too" is answerable.

The other three cannot be asked of production and must not be reported as a
clean zero for it. ``UNREQUESTED_WRITE`` is measured against production's own
writes, so production passes by definition. ``UNKNOWN_TOOL`` would accuse the
record of calling a tool that existed when it was called. ``TRUNCATED``
describes a live call's token budget, which the record does not carry.

Served to the console on the summary so the report can render those three as
"not applicable" for production rather than as zero.
"""


class WriteOutcome(StrEnum):
    """Whether the candidate reached one write the live turn made."""

    MATCHED = "matched"
    """Same tool, same key arguments. See ``report.key_arguments``."""

    SAME_TOOL_DIFFERENT_ARGS = "same_tool_different_args"
    """Called the tool, but not with the arguments production used.

    Its own bucket rather than a miss because the two readings are very
    different: a rephrased message body lands here, and so does a note filed
    against the wrong job.
    """

    MISSED = "missed"
    """Did not call the tool at all."""


class TurnOutcome(StrEnum):
    """How the candidate's decision relates to what production did for a turn.

    About production's *writes* only. A write the candidate made and
    production did not is an ``UNREQUESTED_WRITE`` finding rather than an
    outcome, so one turn is never described twice.
    """

    NOT_REPLAYED = "not_replayed"
    """The candidate's call errored, so there is no decision to show."""

    NO_WRITE = "no_write"
    """The live turn wrote nothing, so there is no task outcome to check."""

    WRITE_MATCHED = "write_matched"
    WRITE_ARGS_DIFFER = "write_args_differ"
    WRITE_MISSED = "write_missed"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    """Process died mid-run. Set by the sweep, never by the run itself."""
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RecordedToolResult:
    """A tool call and the result it returned.

    On a ``ReplaySample`` these are what the live turn called and got back,
    read from the stored ``tool_interactions_json``. On a ``ModelCallResult``
    they are the lookups a replay fed back to the model before its decision.
    Replaying a recorded result is not execution: nothing is called.
    """

    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool = False


@dataclass(frozen=True)
class ReplaySample:
    """One historic inbound turn, selected for replay.

    ``message_context`` is the persisted ``processed_context`` when present,
    which is the exact string production passed to the agent for this turn
    (body plus media transcription and OCR). Falling back to ``body`` only
    affects rows written before that column was populated.
    """

    seq: int
    timestamp: str
    message_context: str
    production_reply: str = ""
    production_tool_calls: tuple[RecordedToolResult, ...] = ()
    """Every tool call the live turn made, with the result it returned.

    Three jobs at once, which is why it is one list. It is the baseline the
    report shows beside the candidate; it is what lets a replay continue past
    a lookup, by feeding back a recorded result instead of calling anything;
    and it is the standard ``UNREQUESTED_WRITE`` and the write comparison are
    measured against.
    """
    batched_messages: tuple[str, ...] = ()
    """Earlier messages of the batch this turn closes, oldest first.

    Production answers a rapid-fire batch once, at its last row, with the
    earlier rows already in the loaded history. The replay does the same, so
    these are in the prompt as history rather than in ``message_context``;
    they are carried here for the report, which would otherwise show a
    fraction of what the user asked.
    """

    @property
    def user_text(self) -> str:
        """Everything the user sent for this turn, batch included."""
        return "\n\n".join([*self.batched_messages, self.message_context])

    @property
    def production_tool_names(self) -> list[str]:
        return [call.name for call in self.production_tool_calls]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation parsed out of a model's response blocks."""

    name: str
    arguments: dict[str, Any]
    # The provider's ``tool_use`` id, needed to pair a replayed result with
    # its call. Not part of what the call *is*, so equality ignores it.
    id: str = field(default="", compare=False)


@dataclass
class ModelCallResult:
    """What the candidate returned for one replayed turn."""

    provider: str
    model: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""
    replayed_lookups: list[RecordedToolResult] = field(default_factory=list)
    """Lookups fed back from the live turn before this decision, in order.

    The decision shown (``text``, ``tool_calls``) is what the model did after
    them, and the usage above covers every round. Empty when the model
    decided on its first round.
    """
    truncation_retries: int = 0
    """Times this decision was re-asked at a larger budget after truncating.

    Production retries a reply cut off at ``max_tokens`` with no tool call,
    so the replay does too, and the usage above includes the spent attempts.
    """

    @property
    def acted(self) -> bool:
        """Whether the model chose to call at least one tool."""
        return bool(self.tool_calls)


@dataclass
class Issue:
    """A single finding with enough detail to act on it."""

    finding: Finding
    tool_name: str = ""
    detail: str = ""
    side: Side = Side.CANDIDATE

    @property
    def is_violation(self) -> bool:
        return self.finding in HARD_VIOLATIONS


@dataclass(frozen=True)
class WriteComparison:
    """One write the live turn made, and what the candidate did about it."""

    tool_name: str
    outcome: WriteOutcome
    key_arguments: dict[str, Any]
    """The arguments compared, as production spelled them. See
    ``report.key_arguments`` for what counts as key."""
    candidate_arguments: dict[str, Any] | None = None
    """What the candidate passed to the same tool, when it called it."""


@dataclass
class TurnReport:
    """Everything recorded for one replayed turn."""

    sample: ReplaySample
    candidate: ModelCallResult
    outcome: TurnOutcome
    issues: list[Issue] = field(default_factory=list)
    writes: list[WriteComparison] = field(default_factory=list)

    def violations(self, side: Side) -> int:
        return sum(1 for issue in self.issues if issue.is_violation and issue.side is side)
