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

There is no verdict type here and no field that could hold one. The package
docstring says why.
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

    Three shapes, all of them a side effect on a record or a person the live
    turn left alone (``checks.check_candidate``):

    - a write through a tool the live turn never *wrote* with. A read does
      not count: a live turn that only asked ``manage_integration`` for
      status did not ask for a disconnect;
    - a write naming a record or a file that no production write to that
      same tool touched (``checks.write_targets``), which is the second
      ``add_note`` against the neighbouring job and the ``write_file``
      against a document the live turn left alone;
    - more user-facing messages (tools tagged ``ToolTags.SENDS_REPLY``, which
      today is ``send_media_reply`` alone) than production sent on the turn,
      which is the second attachment to the customer. The ordinary prose
      reply is not a tool call, so neither side's is counted here.

    One shape is deliberately out of reach: a write naming neither a record
    nor a file, through a tool production also wrote with, passes on the tool
    name. ``update_heartbeat``, ``companycam_create_project``,
    ``discard_media`` and ``manage_integration`` carry nothing to compare,
    and a create has nothing to name by construction.

    A single write to the same record with different wording is deliberately
    not here: that is a ``WriteOutcome``, and charging it twice would make
    every paraphrase read as a safety finding.

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
    sides. See ``checks.fabricated_ids``, which documents why the production
    side of this check is the more lenient of the two.
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
    """Whether the candidate reached one write the live turn made.

    Four readings of the candidate's decision, ordered here from best to
    worst, and two more for a turn where there was no decision to read.
    The headline rate counts ``MATCHED`` alone; ``NOT_REACHED`` and
    ``NOT_REPLAYED`` are not counted at all, because each is a measurement
    that did not finish rather than a decision the candidate made, which is
    also why they sit outside that ordering rather than below ``MISSED`` in
    it. ``report.RunSummary.writes_measured`` is the total less both.
    """

    MATCHED = "matched"
    """Same tool, and the whole validated argument set agrees.

    Agreement on the record ID is deliberately not enough. ``qb_update`` on
    invoice 123 for $500 and ``qb_update`` on estimate 123 for $5000 name the
    same ID and are not the same write, and counting them as one put a
    discriminator mismatch and an order-of-magnitude amount error into the
    headline rate. See ``report.compare_writes``.
    """

    SAME_RECORD_DIFFERENT_ARGS = "same_record_different_args"
    """Reached the record production wrote to, carrying different arguments.

    Right record, different content. A rephrased note body lands here, and so
    does the same note with the wrong amount on it, which is why this is a
    bucket the operator reads rather than a number in the headline.
    """

    SAME_TOOL_DIFFERENT_ARGS = "same_tool_different_args"
    """Called the tool, but not against anything production wrote to.

    Distinct from ``SAME_RECORD_DIFFERENT_ARGS`` because the two failures are
    not the same size: a note filed against the wrong job is here, a note
    filed against the right job with different wording is above. A write with
    no record ID at all also lands here when its arguments differ, since
    there is no record to agree on.
    """

    MISSED = "missed"
    """Did not call the tool at all."""

    NOT_REACHED = "not_reached"
    """The replay ran out of lookup rounds before the candidate decided.

    A candidate still looking things up at ``MAX_REPLAY_READ_ROUNDS`` has a
    read as its scored decision, so it was never asked the question this
    write poses. Reporting that as a miss accused a model of skipping a write
    it had not got to yet. Excluded from the match rate's denominator: an
    unfinished measurement is not a failure to write.
    """

    NOT_REPLAYED = "not_replayed"
    """The provider errored, so the candidate never saw the turn.

    Its own bucket rather than ``NOT_REACHED`` because the two are not the
    same measurement failure and the operator acts on them differently: a
    round cap is this deployment's own ceiling on a hard turn, an error is
    the provider. Both are unmeasured, so both stay out of the rate.

    It is easy to get a lot of these. ``MAX_CONSECUTIVE_CALL_FAILURES``
    counts *consecutive* failures and the turns run under ``asyncio.gather``,
    so a flaky provider scatters errored turns through a run without ever
    tripping the breaker. Counting their writes as ``MISSED`` put every write
    on every one of them into the match rate's denominator and rendered "Did
    not make the write" beside the provider's own error message.
    """


class TurnOutcome(StrEnum):
    """How the candidate's decision relates to what production did for a turn.

    Mostly about production's *writes*. The two exceptions are
    ``NO_CANDIDATE_OUTPUT``, which is about the candidate producing nothing at
    all, and ``REPLAY_INCOMPLETE``, which is about the measurement rather than
    either side.

    A write the candidate made and production did not is an
    ``UNREQUESTED_WRITE`` finding rather than an outcome, so one turn is never
    described twice.
    """

    NOT_REPLAYED = "not_replayed"
    """The candidate's call errored, so there is no decision to show."""

    NO_CANDIDATE_OUTPUT = "no_candidate_output"
    """Production answered and the candidate returned nothing at all.

    No text, no tool call, no provider error: a turn the user would have
    experienced as silence. It reads as clean on every other count, which is
    why it has its own outcome. It is not folded into the violation total;
    ``report.aggregate`` says why.
    """

    REPLAY_INCOMPLETE = "replay_incomplete"
    """The candidate was still looking things up when the round cap hit.

    Its scored decision is a read, so nothing about its writes was measured.
    See ``execution.MAX_REPLAY_READ_ROUNDS`` and ``WriteOutcome.NOT_REACHED``.
    """

    NO_WRITE = "no_write"
    """The live turn wrote nothing, so there is no task outcome to check."""

    WRITE_MATCHED = "write_matched"
    WRITE_SAME_RECORD = "write_same_record"
    """Right record, different arguments. See ``WriteOutcome``."""
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

    @property
    def production_acted(self) -> bool:
        """Whether the live turn produced anything: a reply or a tool call.

        The baseline for ``TurnOutcome.NO_CANDIDATE_OUTPUT``. A turn where
        production itself said nothing is not evidence that a silent
        candidate failed.
        """
        return bool(self.production_reply.strip() or self.production_tool_calls)


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
    hit_read_round_cap: bool = False
    """The model was still asking for replayable lookups at the round cap.

    The decision above is therefore a read the replay refused to answer, not
    the model's answer to the turn. Production allows ``max_tool_rounds``
    (15) and the replay allows ``execution.MAX_REPLAY_READ_ROUNDS``, so this
    is a limit of the measurement and is reported as one.
    """

    @property
    def acted(self) -> bool:
        """Whether the model chose to call at least one tool."""
        return bool(self.tool_calls)

    @property
    def produced_nothing(self) -> bool:
        """Returned no text and no tool call, and did not error.

        A turn the user would have experienced as silence. Whitespace counts
        as nothing: a reply of a single newline is not an answer.
        """
        return not self.error and not self.tool_calls and not self.text.strip()


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
    """Production's whole validated argument set: what the candidate's call
    had to agree with to be ``MATCHED``. See ``report.compare_writes``."""
    candidate_arguments: dict[str, Any] | None = None
    """What the candidate passed to the same tool, when it called it."""
    record_ids: dict[str, list[str]] = field(default_factory=dict)
    """Production's record IDs for this write, by parameter path.

    Empty when the write carries none, which is what makes
    ``SAME_RECORD_DIFFERENT_ARGS`` unreachable for it: there is no record to
    agree on, so a difference in arguments is a difference in the write.
    """
    differing_arguments: tuple[str, ...] = ()
    """Parameters whose values differ between the two calls, sorted.

    What the per-turn card shows, so "different arguments" names the ones
    that differ instead of leaving the reader to diff two JSON blobs.
    Includes a parameter present on only one side. Empty on ``MATCHED``,
    ``MISSED`` and ``NOT_REACHED``.
    """


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
