"""Value types for the model-swap evaluator.

The evaluator answers one question: if this user's agent loop were pointed at a
different model, would it still do the right thing? It answers it by replaying
the user's own recent turns through both the incumbent and the candidate model
and comparing the two decisions.

Nothing here executes a tool. A replay continues through a lookup only when
the live turn made that same lookup, by feeding back the result it recorded,
and stops at the first decision that would need a live tool. That decision is
the thing a model swap actually changes and the only thing that can be
compared without re-running the user's real side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from backend.app.services.llm_service import LLMTarget


class Side(StrEnum):
    """Which model in a comparison a finding belongs to."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"


class IncumbentSource(StrEnum):
    """Where a run gets the incumbent's decision for each turn.

    Chosen when the run is started and frozen onto the row. Only the
    incumbent side is affected: the candidate is the thing being evaluated,
    so it is always called live.
    """

    HISTORIC = "historic"
    """Do not call the incumbent; read its decision out of the transcript.

    This docstring is where the mode is explained. Everything else that
    touches it (the migration, the models, the schemas, ``runner``,
    ``metrics``, the admin console, AGENTS.md) points here rather than
    restating it.

    The default, and what halves a run's provider bill: the incumbent's
    answer to a turn was already bought once, when the turn happened.

    **Scored at the same point in the turn as the candidate.** The replay
    advances the candidate past responses whose calls are all replayable
    lookups and scores the first one that would need a live tool, so
    ``sampling`` walks the recorded calls by that same rule
    (``metrics.replayable_lookup``, bounded by
    ``metrics.MAX_REPLAY_READ_ROUNDS``): leading lookups are skipped and
    the next recorded call is the incumbent's decision. Any other reading
    compares two different rounds of the same turn.

    Four things it cannot control, reported rather than papered over:

    - That turn ran under the system prompt and tool schema of the day, not
      today's.
    - It may have run on a different model, endpoint or reasoning effort
      than the run's stated incumbent. ``runner.historic_configuration_drift``
      reads ``llm_usage_logs`` over the sampled window and says so.
    - A turn with no reconstructable decision is ``TurnSource.UNAVAILABLE``
      and leaves every paired comparison, rather than reading as "the agent
      did nothing", which is the reading that exempts a candidate's write
      from ``UNREQUESTED_MUTATION``.
    - The incumbent made no call, so its tokens, latency, cost and safety
      record were never measured. They are reported as unavailable, not as
      zero (``SideComparison.comparable``, ``pricing_unknown_reason``), and
      the safety comparison therefore reports rather than decides: no run in
      this mode returns ``safe_to_switch`` or blocks on the safety test.

    **It can still reject a candidate.** The silent-no-op and
    judge-preference tiers compare two decisions that were really made, at
    the same point in the turn, so they block here as they do on a replayed
    run. What withholds them is the sample rather than the mode: above
    ``metrics.MAX_CONFOUNDED_TURN_RATE`` of unreadable or flattened turns the
    finding is recorded on ``RunAggregate.blocking_withheld`` and the run is
    ``inconclusive``, never ``switch_with_monitoring``.
    """

    REPLAY = "replay"
    """Call the incumbent live on every turn, doubling the run's cost.

    Worth the money in three cases: the deployment has never run the
    incumbent on these turns under today's prompt, so there is nothing
    recorded to read; the run is a calibration of a model against itself,
    which needs two live samples to mean anything; or the prompt or tool
    schema has changed since the turns happened, which is the thing
    ``HISTORIC`` cannot correct for.
    """


class TurnSource(StrEnum):
    """Where one turn's incumbent decision actually came from.

    A run's ``IncumbentSource`` is what was asked for; this is what
    happened, recorded per turn because a historic run cannot reconstruct
    every turn.
    """

    LIVE = "live"
    HISTORIC = "historic"
    UNAVAILABLE = "unavailable"
    """The turn's recorded interactions were missing or unparseable.

    No decision could be reconstructed, so there is nothing to compare the
    candidate with. The turn is kept as evidence that it was sampled and
    dropped from every paired comparison, rather than guessed at.
    """


class SafetyFinding(StrEnum):
    """Something a model did that production would have acted on badly.

    Recorded per side and never averaged into a quality score. A finding is
    not disqualifying on its own: both models are held to the same checks,
    and the recommendation turns on whether the candidate does these things
    materially more often than the incumbent (see ``metrics._decide``). A
    model the incumbent matches finding for finding is not worse for having
    findings.
    """

    UNKNOWN_TOOL = "unknown_tool"
    """Called a tool name that was not in the schema it was offered."""

    INVALID_ARGS = "invalid_args"
    """Emitted arguments the tool would reject: its params model or its own precheck."""

    UNREQUESTED_MUTATION = "unrequested_mutation"
    """Made a write that neither the other model nor the live turn made.

    Counted because a model that reaches for writes nobody else reached for
    will bury the user in approval prompts, or write without one where the
    tool is ungated, and the prompt is only as good as the user reading it.
    """

    FABRICATED_ID = "fabricated_id"
    """Wrote to a record ID that appears nowhere in what the model was shown.

    The failure a reply-quality judge misses: a model that calls a search
    and a write in one response has to guess the ID the search would have
    returned, and a note filed against the neighbouring work order reads as
    decisive action. Checked deterministically against the prompt, the
    user's message and every tool result the model saw. See
    ``metrics.fabricated_ids``.
    """

    JUDGED_UNSAFE = "judged_unsafe"
    """The judge said this response would cause real harm if executed."""

    TRUNCATED = "truncated"
    """Hit the output token ceiling even after production's retry."""

    CALL_FAILED = "call_failed"
    """The provider raised. Recorded per turn rather than failing the run."""

    UNRESOLVED_TOOL_NAME = "unresolved_tool_name"
    """Called a tool that is in the replayed history but not in today's schema.

    A property of the fixture, not of the model: both read the name out of
    the conversation history and reach for it. The run warns instead,
    because it means the replay is scoring a tool surface the user no longer
    has.
    """


_SAFETY_FINDINGS = frozenset(
    {
        SafetyFinding.UNKNOWN_TOOL,
        SafetyFinding.INVALID_ARGS,
        SafetyFinding.UNREQUESTED_MUTATION,
        SafetyFinding.FABRICATED_ID,
        SafetyFinding.JUDGED_UNSAFE,
        SafetyFinding.TRUNCATED,
    }
)
"""Findings compared between the two sides to decide whether a switch is safe.

Lives here rather than in ``metrics`` so ``TurnComparison`` can consult it
without importing the module that imports this one. ``metrics`` re-exports it
as ``SAFETY_FINDINGS``, which is the name the rest of the package uses.

``CALL_FAILED`` and ``UNRESOLVED_TOOL_NAME`` are deliberately absent. Neither
is something a model did: the first is a failure to measure and the second is
a property of the replayed fixture. Both are still recorded on the turn and
surfaced, the first through ``turns_failed`` and the second through a run
warning.
"""


class AgreementClass(StrEnum):
    """How a candidate's decision for one turn relates to the incumbent's."""

    IDENTICAL = "identical"
    """Same tools, same arguments."""

    SAME_TOOLS_DIFFERENT_ARGS = "same_tools_different_args"

    DIFFERENT_TOOLS = "different_tools"

    REPLIED_INSTEAD_OF_ACTING = "replied_instead_of_acting"
    """Candidate answered in prose where the incumbent called a tool.

    The most important bucket in a downgrade. A weaker model that talks
    instead of acting still looks fluent, so this failure is invisible to
    any judge scoring reply quality and has to be counted structurally.
    """

    ACTED_INSTEAD_OF_REPLYING = "acted_instead_of_replying"

    BOTH_REPLIED = "both_replied"
    """Neither called a tool. Text quality is the judge's problem, not ours."""

    NOT_COMPARED = "not_compared"
    """The turn could not be replayed, so there is no decision to compare.

    Distinct from every class above: those describe a choice the candidate
    made. Recording one of them for a turn that never ran would put a
    fabricated value in the stored ``agreement`` column and sort the hardest
    failure to the bottom of the report."""


class JudgeVerdict(StrEnum):
    """Which of two diverging decisions the judge preferred."""

    EQUIVALENT = "equivalent"
    CANDIDATE_BETTER = "candidate_better"
    CANDIDATE_WORSE = "candidate_worse"
    CANDIDATE_UNSAFE = "candidate_unsafe"
    """Stored by runs recorded before unsafe flags became per-side findings.

    New runs record the preference here and the flag as a ``JUDGED_UNSAFE``
    finding on whichever side the judge named, including the incumbent.
    """
    NOT_JUDGED = "not_judged"
    JUDGE_FAILED = "judge_failed"


class JudgeSkipReason(StrEnum):
    """Why a turn carries no judge verdict.

    Only diverging, measurable, non-disqualified turns are adjudicated, so
    most turns in a healthy run are skipped for a good reason. Recording
    which reason lets the report account for every turn instead of leaving
    the operator to subtract the judge counts from the turn count and guess.
    """

    IDENTICAL = "identical"
    """Both models made the same call with the same arguments."""

    SAME_PROSE = "same_prose"
    """Neither called a tool and the two replies were the same text."""

    BLOCKING_FINDING = "blocking_finding"
    """Recorded by runs where one finding disqualified a switch. New runs
    judge those turns too, since a single finding no longer decides."""

    CALL_FAILED = "call_failed"
    """One of the two calls errored, so there was no decision to compare."""

    JUDGE_DISABLED = "judge_disabled"
    """The run was started with the judge turned off."""

    INCUMBENT_UNAVAILABLE = "incumbent_unavailable"
    """The incumbent's decision for this turn could not be reconstructed.

    Only reachable in ``IncumbentSource.HISTORIC``. There is one decision on
    the table, not two, and a judge asked to prefer one of them would be
    scoring the candidate against nothing.
    """


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    """Process died mid-run. Set at startup, never by the run itself."""
    CANCELLED = "cancelled"


class Recommendation(StrEnum):
    SAFE_TO_SWITCH = "safe_to_switch"
    SWITCH_WITH_MONITORING = "switch_with_monitoring"
    DO_NOT_SWITCH = "do_not_switch"
    INCONCLUSIVE = "inconclusive"
    """Too few turns completed to say anything. Not a pass."""


@dataclass(frozen=True)
class RunTargets:
    """Where each side of one run sends its calls, and at what effort.

    Resolved once per run rather than per turn, so an endpoint edited
    mid-run cannot move one side's traffic partway through a comparison and
    leave the two halves of the report describing different destinations.

    The two efforts are independent. Effort is not portable across model
    families, so holding both sides to one value measures the value rather
    than the candidate, and an endpoint that spells reasoning differently may
    refuse the incumbent's spelling outright.
    """

    baseline: LLMTarget
    candidate: LLMTarget
    # The judge rides the incumbent's endpoint. Its model is empty when the
    # run was started with judging off, in which case it is never called.
    judge: LLMTarget
    baseline_reasoning_effort: str = ""
    candidate_reasoning_effort: str = ""


@dataclass(frozen=True)
class RecordedToolResult:
    """A tool call and the result it returned.

    On a ``ReplaySample`` these are what the live turn called and got back,
    read from the stored ``tool_interactions_json``. On a ``ModelCallResult``
    they are the lookups a replay fed back to the model before its scored
    decision. Replaying a recorded result is not execution: nothing is called.
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
    historic_reply: str = ""
    historic_tool_names: list[str] = field(default_factory=list)
    historic_tool_results: tuple[RecordedToolResult, ...] = ()
    """Every tool call the live turn made, with the result it returned.

    What lets a replay continue past a lookup: a read-only call that matches
    one of these gets its recorded result back instead of a live call.
    """
    historic_first_calls: tuple[ToolCall, ...] = ()
    """The live turn's scored decision, as tool calls with their arguments.

    Not the same thing as ``historic_tool_names``, which is every call the
    turn made across every round, and not ``historic_reply``, which is the
    prose the user saw after every round had run. Reconstructed at the point
    in the turn the replay scores the candidate at: see
    ``sampling.HistoricDecision``.

    Empty when the decision was prose, which ``historic_decision_text``
    carries. See ``historic_decision_available`` for the case where there is
    no decision to read at all.
    """
    historic_decision_lookups: tuple[RecordedToolResult, ...] = ()
    """Recorded lookups the replay would have fed back before that decision.

    The incumbent's counterpart to ``ModelCallResult.replayed_lookups``, and
    what the judge prompt shows for this side so the two responses are not
    told apart by which of them lists lookups.
    """
    historic_decision_text: str = ""
    """The prose belonging to the scored round, empty when it has none.

    An outbound row holds the turn's final reply and its calls in one flat
    list, so a decision that opened with a call has no recorded text of its
    own: it carries none rather than the reply written after it.
    """
    historic_decision_available: bool = True
    """Whether a decision could be reconstructed for this turn.

    False when the turn was never answered, when an outbound row carried
    tool interactions that did not parse, or when the turn spent itself on
    lookups and recorded no prose after them. Those turns leave every paired
    comparison instead of being read as "the agent did nothing", which is
    what an unparseable row would otherwise look like.
    """
    historic_calls_flattened: bool = False
    """Whether this turn's recorded calls cannot be split into rounds.

    ``tool_interactions_json`` stores one flat, ordered list per outbound
    row, so a turn that recorded several calls could have made them in one
    round or in several. Turns in this state are counted and surfaced,
    because on one of them the incumbent may have asked for more in one
    breath than the reconstruction reads.
    """
    batched_messages: tuple[str, ...] = ()
    """Earlier messages of the batch this turn closes, oldest first.

    Production answers a rapid-fire batch once, at its last row, with the
    earlier rows already in the loaded history. The replay does the same, so
    these are in the prompt as history rather than in ``message_context``;
    they are carried here for the judge and the report, which would
    otherwise show a fraction of what the user asked.
    """

    @property
    def user_text(self) -> str:
        """Everything the user sent for this turn, batch included."""
        return "\n\n".join([*self.batched_messages, self.message_context])


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
    """What one model returned for one replayed turn."""

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

    The scored decision (``text``, ``tool_calls``) is what the model did
    after them, and the usage above covers every round. Empty when the model
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
class SafetyIssue:
    """A single safety finding with enough detail to act on it."""

    finding: SafetyFinding
    tool_name: str = ""
    detail: str = ""
    side: Side = Side.CANDIDATE
    """Whose finding this is. Rows written before both sides were checked
    carry none and were always the candidate's."""


@dataclass
class TurnComparison:
    """The full result for one replayed turn: both calls plus the verdict."""

    sample: ReplaySample
    baseline: ModelCallResult
    candidate: ModelCallResult
    agreement: AgreementClass
    baseline_source: TurnSource = TurnSource.LIVE
    """Where the incumbent's decision came from. See ``TurnSource``."""
    safety_issues: list[SafetyIssue] = field(default_factory=list)
    judge_verdict: JudgeVerdict = JudgeVerdict.NOT_JUDGED
    judge_rationale: str = ""
    judge_skip_reason: str | None = None
    """Set when ``judge_verdict`` is ``NOT_JUDGED``. See ``JudgeSkipReason``."""

    def has_safety_finding(self, side: Side) -> bool:
        """Whether *side* has a finding that counts in the safety comparison.

        ``CALL_FAILED`` and ``UNRESOLVED_TOOL_NAME`` are recorded on the turn
        but are not something a model did, so they must not count here. See
        ``metrics.SAFETY_FINDINGS``.
        """
        return any(
            issue.finding in _SAFETY_FINDINGS and issue.side is side for issue in self.safety_issues
        )

    def has_finding(self, side: Side, finding: SafetyFinding) -> bool:
        return any(issue.finding is finding and issue.side is side for issue in self.safety_issues)

    @property
    def diverged(self) -> bool:
        return self.agreement is not AgreementClass.IDENTICAL
