"""Task outcome per turn, and the run totals the report shows.

Two things live here. ``compare_writes`` answers "did the candidate get the
job done": for every write the live turn made, whether the candidate reached
the same one. ``aggregate`` rolls the per-turn reports into the counts the
console renders.

Nothing here decides anything: no ceiling and no threshold, only counts. The
package docstring says why.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from backend.app.agent.tools.base import Tool
from backend.app.services.llm_pricing import compute_cost, is_known_model
from backend.app.services.llm_service import LLMTarget
from backend.app.services.model_comparison.checks import (
    canonical_args,
    collect_ids,
    id_properties,
    is_mutating_call,
)
from backend.app.services.model_comparison.production_usage import ProductionUsage
from backend.app.services.model_comparison.types import (
    Finding,
    ModelCallResult,
    RecordedToolResult,
    ReplaySample,
    Side,
    TurnOutcome,
    TurnReport,
    WriteComparison,
    WriteOutcome,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task outcome on write turns
# ---------------------------------------------------------------------------


def _validated(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """*args* after the params model fills its defaults, or as given."""
    try:
        return tool.params_model.model_validate(args).model_dump(mode="json")
    except ValidationError:
        return dict(args)


def key_arguments(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """The arguments two calls to *tool* have to agree on to be the same write.

    The whole validated argument set, always. Conservative in the sense that
    matters here: it never reports a match on a resemblance, and where it is
    unsure it reports a difference, which understates the candidate rather
    than flattering it.

    Agreement on record IDs alone is not a match. ``qb_update`` naming
    invoice 123 at $500 and ``qb_update`` naming estimate 123 at $5000 share
    every ID they carry, and scoring them as one write put an entity-type
    mismatch and a tenfold amount error into the headline rate. Record IDs
    still do a job, in ``record_ids`` below: they separate "right record,
    different content" from "a write against something else entirely".

    A rephrased message body therefore lands in a difference bucket rather
    than in ``MATCHED``. That is the intended reading: a reply is free text,
    two models will never spell it the same, and counting a paraphrase as a
    match would make the headline rate meaningless. The middle buckets are on
    the report for this reason.
    """
    return _validated(tool, args)


def record_ids(tool: Tool, args: dict[str, Any]) -> dict[str, list[str]]:
    """Every ID-shaped argument of *args*, grouped by parameter path, sorted.

    The same detection ``FABRICATED_ID`` uses (``checks.collect_ids``). Two
    writes that agree here are aimed at the same record, whatever else they
    carry, which is the difference between a note on the right job and a note
    on the neighbouring one.
    """
    grouped: dict[str, list[str]] = {}
    for path, value in collect_ids(args, id_properties(tool.params_model)):
        grouped.setdefault(path, []).append(value)
    return {path: sorted(values) for path, values in sorted(grouped.items())}


def _differing_arguments(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[str, ...]:
    """Top-level parameters whose values differ between two validated calls.

    A parameter present on one side only counts as differing: the candidate
    that dropped ``due_date`` made a different write, and a card that did not
    name it would leave the reader diffing two JSON blobs by eye.
    """
    names = set(expected) | set(actual)
    return tuple(
        sorted(
            name
            for name in names
            if canonical_args({"v": expected.get(name)}) != canonical_args({"v": actual.get(name)})
        )
    )


def production_writes(
    sample: ReplaySample, tools_by_name: dict[str, Tool]
) -> list[RecordedToolResult]:
    """The writes the live turn made, as today's tool classification reads them.

    A recorded call to a tool that has since left the schema is skipped: its
    params model is gone, so there is nothing to classify it with or to
    compare arguments against. Those calls are already surfaced as
    ``TOOL_NOT_IN_SCHEMA`` on the turn, so they are visible rather than
    silently dropped.
    """
    writes: list[RecordedToolResult] = []
    for call in sample.production_tool_calls:
        tool = tools_by_name.get(call.name)
        if tool is not None and is_mutating_call(tool, call.arguments):
            writes.append(call)
    return writes


# Best to worst, for picking one reading when the candidate called the same
# tool more than once. A candidate that reached the record and one that did
# not are not the same answer, so the better one is the one reported.
_WRITE_RANK = {
    WriteOutcome.MATCHED: 0,
    WriteOutcome.SAME_RECORD_DIFFERENT_ARGS: 1,
    WriteOutcome.SAME_TOOL_DIFFERENT_ARGS: 2,
    WriteOutcome.MISSED: 3,
    WriteOutcome.NOT_REACHED: 4,
}


def compare_writes(
    sample: ReplaySample,
    candidate: ModelCallResult,
    tools_by_name: dict[str, Tool],
) -> list[WriteComparison]:
    """For each write the live turn made, what the candidate did about it.

    The candidate's decision is its final round, after any recorded lookups
    it was allowed to continue through. That continuation is what makes this
    question answerable at all: a candidate that correctly looks a record up
    before writing would otherwise be scored on the lookup, and every
    lookup-then-write turn would report a miss. A candidate still looking
    things up when the round cap hit never got as far as the question, so
    every one of its writes is ``NOT_REACHED`` rather than ``MISSED``.

    Three readings above a miss, from the whole validated argument set down:
    the same arguments is ``MATCHED``, the same record IDs with different
    arguments is ``SAME_RECORD_DIFFERENT_ARGS``, and anything else the tool
    was called with is ``SAME_TOOL_DIFFERENT_ARGS``. A write carrying no
    record ID cannot reach the middle one: there is nothing to agree on.

    **Matching is greedy and per production write.** Each of production's
    writes is compared against every candidate call to that tool
    independently, and the best reading wins, so two production writes to one
    tool can both be judged against the same candidate call. Two
    ``add_note`` calls production made and one the candidate made can
    therefore report two matches rather than one match and one miss. A
    pairing that consumed each candidate call once would have to choose which
    production write to charge for the shortfall, and choosing wrong is worse
    than over-crediting a tool the candidate did reach: the count of writes
    is production's, and the operator reads the turn.
    """
    comparisons: list[WriteComparison] = []
    for write in production_writes(sample, tools_by_name):
        tool = tools_by_name[write.name]
        expected = key_arguments(tool, write.arguments)
        expected_ids = record_ids(tool, write.arguments)
        comparisons.append(
            _compare_one_write(
                tool,
                candidate,
                expected=expected,
                expected_ids=expected_ids,
            )
        )
    return comparisons


def _compare_one_write(
    tool: Tool,
    candidate: ModelCallResult,
    *,
    expected: dict[str, Any],
    expected_ids: dict[str, list[str]],
) -> WriteComparison:
    """The best reading of *candidate*'s calls against one production write."""
    if candidate.hit_read_round_cap:
        return WriteComparison(
            tool_name=tool.name,
            outcome=WriteOutcome.NOT_REACHED,
            key_arguments=expected,
            record_ids=expected_ids,
        )
    best = WriteComparison(
        tool_name=tool.name,
        outcome=WriteOutcome.MISSED,
        key_arguments=expected,
        record_ids=expected_ids,
    )
    for call in candidate.tool_calls:
        if call.name != tool.name:
            continue
        actual = key_arguments(tool, call.arguments)
        if canonical_args(actual) == canonical_args(expected):
            outcome = WriteOutcome.MATCHED
            differing: tuple[str, ...] = ()
        else:
            actual_ids = record_ids(tool, call.arguments)
            outcome = (
                WriteOutcome.SAME_RECORD_DIFFERENT_ARGS
                if expected_ids and actual_ids == expected_ids
                else WriteOutcome.SAME_TOOL_DIFFERENT_ARGS
            )
            differing = _differing_arguments(expected, actual)
        if _WRITE_RANK[outcome] < _WRITE_RANK[best.outcome]:
            best = WriteComparison(
                tool_name=tool.name,
                outcome=outcome,
                key_arguments=expected,
                candidate_arguments=call.arguments,
                record_ids=expected_ids,
                differing_arguments=differing,
            )
        if outcome is WriteOutcome.MATCHED:
            break
    return best


def turn_outcome(
    candidate: ModelCallResult,
    writes: Sequence[WriteComparison],
    *,
    production_acted: bool = False,
) -> TurnOutcome:
    """How this turn reads at a glance. Worst write wins.

    Three states come before the writes, because each of them says the write
    comparison is not what the reader should be looking at:

    - the provider errored, so there is no decision (``NOT_REPLAYED``);
    - the replay ran out of lookup rounds, so the decision on record is a
      read (``REPLAY_INCOMPLETE``);
    - production answered and the candidate returned nothing at all
      (``NO_CANDIDATE_OUTPUT``). Before the writes rather than after, because
      a silent candidate that also missed a write is better described by the
      silence. Nothing is lost by the ordering: the writes are still compared
      and still counted in ``writes_missed``.

    *production_acted* is whether the live turn produced anything the user
    would have seen, a reply or a tool call. Without it a turn where neither
    side said anything would read as a candidate failure.
    """
    if candidate.error:
        return TurnOutcome.NOT_REPLAYED
    if candidate.hit_read_round_cap:
        return TurnOutcome.REPLAY_INCOMPLETE
    if production_acted and candidate.produced_nothing:
        return TurnOutcome.NO_CANDIDATE_OUTPUT
    if not writes:
        return TurnOutcome.NO_WRITE
    outcomes = {w.outcome for w in writes}
    if WriteOutcome.MISSED in outcomes:
        return TurnOutcome.WRITE_MISSED
    if WriteOutcome.SAME_TOOL_DIFFERENT_ARGS in outcomes:
        return TurnOutcome.WRITE_ARGS_DIFFER
    if WriteOutcome.SAME_RECORD_DIFFERENT_ARGS in outcomes:
        return TurnOutcome.WRITE_SAME_RECORD
    return TurnOutcome.WRITE_MATCHED


# ---------------------------------------------------------------------------
# Run totals
# ---------------------------------------------------------------------------


@dataclass
class ModelTotals:
    """Token, cost, and latency totals for the candidate across a run."""

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_cost: Decimal | None = None
    """Dollars, or ``None`` when nothing here can price these tokens.

    ``None`` rather than zero, all the way to the wire. A gateway model name
    has no price-list entry, so the old column reported ``0.000000`` next to
    a warning nobody read, and a real-looking number beat the warning every
    time. See ``cost_unavailable_reason``.
    """
    cost_unavailable_reason: str = ""
    """Why there is no cost: "endpoint" or "model". Empty when priced."""
    latency_ms_samples: list[float] = field(default_factory=list)

    @property
    def billed_prompt_tokens(self) -> int:
        """Every prompt token the candidate was billed for, cached or not."""
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens

    def percentile_latency_ms(self, pct: float) -> float | None:
        """The *pct* latency across the run, or ``None`` with no samples.

        ``None`` rather than ``0.0``, for the same reason the cost is: a run
        where every call failed has no latency, and a zero there reads as an
        instantaneous model rather than as a measurement nobody took.
        """
        if not self.latency_ms_samples:
            return None
        ordered = sorted(self.latency_ms_samples)
        index = min(len(ordered) - 1, round((len(ordered) - 1) * pct))
        return ordered[index]


COST_COMPARABILITY = (
    "Cost is not comparable call for call. Providers bill different prompt "
    "token counts for byte-identical prompts (measured at up to 1.7x between "
    "two of the ones this deployment uses), and a replay's cache hits are not "
    "the live loop's: it re-sends one turn rather than following a "
    "conversation, so its cache reads and writes are an artifact of the "
    "measurement. Read the candidate figure as an order of magnitude beside "
    "the production window below it, not as a quote."
)
"""The caveat the cost tile is meaningless without.

It was measured evidence in the evaluator this replaced, and it did not
survive the rewrite. Kept as a constant so the note, the console and the
tests all say the same thing.
"""


@dataclass
class RunSummary:
    """Everything the report shows that is not a per-turn detail.

    Counts and totals only. There is no recommendation field here; the
    package docstring says why.
    """

    turns_total: int = 0
    """Turns the run *attempted*, which is every turn it sampled and every
    turn it could not replay. Not the size of the user's history, and not
    necessarily ``requested_samples``: a run that stopped on a dead provider
    attempted fewer."""
    turns_replayed: int = 0
    turns_failed: int = 0

    outcome_counts: dict[str, int] = field(default_factory=dict)
    candidate_findings: dict[str, int] = field(default_factory=dict)
    production_findings: dict[str, int] = field(default_factory=dict)
    candidate_violations: int = 0
    production_violations: int = 0

    writes_total: int = 0
    writes_matched: int = 0
    writes_same_record: int = 0
    writes_args_differ: int = 0
    writes_missed: int = 0
    writes_not_reached: int = 0
    """Writes on turns whose replay ran out of lookup rounds. Unmeasured, so
    they are excluded from ``write_match_rate`` rather than counted against
    the candidate."""

    candidate: ModelTotals = field(default_factory=ModelTotals)
    production: ProductionUsage = field(default_factory=ProductionUsage)
    notes: list[str] = field(default_factory=list)
    """Caveats about the measurement, never about the candidate."""

    @property
    def writes_measured(self) -> int:
        """Production writes the candidate was actually asked about."""
        return self.writes_total - self.writes_not_reached

    @property
    def write_match_rate(self) -> float:
        """Share of the measured writes the candidate reached identically.

        ``MATCHED`` only, which is the whole validated argument set. Reaching
        the right record with different arguments is its own bucket and is
        not in this number: an operator who wants credit for it can read the
        bucket, and folding it in here was how a tenfold amount error counted
        as a success.
        """
        return self.writes_matched / self.writes_measured if self.writes_measured else 0.0

    @property
    def silent_turns(self) -> int:
        """Turns production answered and the candidate did not."""
        return self.outcome_counts.get(str(TurnOutcome.NO_CANDIDATE_OUTPUT), 0)


def _accumulate(totals: ModelTotals, call: ModelCallResult) -> None:
    if call.error:
        return
    totals.provider = totals.provider or call.provider
    totals.model = totals.model or call.model
    totals.input_tokens += call.input_tokens
    totals.output_tokens += call.output_tokens
    totals.cache_read_tokens += call.cache_read_input_tokens
    totals.cache_creation_tokens += call.cache_creation_input_tokens
    totals.latency_ms_samples.append(call.latency_ms)


def _price(totals: ModelTotals, target: LLMTarget | None) -> None:
    """Fill in the cost, or say why there is none.

    A model served through a gateway is billed by whoever is behind it, which
    the (provider, model) pair no longer names, so a price-list hit on that
    pair is a coincidence rather than a cost. The endpoint says so via
    ``LLMTarget.priced``, and an unpriced endpoint leaves the cost ``None``.
    Wiring a real per-endpoint price would need price columns on
    ``llm_endpoints``, which do not exist; until they do, the honest report is
    the reason rather than a number.
    """
    priced_endpoint = target.priced if target else True
    if not priced_endpoint:
        totals.cost_unavailable_reason = "endpoint"
        return
    if not is_known_model(totals.model, provider=totals.provider):
        totals.cost_unavailable_reason = "model"
        return
    totals.total_cost = compute_cost(
        totals.model,
        totals.input_tokens,
        totals.output_tokens,
        provider=totals.provider,
        cache_creation_input_tokens=totals.cache_creation_tokens,
        cache_read_input_tokens=totals.cache_read_tokens,
    )


def _cost_note(summary: RunSummary, endpoint: str) -> str:
    """Why there is no cost figure.

    Appended whenever the cost is ``None``, including on a run where nothing
    was replayed. Gating it on a non-empty model left an all-failed run
    pointing at a note that was never written: ``_price`` reports "no pricing
    entry" for a model that is empty only because no call ever returned one.
    """
    totals = summary.candidate
    if not summary.turns_replayed:
        return (
            "Cost is not available: no turn was replayed successfully, so there are "
            "no tokens to price."
        )
    if totals.cost_unavailable_reason == "endpoint":
        where = f"endpoint {endpoint}" if endpoint else "this endpoint"
        return (
            f"Cost is not available: {where} is marked unpriced, so "
            f"({totals.provider}, {totals.model}) does not name who bills these tokens. "
            f"The token counts below are real; the dollar figure is not shown rather "
            f"than shown as zero."
        )
    named = totals.model or "this model"
    return (
        f"Cost is not available: no pricing entry for {named}. The token counts "
        f"below are real; the dollar figure is not shown rather than shown as zero."
    )


def aggregate(
    turns: Sequence[TurnReport],
    *,
    target: LLMTarget | None = None,
    endpoint: str = "",
    production: ProductionUsage | None = None,
) -> RunSummary:
    """Roll per-turn reports up into the run's counts.

    *target* supplies the candidate's pricing honesty and is omitted only by
    callers with no run to speak of. *endpoint* names it in the note.
    *production* is what the live side billed over the sampled window, read
    from ``llm_usage_logs`` by the caller, so the cost tile has both sides.
    """
    summary = RunSummary(turns_total=len(turns))
    if production is not None:
        summary.production = production

    for turn in turns:
        if turn.candidate.error:
            summary.turns_failed += 1
        else:
            summary.turns_replayed += 1

        key = str(turn.outcome)
        summary.outcome_counts[key] = summary.outcome_counts.get(key, 0) + 1

        for issue in turn.issues:
            counts = (
                summary.candidate_findings
                if issue.side is Side.CANDIDATE
                else summary.production_findings
            )
            name = str(issue.finding)
            counts[name] = counts.get(name, 0) + 1
            if issue.is_violation:
                if issue.side is Side.CANDIDATE:
                    summary.candidate_violations += 1
                else:
                    summary.production_violations += 1

        for write in turn.writes:
            summary.writes_total += 1
            if write.outcome is WriteOutcome.MATCHED:
                summary.writes_matched += 1
            elif write.outcome is WriteOutcome.SAME_RECORD_DIFFERENT_ARGS:
                summary.writes_same_record += 1
            elif write.outcome is WriteOutcome.SAME_TOOL_DIFFERENT_ARGS:
                summary.writes_args_differ += 1
            elif write.outcome is WriteOutcome.NOT_REACHED:
                summary.writes_not_reached += 1
            else:
                summary.writes_missed += 1

        _accumulate(summary.candidate, turn.candidate)

    _price(summary.candidate, target)
    summary.notes.extend(_notes(summary, endpoint))
    return summary


def _notes(summary: RunSummary, endpoint: str) -> list[str]:
    """The caveats a reader needs to read the numbers above them correctly."""
    notes: list[str] = []
    if summary.candidate.total_cost is None:
        notes.append(_cost_note(summary, endpoint))
    elif summary.turns_replayed:
        notes.append(COST_COMPARABILITY)

    if summary.turns_failed:
        notes.append(
            f"{summary.turns_failed} turn(s) could not be replayed, so the candidate has "
            f"no decision for them. They are listed with the provider's error."
        )

    if summary.silent_turns:
        notes.append(
            f"{summary.silent_turns} turn(s) got no text and no tool call from the "
            f"candidate where the live turn answered. Nothing else flags these: they "
            f"pass every check by producing nothing to check."
        )

    incomplete = summary.outcome_counts.get(str(TurnOutcome.REPLAY_INCOMPLETE), 0)
    if incomplete:
        notes.append(
            f"{incomplete} turn(s) were still looking records up when the replay's "
            f"lookup-round cap hit, so what the candidate would have written is not "
            f"measured. Their {summary.writes_not_reached} write(s) are left out of the "
            f"match rate rather than counted as missed."
        )

    retired = summary.candidate_findings.get(
        str(Finding.TOOL_NOT_IN_SCHEMA), 0
    ) + summary.production_findings.get(str(Finding.TOOL_NOT_IN_SCHEMA), 0)
    if retired:
        notes.append(
            f"{retired} call(s) named a tool this user's history carries but the current "
            f"schema does not. The replay is describing a tool surface the user no longer "
            f"has, on whichever side reached for it."
        )
    return notes
