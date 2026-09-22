"""Task outcome per turn, and the run totals the report shows.

Two things live here. ``compare_writes`` answers "did the candidate get the
job done": for every write the live turn made, whether the candidate reached
the same one. ``aggregate`` rolls the per-turn reports into the counts the
console renders.

Nothing here decides anything. There is no recommendation, no ceiling and no
threshold: the summary is a count of hard violations per side, a write-match
rate, and the candidate's cost and latency. A reader who wants a verdict gets
it from reading the turns, which is what the report is laid out for.
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


def key_arguments(tool: Tool, args: dict[str, Any], *, by_ids: bool) -> dict[str, Any]:
    """The arguments two calls to *tool* have to agree on to be the same write.

    Conservative in the sense that matters here: it never reports a match on
    a resemblance, and where it is unsure it reports a difference, which
    understates the candidate rather than flattering it.

    *by_ids* picks the rule, and the live turn's own call picks *by_ids*:

    - **Record IDs, when the write has any.** Every ID-shaped argument
      (``checks.collect_ids``, the same detection ``FABRICATED_ID`` uses),
      grouped by parameter path and sorted. Two calls that file a note
      against the same job are the same write whether or not the note reads
      the same, which is the outcome the operator is asking about.
    - **Every argument, when it has none.** A write with no record ID is
      identified only by what it carries, so the whole validated argument set
      has to match. A rephrased message body therefore lands in
      ``SAME_TOOL_DIFFERENT_ARGS`` rather than in ``MATCHED``. That is the
      intended reading: a reply is free text, two models will never spell it
      the same, and counting a paraphrase as a match would make the headline
      rate meaningless. The middle bucket is on the report for this reason.
    """
    if not by_ids:
        return _validated(tool, args)
    grouped: dict[str, list[str]] = {}
    for path, value in collect_ids(args, id_properties(tool.params_model)):
        grouped.setdefault(path, []).append(value)
    return {path: sorted(values) for path, values in sorted(grouped.items())}


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
    lookup-then-write turn would report a miss.
    """
    comparisons: list[WriteComparison] = []
    for write in production_writes(sample, tools_by_name):
        tool = tools_by_name[write.name]
        by_ids = bool(collect_ids(write.arguments, id_properties(tool.params_model)))
        expected = key_arguments(tool, write.arguments, by_ids=by_ids)
        same_tool = [c for c in candidate.tool_calls if c.name == write.name]
        outcome = WriteOutcome.MISSED
        candidate_arguments: dict[str, Any] | None = None
        for call in same_tool:
            candidate_arguments = candidate_arguments or call.arguments
            actual = key_arguments(tool, call.arguments, by_ids=by_ids)
            if canonical_args(actual) == canonical_args(expected):
                outcome = WriteOutcome.MATCHED
                candidate_arguments = call.arguments
                break
            outcome = WriteOutcome.SAME_TOOL_DIFFERENT_ARGS
        comparisons.append(
            WriteComparison(
                tool_name=write.name,
                outcome=outcome,
                key_arguments=expected,
                candidate_arguments=candidate_arguments,
            )
        )
    return comparisons


def turn_outcome(candidate: ModelCallResult, writes: Sequence[WriteComparison]) -> TurnOutcome:
    """How this turn reads at a glance. Worst write wins."""
    if candidate.error:
        return TurnOutcome.NOT_REPLAYED
    if not writes:
        return TurnOutcome.NO_WRITE
    outcomes = {w.outcome for w in writes}
    if WriteOutcome.MISSED in outcomes:
        return TurnOutcome.WRITE_MISSED
    if WriteOutcome.SAME_TOOL_DIFFERENT_ARGS in outcomes:
        return TurnOutcome.WRITE_ARGS_DIFFER
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

    def percentile_latency_ms(self, pct: float) -> float:
        if not self.latency_ms_samples:
            return 0.0
        ordered = sorted(self.latency_ms_samples)
        index = min(len(ordered) - 1, round((len(ordered) - 1) * pct))
        return ordered[index]


@dataclass
class RunSummary:
    """Everything the report shows that is not a per-turn detail.

    Counts and totals only. There is no recommendation field here, and
    adding one back is the change this rewrite exists to prevent: the
    operator reads three users' transcripts anyway, and a verdict computed
    off a handful of noisy turns was wrong in both directions.
    """

    turns_total: int = 0
    turns_replayed: int = 0
    turns_failed: int = 0

    outcome_counts: dict[str, int] = field(default_factory=dict)
    candidate_findings: dict[str, int] = field(default_factory=dict)
    production_findings: dict[str, int] = field(default_factory=dict)
    candidate_violations: int = 0
    production_violations: int = 0

    writes_total: int = 0
    writes_matched: int = 0
    writes_args_differ: int = 0
    writes_missed: int = 0

    candidate: ModelTotals = field(default_factory=ModelTotals)
    notes: list[str] = field(default_factory=list)
    """Caveats about the measurement, never about the candidate."""

    @property
    def write_match_rate(self) -> float:
        """Share of the live turns' writes the candidate reached identically."""
        return self.writes_matched / self.writes_total if self.writes_total else 0.0


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


def _cost_note(totals: ModelTotals, endpoint: str) -> str:
    if totals.cost_unavailable_reason == "endpoint":
        where = f"endpoint {endpoint}" if endpoint else "this endpoint"
        return (
            f"Cost is not available: {where} is marked unpriced, so "
            f"({totals.provider}, {totals.model}) does not name who bills these tokens. "
            f"The token counts below are real; the dollar figure is not shown rather "
            f"than shown as zero."
        )
    return (
        f"Cost is not available: no pricing entry for {totals.model}. The token counts "
        f"below are real; the dollar figure is not shown rather than shown as zero."
    )


def aggregate(
    turns: Sequence[TurnReport],
    *,
    target: LLMTarget | None = None,
    endpoint: str = "",
) -> RunSummary:
    """Roll per-turn reports up into the run's counts.

    *target* supplies the candidate's pricing honesty and is omitted only by
    callers with no run to speak of. *endpoint* names it in the note.
    """
    summary = RunSummary(turns_total=len(turns))

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
            elif write.outcome is WriteOutcome.SAME_TOOL_DIFFERENT_ARGS:
                summary.writes_args_differ += 1
            else:
                summary.writes_missed += 1

        _accumulate(summary.candidate, turn.candidate)

    _price(summary.candidate, target)
    if summary.candidate.total_cost is None and summary.candidate.model:
        summary.notes.append(_cost_note(summary.candidate, endpoint))

    if summary.turns_failed:
        summary.notes.append(
            f"{summary.turns_failed} turn(s) could not be replayed, so the candidate has "
            f"no decision for them. They are listed with the provider's error."
        )

    retired = summary.candidate_findings.get(
        str(Finding.TOOL_NOT_IN_SCHEMA), 0
    ) + summary.production_findings.get(str(Finding.TOOL_NOT_IN_SCHEMA), 0)
    if retired:
        summary.notes.append(
            f"{retired} call(s) named a tool this user's history carries but the current "
            f"schema does not. The replay is describing a tool surface the user no longer "
            f"has, on whichever side reached for it."
        )

    return summary
