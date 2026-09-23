"""Model comparison: run lifecycle, per-turn reports, and run summaries.

Nothing on the wire carries a verdict or a score. The summary is counts: hard
safety violations per side, how many of production's writes the candidate
reached, what it cost and how long it took.

The response models declare every field required, unlike the request model.
A field with a default is optional in the exported spec, so the generated
frontend types would make every count and every list possibly-undefined and
the console would spend its length in ``?? 0`` and ``?? []``. The server
writes all of these on every run, so required is both true and the shape the
console can read.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.schemas.common import ReasoningEffort


class ComparisonRunCreate(BaseModel):
    """Request to replay a user's recent turns through a candidate model.

    The baseline is not accepted from the client and is not a model at all:
    it is what production actually did, read back from the user's transcript.
    The run records the user's current model as a label so the report says
    what the candidate was being considered against.
    """

    candidate_endpoint: str = Field(default="", max_length=64)
    candidate_provider: str = Field(default="", max_length=64)
    candidate_model: str = Field(min_length=1, max_length=128)
    # Empty means "the deployment's current setting", resolved and frozen
    # onto the run at creation, so a mid-run change cannot redefine what was
    # measured.
    candidate_reasoning_effort: ReasoningEffort | Literal[""] = ""
    sample_count: int = Field(default=50, ge=1)


class ComparisonModelTotals(BaseModel):
    """Token, cost, and latency totals for the candidate across a run."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    billed_prompt_tokens: int
    # ``None`` when nothing can price these tokens, never "0.000000". A zero
    # reads as a measurement, and for a gateway model name it never is one.
    # ``cost_unavailable_reason`` says which: "endpoint" (marked unpriced, so
    # the (provider, model) pair does not name who billed) or "model" (no
    # price-list entry). The reason is also spelled out in ``notes``.
    total_cost_usd: str | None
    cost_unavailable_reason: str
    # ``None`` with no samples, never 0.0, for the same reason the cost is:
    # a run where every call failed took no measurement, and a zero reads as
    # an instantaneous model.
    latency_p50_ms: float | None
    latency_p95_ms: float | None


class ComparisonProductionUsage(BaseModel):
    """What the user's live loop billed over the window the run sampled.

    The other half of the cost tile. Not a like-for-like total: the window is
    the sampled turns' own timestamps, so it covers every call the live agent
    made inside it, while the candidate's figure counts one decision per
    turn. The console labels it as the user's spend over the same days rather
    than as what the replay would have cost.

    ``calls`` is 0 when there is nothing to show (no usage rows in the
    window, no parseable sample timestamps, or the read failed), which the
    console renders as unavailable.
    """

    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    billed_prompt_tokens: int
    # Summed over the priced rows only. ``None`` when none of them is priced.
    total_cost_usd: str | None
    # Rows behind an unpriced gateway, whose recorded cost is not a cost.
    # Non-zero means the dollar figure covers only part of the window.
    unpriced_calls: int
    window_start: str
    window_end: str


class ComparisonSummary(BaseModel):
    """The frozen summary stored on the run when it finished.

    Counts and totals. A reader who wants a recommendation reads the turns.
    """

    # Turns the run *attempted*: every turn it sampled plus every turn it
    # could not replay. Not the size of the user's history, and not
    # necessarily ``requested_samples``, which a run that stopped early never
    # reached.
    turns_total: int
    turns_replayed: int
    turns_failed: int

    # Keyed by ``TurnOutcome``: how each turn's candidate decision read
    # against the writes production made on it.
    outcome_counts: dict[str, int]

    # Findings by kind, per side. Both sides run the checks in
    # ``production_checked_findings``; the rest are candidate-only because
    # they cannot be asked of the record, and the console renders those as
    # not applicable for production rather than as a clean zero.
    candidate_findings: dict[str, int]
    production_findings: dict[str, int]
    candidate_violations: int
    production_violations: int
    production_checked_findings: list[str]

    # Task outcome on write turns, one entry per write production made.
    # ``writes_matched`` is agreement on the whole validated argument set;
    # ``writes_same_record`` is the right record with different arguments;
    # ``writes_args_differ`` is the tool called against something else;
    # ``writes_not_reached`` is a turn whose replay ran out of lookup rounds,
    # so the candidate was never asked. The last is excluded from
    # ``writes_measured``, which is the rate's denominator, because an
    # unfinished measurement is not a failure to write.
    writes_total: int
    writes_matched: int
    writes_same_record: int
    writes_args_differ: int
    writes_missed: int
    writes_not_reached: int
    writes_measured: int
    # ``writes_matched / writes_measured``. Full-argument agreement only:
    # reaching the right record with different arguments is its own bucket
    # and is deliberately not in this number.
    write_match_rate: float

    candidate: ComparisonModelTotals
    production: ComparisonProductionUsage
    # Caveats about the measurement: unavailable pricing and why, turns that
    # could not be replayed, calls naming a tool the current schema lacks.
    # Never about the candidate.
    notes: list[str]


class ComparisonRunItem(BaseModel):
    """One run, without its per-turn evidence."""

    id: str
    """The run's ``public_id``. Its report is addressed by this, not the row id."""

    user_id: str
    user_email: str
    """Whose run this is, for the cross-user listing. Empty when unknown."""

    user_consented: bool
    """Whether this run's evidence is still readable.

    A run survives its user withdrawing data-sharing consent, but the report
    endpoint refuses it from then on. The listing says so rather than offering
    a link that 403s.
    """
    # The user's model at the time the run was started. A label: no call was
    # sent to it, and the replayed turns may predate it.
    incumbent_endpoint: str
    incumbent_provider: str
    incumbent_model: str
    candidate_endpoint: str
    candidate_provider: str
    candidate_model: str
    candidate_reasoning_effort: str
    requested_samples: int
    status: str
    progress_completed: int
    progress_total: int
    error: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    summary: ComparisonSummary | None


class ComparisonRunListResponse(BaseModel):
    """Runs, plus the bounds the start form has to respect.

    ``max_samples`` is ``MODEL_COMPARISON_MAX_SAMPLES``, which ``start_run``
    enforces. Without it on the wire the sample control can only guess, and a
    deployment that lowers the setting gets a form offering values the API
    rejects.
    """

    runs: list[ComparisonRunItem]
    total: int
    """Runs matching the query, not just this page, so the console can page."""

    max_samples: int

    max_page_size: int
    """The largest ``limit`` this endpoint accepts.

    On the wire for the same reason ``max_samples`` is: a console that grows
    its own page size past the server's ceiling gets a 422 and a table that
    stops loading, including on every subsequent poll.
    """


class ComparisonRunProgress(BaseModel):
    """Just enough to answer "is it done yet".

    Carries no conversation content and no per-turn evidence, which is what
    lets the console poll it without writing an audit row every two seconds
    for a single human read.
    """

    id: str
    status: str
    progress_completed: int
    progress_total: int


class ComparisonFinding(BaseModel):
    finding: str
    tool_name: str
    detail: str
    # Whether this finding counts in the violation totals
    # (``model_comparison.types.HARD_VIOLATIONS``), as opposed to one that
    # describes the fixture or a failure to measure. Served rather than
    # re-derived client-side, so the console keeps no mirror of the set.
    violation: bool
    side: Literal["production", "candidate"]


class ComparisonToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    # Populated for a call read back from the record. The candidate's calls
    # were never executed, so they have no result.
    result: str
    is_error: bool


class ComparisonWrite(BaseModel):
    """One write the live turn made, and what the candidate did about it."""

    tool_name: str
    outcome: str
    # Production's whole validated argument set: what the candidate's call
    # had to agree with to be ``matched``. See ``report.compare_writes``.
    key_arguments: dict[str, Any]
    candidate_arguments: dict[str, Any] | None
    # The write's record IDs, by parameter path. Empty when it carries none,
    # which is what makes ``same_record_different_args`` unreachable for it.
    record_ids: dict[str, list[str]]
    # Parameters whose values differ between the two calls, so the card can
    # name them instead of leaving the reader to diff two JSON blobs.
    differing_arguments: list[str]


class ComparisonTurnItem(BaseModel):
    """One replayed turn, production's decision beside the candidate's."""

    message_seq: int
    message_timestamp: str
    user_message: str

    production_reply: str
    production_tool_calls: list[ComparisonToolCall]

    candidate_text: str
    candidate_tool_calls: list[ComparisonToolCall]
    # Read-only calls the replay answered from the live turn's recorded
    # results before the decision above, oldest first. Empty on a turn the
    # candidate decided in one round.
    candidate_replayed_lookups: list[ComparisonToolCall]
    candidate_stop_reason: str
    candidate_input_tokens: int
    candidate_output_tokens: int
    candidate_cache_read_tokens: int
    candidate_cache_creation_tokens: int
    candidate_latency_ms: float
    candidate_error: str

    outcome: str
    writes: list[ComparisonWrite]
    findings: list[ComparisonFinding]


class ComparisonReportResponse(BaseModel):
    """A run plus a page of its turns, the ones worth reading first."""

    run: ComparisonRunItem
    turns: list[ComparisonTurnItem]
    # Total turns stored for the run, so a caller can tell whether the page
    # it received is the whole story.
    total_turns: int
