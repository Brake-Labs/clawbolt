"""Model comparison: run lifecycle, per-turn reports, and run summaries.

Nothing here carries a verdict, a recommendation or a score. The summary is
counts: hard safety violations per side, how many of production's writes the
candidate reached, what it cost and how long it took.
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

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    billed_prompt_tokens: int = 0
    # ``None`` when nothing can price these tokens, never "0.000000". A zero
    # reads as a measurement, and for a gateway model name it never is one.
    # ``cost_unavailable_reason`` says which: "endpoint" (marked unpriced, so
    # the (provider, model) pair does not name who billed) or "model" (no
    # price-list entry). The reason is also spelled out in ``notes``.
    total_cost_usd: str | None = None
    cost_unavailable_reason: str = ""
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0


class ComparisonSummary(BaseModel):
    """The frozen summary stored on the run when it finished.

    Counts and totals. A reader who wants a recommendation reads the turns.
    """

    turns_total: int = 0
    turns_replayed: int = 0
    turns_failed: int = 0

    # Keyed by ``TurnOutcome``: how each turn's candidate decision read
    # against the writes production made on it.
    outcome_counts: dict[str, int] = Field(default_factory=dict)

    # Findings by kind, per side. Both sides run the checks in
    # ``production_checked_findings``; the rest are candidate-only because
    # they cannot be asked of the record, and the console renders those as
    # not applicable for production rather than as a clean zero.
    candidate_findings: dict[str, int] = Field(default_factory=dict)
    production_findings: dict[str, int] = Field(default_factory=dict)
    candidate_violations: int = 0
    production_violations: int = 0
    production_checked_findings: list[str] = Field(default_factory=list)

    # Task outcome on write turns. ``writes_args_differ`` is its own bucket
    # because a rephrased message body and a note filed against the wrong job
    # both land there, and only one of those is a problem.
    writes_total: int = 0
    writes_matched: int = 0
    writes_args_differ: int = 0
    writes_missed: int = 0
    write_match_rate: float = 0.0

    candidate: ComparisonModelTotals = Field(default_factory=ComparisonModelTotals)
    # Caveats about the measurement: unavailable pricing and why, turns that
    # could not be replayed, calls naming a tool the current schema lacks.
    # Never about the candidate.
    notes: list[str] = Field(default_factory=list)


class ComparisonRunItem(BaseModel):
    """One run, without its per-turn evidence."""

    id: str
    """The run's ``public_id``. Its report is addressed by this, not the row id."""

    user_id: str
    user_email: str = ""
    """Whose run this is, for the cross-user listing. Empty when unknown."""

    user_consented: bool = True
    """Whether this run's evidence is still readable.

    A run survives its user withdrawing data-sharing consent, but the report
    endpoint refuses it from then on. The listing says so rather than offering
    a link that 403s.
    """
    # The user's model at the time the run was started. A label: no call was
    # sent to it, and the replayed turns may predate it.
    incumbent_endpoint: str = ""
    incumbent_provider: str = ""
    incumbent_model: str = ""
    candidate_endpoint: str = ""
    candidate_provider: str
    candidate_model: str
    candidate_reasoning_effort: str = ""
    requested_samples: int
    status: str
    progress_completed: int
    progress_total: int
    error: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    summary: ComparisonSummary | None = None


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
    tool_name: str = ""
    detail: str = ""
    # Whether this finding counts in the violation totals
    # (``model_comparison.types.HARD_VIOLATIONS``), as opposed to one that
    # describes the fixture or a failure to measure. Served rather than
    # re-derived client-side, so the console keeps no mirror of the set.
    violation: bool = True
    side: Literal["production", "candidate"] = "candidate"


class ComparisonToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Populated for a call read back from the record. The candidate's calls
    # were never executed, so they have no result.
    result: str = ""
    is_error: bool = False


class ComparisonWrite(BaseModel):
    """One write the live turn made, and what the candidate did about it."""

    tool_name: str
    outcome: str
    # The arguments compared: the write's record IDs, or its whole validated
    # argument set when it has none. See ``report.key_arguments``.
    key_arguments: dict[str, Any] = Field(default_factory=dict)
    candidate_arguments: dict[str, Any] | None = None


class ComparisonTurnItem(BaseModel):
    """One replayed turn, production's decision beside the candidate's."""

    message_seq: int
    message_timestamp: str
    user_message: str

    production_reply: str = ""
    production_tool_calls: list[ComparisonToolCall] = Field(default_factory=list)

    candidate_text: str = ""
    candidate_tool_calls: list[ComparisonToolCall] = Field(default_factory=list)
    # Read-only calls the replay answered from the live turn's recorded
    # results before the decision above, oldest first. Empty on a turn the
    # candidate decided in one round.
    candidate_replayed_lookups: list[ComparisonToolCall] = Field(default_factory=list)
    candidate_stop_reason: str = ""
    candidate_input_tokens: int = 0
    candidate_output_tokens: int = 0
    candidate_cache_read_tokens: int = 0
    candidate_cache_creation_tokens: int = 0
    candidate_latency_ms: float = 0.0
    candidate_error: str = ""

    outcome: str
    writes: list[ComparisonWrite] = Field(default_factory=list)
    findings: list[ComparisonFinding] = Field(default_factory=list)


class ComparisonReportResponse(BaseModel):
    """A run plus a page of its turns, the ones worth reading first."""

    run: ComparisonRunItem
    turns: list[ComparisonTurnItem]
    # Total turns stored for the run, so a caller can tell whether the page
    # it received is the whole story.
    total_turns: int = 0
