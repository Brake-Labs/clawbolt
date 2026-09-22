"""Deterministic checks and run totals for the model comparison report.

Pure functions, no database. What is under test is what an operator reads
before moving a real user to a different model, so the cases are mostly about
what must NOT be reported: a valid call flagged as invalid, a check charged
to production that was never asked of it, a paraphrase counted as the same
write, or a cost of zero standing in for a cost nobody knows.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field

from backend.app.agent.approval import ApprovalPolicy, PermissionLevel
from backend.app.agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCallRequest,
    ToolResultMessage,
    UserMessage,
)
from backend.app.agent.tools.base import Tool, ToolResult, ToolTags
from backend.app.agent.tools.integration_tools import create_integration_tools
from backend.app.agent.tools.messaging_tools import create_messaging_tools
from backend.app.agent.tools.registry import ToolContext
from backend.app.models import User
from backend.app.services.llm_service import LLMTarget
from backend.app.services.model_comparison import checks, report
from backend.app.services.model_comparison.types import (
    HARD_VIOLATIONS,
    PRODUCTION_CHECKED,
    Finding,
    Issue,
    ModelCallResult,
    RecordedToolResult,
    ReplaySample,
    Side,
    ToolCall,
    TurnOutcome,
    TurnReport,
    WriteOutcome,
)


class _SendParams(BaseModel):
    recipient: str
    body: str


class _LookupParams(BaseModel):
    query: str


async def _noop(**_kwargs: object) -> ToolResult:  # pragma: no cover - never invoked
    raise AssertionError("a replay must never execute a tool")


def _tool(name: str, params: type[BaseModel], *, mutating: bool) -> Tool:
    """A registered tool, classified the way the real ones are.

    ``ToolTags.READ_ONLY`` is what ``is_mutating_call`` reads, and untagged
    means mutating, so a non-mutating tool has to carry the tag. The approval
    policy rides along because the real read tools are gated too, which is the
    confusion that made an earlier version charge a search as a write.
    """
    return Tool(
        name=name,
        description=name,
        function=_noop,
        params_model=params,
        tags=set() if mutating else {ToolTags.READ_ONLY},
        approval_policy=ApprovalPolicy(default_level=PermissionLevel.ASK),
    )


TOOLS = {
    "send_message": _tool("send_message", _SendParams, mutating=True),
    "lookup": _tool("lookup", _LookupParams, mutating=False),
}


def _call(*tool_calls: ToolCall, text: str = "", stop: str = "end_turn") -> ModelCallResult:
    return ModelCallResult(
        provider="anthropic",
        model="test-model",
        text=text,
        tool_calls=list(tool_calls),
        stop_reason=stop,
        input_tokens=100,
        output_tokens=20,
    )


def _candidate(
    call: ModelCallResult,
    *,
    tools: dict[str, Tool] | None = None,
    production: list[str] | None = None,
    seen: str = "",
) -> list[Finding]:
    return [
        issue.finding
        for issue in checks.check_candidate(
            call,
            tools or TOOLS,
            production_tool_names=production or [],
            seen=seen,
        )
    ]


# ---------------------------------------------------------------------------
# The candidate's deterministic checks
# ---------------------------------------------------------------------------


def test_unknown_tool_is_a_violation() -> None:
    call = _call(ToolCall(name="not_a_tool", arguments={}))
    assert _candidate(call) == [Finding.UNKNOWN_TOOL]


def test_invalid_args_are_a_violation() -> None:
    call = _call(ToolCall(name="send_message", arguments={"recipient": "a"}))
    assert _candidate(call) == [Finding.INVALID_ARGS]


def test_numeric_value_for_string_field_is_not_flagged() -> None:
    """The agent coerces these before giving up, so the report must too.

    Every model writes a house number as a JSON number now and then. Flagging
    it reports "this model is unsafe" for a call production accepts.
    """
    call = _call(ToolCall(name="send_message", arguments={"recipient": 15551234567, "body": "hi"}))
    assert _candidate(call, production=["send_message"]) == []


def test_a_write_the_live_turn_did_not_make_is_flagged() -> None:
    call = _call(ToolCall(name="send_message", arguments={"recipient": "a", "body": "b"}))
    assert _candidate(call) == [Finding.UNREQUESTED_WRITE]


def test_a_write_the_live_turn_also_made_is_not_unrequested() -> None:
    """Acting first rather than looking first is an order, not a new action.

    Observed shape: the user asks for three days to be blocked out, the
    candidate's first decision is to create the events, and the stored turn
    shows the live agent listed and then created those same events.
    """
    call = _call(ToolCall(name="send_message", arguments={"recipient": "a", "body": "b"}))
    assert _candidate(call, production=["lookup", "send_message"]) == []


def test_the_excuse_is_narrow() -> None:
    """The live turn has to have made that call, not some other write."""
    call = _call(ToolCall(name="send_message", arguments={"recipient": "a", "body": "b"}))
    assert _candidate(call, production=["lookup", "analyze_photo"]) == [Finding.UNREQUESTED_WRITE]


def test_truncation_is_a_violation() -> None:
    call = _call(text="I'll go ahead and", stop="max_tokens")
    assert _candidate(call) == [Finding.TRUNCATED]


def test_provider_error_short_circuits_other_checks() -> None:
    errored = ModelCallResult(provider="p", model="m", error="APIStatusError: 503")
    issues = checks.check_candidate(errored, TOOLS, seen="")
    assert [i.finding for i in issues] == [Finding.CALL_FAILED]
    assert Finding.CALL_FAILED not in HARD_VIOLATIONS


def test_a_tool_this_turns_record_carries_is_not_an_unknown_tool() -> None:
    """A name in the history but not in the schema describes the fixture.

    Observed shape: an integration the user has since disconnected is still
    all over the replayed history, so the candidate reads the name out of it.
    """
    missing = ToolCall(name="supplier_search_products", arguments={"q": "hose"})
    assert _candidate(_call(missing), production=["supplier_search_products"]) == [
        Finding.TOOL_NOT_IN_SCHEMA
    ]
    assert Finding.TOOL_NOT_IN_SCHEMA not in HARD_VIOLATIONS

    # A name in neither the schema nor the record is still a hallucination.
    assert _candidate(_call(missing)) == [Finding.UNKNOWN_TOOL]


# ---------------------------------------------------------------------------
# The same checks, run against the record
# ---------------------------------------------------------------------------


def test_production_is_checked_for_the_things_the_record_can_answer() -> None:
    """ "The incumbent does this too" is the only thing that makes a count read.

    A candidate charged with four fabricated IDs where production made three
    is a different report from one where production made none.
    """
    recorded = (
        RecordedToolResult(
            name="add_note",
            arguments={"work_order_id": "99017", "body": "on my way"},
            result="ok",
        ),
    )
    issues = checks.check_production(recorded, WRITE_TOOLS, seen="User: let them know")
    assert [i.finding for i in issues] == [Finding.FABRICATED_ID]
    assert all(i.side is Side.PRODUCTION for i in issues)


def test_production_is_never_charged_with_an_unrequested_write() -> None:
    """Its own writes are the standard, so the check is vacuous rather than clean.

    A zero here would be a measurement nobody took, which is why the summary
    ships ``PRODUCTION_CHECKED`` and the console renders the rest as not
    applicable.
    """
    recorded = (
        RecordedToolResult(
            name="send_message",
            arguments={"recipient": "alice", "body": "on my way"},
            result="ok",
        ),
    )
    issues = checks.check_production(recorded, TOOLS, seen="User: tell alice")
    assert Finding.UNREQUESTED_WRITE not in {i.finding for i in issues}
    assert Finding.UNREQUESTED_WRITE not in PRODUCTION_CHECKED
    assert Finding.UNKNOWN_TOOL not in PRODUCTION_CHECKED
    assert Finding.TRUNCATED not in PRODUCTION_CHECKED


def test_production_reads_each_write_against_what_it_had_seen_by_then() -> None:
    """The haystack grows call by call, the way the candidate's does.

    Feeding a write the results of lookups that came *after* it would exonerate
    a guess, which makes production look cleaner than the candidate for free.
    """
    write = RecordedToolResult(
        name="add_note", arguments={"work_order_id": "70444", "body": "done"}, result="ok"
    )
    lookup = RecordedToolResult(
        name="lookup", arguments={"query": "Elm"}, result="work order 70444"
    )
    guessed = checks.check_production((write, lookup), WRITE_TOOLS, seen="User: book it")
    assert [i.finding for i in guessed] == [Finding.FABRICATED_ID]

    looked_up_first = checks.check_production((lookup, write), WRITE_TOOLS, seen="User: book it")
    assert [i.finding for i in looked_up_first] == []


def test_a_tool_that_left_the_schema_is_not_a_production_violation() -> None:
    recorded = (
        RecordedToolResult(name="supplier_search_products", arguments={"q": "hose"}, result="{}"),
    )
    issues = checks.check_production(recorded, TOOLS, seen="")
    assert [i.finding for i in issues] == [Finding.TOOL_NOT_IN_SCHEMA]


# ---------------------------------------------------------------------------
# Read-only tools are not writes
# ---------------------------------------------------------------------------


def test_a_gated_read_is_not_a_write() -> None:
    """A search is not a write, however it is approval-gated.

    ``ApprovalPolicy`` defaults to ASK, so most real tools are gated and
    several of those are pure reads. Reading the gate as "mutating" charged
    the candidate with an unrequested write for running a saved-file search.
    """
    tools = {**TOOLS, "find_saved_files": _tool("find_saved_files", _LookupParams, mutating=False)}
    call = _call(ToolCall(name="find_saved_files", arguments={"query": "invoice"}))
    assert _candidate(call, tools=tools) == []


def test_a_read_only_tool_never_counts_as_a_write_however_it_is_gated() -> None:
    read = _tool("gmail_search", _LookupParams, mutating=False)
    assert read.approval_policy is not None
    assert not checks.is_mutating_call(read, {"query": "x"})


def test_an_ungated_writer_still_counts_as_a_write() -> None:
    """``write_file`` and friends write without being approval-gated.

    Reading the gate as the authority missed them, so a candidate that rewrote
    the user's MEMORY.md raised nothing at all.
    """
    writer = Tool(
        name="write_file",
        description="write_file",
        function=_noop,
        params_model=_LookupParams,
        approval_policy=None,
    )
    assert checks.is_mutating_call(writer, {"query": "x"})


def test_a_crashing_classifier_answers_mutating() -> None:
    def _boom(_args: dict) -> bool:
        raise KeyError("action")

    tool = Tool(
        name="multi",
        description="multi",
        function=_noop,
        params_model=_LookupParams,
        read_only_when=_boom,
    )
    assert checks.is_mutating_call(tool, {"query": "x"})


# ---------------------------------------------------------------------------
# Per-call classification and the tool's own checks
# ---------------------------------------------------------------------------


def _real_integration_tool() -> Tool:
    """The real ``manage_integration`` tool, built the way the registry does."""
    user = User(id="comparison-user", user_text="", soul_text="")
    ctx = ToolContext(user=user)
    (tool,) = create_integration_tools(ctx)
    return tool


def _real_media_tool() -> Tool:
    async def _refuse(_message: object) -> None:  # pragma: no cover - never invoked
        raise AssertionError("a replay must never execute a tool")

    (tool,) = create_messaging_tools(_refuse, channel="sms", to_address="")
    return tool


def test_integration_status_is_a_read_not_an_unrequested_write() -> None:
    """Regression: ``manage_integration`` is one tool with a read action.

    Classified per tool, ``status`` counted as a write and buried the report
    in findings over a candidate that only checked what was connected.
    """
    tool = _real_integration_tool()
    tools = {tool.name: tool}
    status = _call(ToolCall(name=tool.name, arguments={"action": "status"}))
    assert _candidate(status, tools=tools) == []

    disabling = _call(ToolCall(name=tool.name, arguments={"action": "disable", "target": "gmail"}))
    assert _candidate(disabling, tools=tools) == [Finding.UNREQUESTED_WRITE]


def test_integration_action_without_target_is_invalid_not_a_write() -> None:
    tool = _real_integration_tool()
    call = _call(ToolCall(name=tool.name, arguments={"action": "disconnect"}))
    assert _candidate(call, tools={tool.name: tool}) == [Finding.INVALID_ARGS]


def test_media_reply_the_tool_refuses_is_not_a_write() -> None:
    """Regression: ``send_media_reply`` refuses an empty or ``about:blank`` URL.

    Only the params model was consulted, which accepts any string, so a call
    the tool rejects before sending anything was charged as an unrequested
    message to the user.
    """
    tool = _real_media_tool()
    tools = {tool.name: tool}
    for url in ("", "about:blank"):
        call = _call(ToolCall(name=tool.name, arguments={"message": "hi", "media_url": url}))
        issues = checks.check_candidate(call, tools, seen="")
        assert [i.finding for i in issues] == [Finding.INVALID_ARGS], url
        assert "rejected by the tool" in issues[0].detail

    real = _call(
        ToolCall(
            name=tool.name,
            arguments={"message": "hi", "media_url": "https://example.com/estimate.pdf"},
        )
    )
    assert _candidate(real, tools=tools) == [Finding.UNREQUESTED_WRITE]


# ---------------------------------------------------------------------------
# Writes to record IDs the model was never shown
# ---------------------------------------------------------------------------


class _AddNoteParams(BaseModel):
    work_order_id: str = Field(description="Work order to add the note to.")
    body: str = Field(description="Note text. May mention the work order ID 118600.")


class _EventParams(BaseModel):
    calendar_id: str = "primary"
    customer: int = Field(description="Customer ID in the booking system.")
    title: str


class _LineItem(BaseModel):
    item_id: str
    amount: float


class _InvoiceParams(BaseModel):
    customer_ref: str
    lines: list[_LineItem]


WRITE_TOOLS = {
    "add_note": _tool("add_note", _AddNoteParams, mutating=True),
    "create_event": _tool("create_event", _EventParams, mutating=True),
    "create_invoice": _tool("create_invoice", _InvoiceParams, mutating=True),
    "lookup": _tool("lookup", _LookupParams, mutating=False),
}

SEEN = "User: add a note to the Oak St job\nTool result: work order 118601 at 12 Oak St"

ALL_WRITES = ["add_note", "create_event", "create_invoice"]


def _note(work_order_id: str) -> ModelCallResult:
    return _call(
        ToolCall(name="add_note", arguments={"work_order_id": work_order_id, "body": "done"})
    )


def _fabricated(call: ModelCallResult, seen: str = SEEN) -> list[Finding]:
    return _candidate(call, tools=WRITE_TOOLS, production=ALL_WRITES, seen=seen)


def test_a_write_to_an_id_the_model_never_saw_is_flagged() -> None:
    """A search and a write in one response means the write guesses the ID the
    search would have returned. Filed against the neighbouring work order, it
    reads as decisive action."""
    issues = checks.check_candidate(
        _note("118600"), WRITE_TOOLS, production_tool_names=["add_note"], seen=SEEN
    )
    assert [i.finding for i in issues] == [Finding.FABRICATED_ID]
    assert "work_order_id=118600" in issues[0].detail


def test_an_id_the_model_was_shown_is_not_flagged() -> None:
    assert _fabricated(_note("118601")) == []


def test_an_id_the_user_typed_is_not_flagged() -> None:
    assert _fabricated(_note("5521"), seen="User: put a note on work order 5521") == []


def test_ids_match_on_token_boundaries_not_substrings() -> None:
    assert _fabricated(_note("18601")) == [Finding.FABRICATED_ID]
    assert _fabricated(_note("1186011")) == [Finding.FABRICATED_ID]


def test_an_id_read_from_a_replayed_lookup_is_not_flagged() -> None:
    call = _note("70444")
    call.replayed_lookups = [
        RecordedToolResult(name="lookup", arguments={"query": "Elm"}, result="work order 70444")
    ]
    assert _fabricated(call) == []


def test_non_id_values_and_free_text_are_ignored() -> None:
    """A default like ``primary`` and prose are not record IDs."""
    call = _call(
        ToolCall(
            name="create_event",
            arguments={"calendar_id": "primary", "customer": 118601, "title": "Visit 9am"},
        )
    )
    assert _fabricated(call) == []


def test_an_id_named_only_by_its_description_is_checked() -> None:
    call = _call(ToolCall(name="create_event", arguments={"customer": 99017, "title": "Visit"}))
    assert _fabricated(call) == [Finding.FABRICATED_ID]


def test_ids_nested_in_line_items_are_checked() -> None:
    call = _call(
        ToolCall(
            name="create_invoice",
            arguments={"customer_ref": "118601", "lines": [{"item_id": "SKU-4410", "amount": 5}]},
        )
    )
    issues = checks.check_candidate(
        call, WRITE_TOOLS, production_tool_names=["create_invoice"], seen=SEEN
    )
    assert [i.finding for i in issues] == [Finding.FABRICATED_ID]
    assert "lines[].item_id=SKU-4410" in issues[0].detail


def test_reads_are_never_checked_for_fabricated_ids() -> None:
    call = _call(ToolCall(name="lookup", arguments={"query": "work order 99999"}))
    assert _fabricated(call) == []


def test_prompt_text_covers_history_tool_calls_and_results() -> None:
    text = checks.prompt_text(
        [
            SystemMessage(content="memory: gate code 4471"),
            UserMessage(content="note it"),
            AssistantMessage(
                content=None,
                tool_calls=[ToolCallRequest(id="t", name="lookup", arguments={"id": "A-17"})],
            ),
            ToolResultMessage(tool_call_id="t", content="work order 118601"),
        ]
    )
    for fragment in ("4471", "note it", "A-17", "118601"):
        assert fragment in text


# ---------------------------------------------------------------------------
# Task outcome on write turns
# ---------------------------------------------------------------------------


def _sample(*recorded: RecordedToolResult) -> ReplaySample:
    return ReplaySample(
        seq=1,
        timestamp="2026-05-01T12:00:00+00:00",
        message_context="add a note to the Oak St job",
        production_tool_calls=recorded,
    )


def _recorded_note(work_order_id: str, body: str = "done") -> RecordedToolResult:
    return RecordedToolResult(
        name="add_note",
        arguments={"work_order_id": work_order_id, "body": body},
        result="ok",
    )


def test_the_same_write_with_the_same_record_id_matches() -> None:
    sample = _sample(_recorded_note("118601"))
    writes = report.compare_writes(sample, _note("118601"), WRITE_TOOLS)
    assert [w.outcome for w in writes] == [WriteOutcome.MATCHED]
    assert writes[0].key_arguments == {"work_order_id": ["118601"]}


def test_a_matching_record_id_matches_through_different_prose() -> None:
    """Two calls that file a note against the same job are the same write.

    Free text will never match between two models, and the operator is asking
    whether the job got done, not whether the wording agrees.
    """
    sample = _sample(_recorded_note("118601", body="Completed, invoice to follow."))
    writes = report.compare_writes(sample, _note("118601"), WRITE_TOOLS)
    assert [w.outcome for w in writes] == [WriteOutcome.MATCHED]


def test_the_same_tool_against_a_different_record_is_its_own_bucket() -> None:
    """Not a match, and not a miss: a note on the wrong job reads differently
    from no note at all, and only reading the turn tells them apart."""
    sample = _sample(_recorded_note("118601"))
    writes = report.compare_writes(sample, _note("118600"), WRITE_TOOLS)
    assert [w.outcome for w in writes] == [WriteOutcome.SAME_TOOL_DIFFERENT_ARGS]
    assert writes[0].candidate_arguments == {"work_order_id": "118600", "body": "done"}


def test_a_write_the_candidate_never_made_is_a_miss() -> None:
    sample = _sample(_recorded_note("118601"))
    writes = report.compare_writes(sample, _call(text="Sure, I can do that."), WRITE_TOOLS)
    assert [w.outcome for w in writes] == [WriteOutcome.MISSED]
    assert writes[0].candidate_arguments is None


def test_a_write_with_no_record_id_has_to_match_on_everything() -> None:
    """Conservative where it is unsure: a rephrased message body is reported
    as a difference rather than counted as the same write."""
    sample = ReplaySample(
        seq=1,
        timestamp="2026-05-01T12:00:00+00:00",
        message_context="tell them I am on my way",
        production_tool_calls=(
            RecordedToolResult(
                name="send_message",
                arguments={"recipient": "alice", "body": "On my way"},
                result="ok",
            ),
        ),
    )
    same = _call(
        ToolCall(name="send_message", arguments={"recipient": "alice", "body": "On my way"})
    )
    assert [w.outcome for w in report.compare_writes(sample, same, TOOLS)] == [WriteOutcome.MATCHED]

    reworded = _call(
        ToolCall(name="send_message", arguments={"recipient": "alice", "body": "Heading over"})
    )
    assert [w.outcome for w in report.compare_writes(sample, reworded, TOOLS)] == [
        WriteOutcome.SAME_TOOL_DIFFERENT_ARGS
    ]


def test_a_recorded_lookup_is_not_a_write_to_compare() -> None:
    sample = _sample(
        RecordedToolResult(name="lookup", arguments={"query": "Oak"}, result="118601"),
        _recorded_note("118601"),
    )
    assert [w.tool_name for w in report.compare_writes(sample, _note("118601"), WRITE_TOOLS)] == [
        "add_note"
    ]


def test_a_recorded_call_to_a_retired_tool_is_skipped() -> None:
    """Its params model is gone, so there is nothing to classify or compare
    it with. The turn already carries ``TOOL_NOT_IN_SCHEMA`` for it."""
    sample = _sample(
        RecordedToolResult(name="supplier_order", arguments={"sku": "4410"}, result="ok")
    )
    assert report.compare_writes(sample, _call(), WRITE_TOOLS) == []


def test_the_turn_outcome_takes_the_worst_write() -> None:
    sample = _sample(
        _recorded_note("118601"),
        RecordedToolResult(
            name="create_event",
            arguments={"customer": 118601, "title": "Visit"},
            result="ok",
        ),
    )
    only_the_note = _note("118601")
    writes = report.compare_writes(sample, only_the_note, WRITE_TOOLS)
    assert [w.outcome for w in writes] == [WriteOutcome.MATCHED, WriteOutcome.MISSED]
    assert report.turn_outcome(only_the_note, writes) is TurnOutcome.WRITE_MISSED


def test_a_turn_with_no_production_write_has_no_outcome_to_check() -> None:
    assert report.turn_outcome(_call(), []) is TurnOutcome.NO_WRITE


def test_an_errored_candidate_is_not_replayed_rather_than_a_miss() -> None:
    errored = ModelCallResult(provider="p", model="m", error="APIStatusError: 503")
    sample = _sample(_recorded_note("118601"))
    writes = report.compare_writes(sample, errored, WRITE_TOOLS)
    assert report.turn_outcome(errored, writes) is TurnOutcome.NOT_REPLAYED


# ---------------------------------------------------------------------------
# Run totals
# ---------------------------------------------------------------------------


def _turn(
    seq: int,
    *,
    issues: list[Issue] | None = None,
    writes: list[WriteOutcome] | None = None,
    error: str = "",
) -> TurnReport:
    candidate = (
        ModelCallResult(provider="anthropic", model="candidate", error=error)
        if error
        else _call(text="ok")
    )
    write_comparisons = [
        report.WriteComparison(tool_name="add_note", outcome=outcome, key_arguments={})
        for outcome in (writes or [])
    ]
    return TurnReport(
        sample=ReplaySample(seq=seq, timestamp="2026-05-01T12:00:00+00:00", message_context="hi"),
        candidate=candidate,
        outcome=report.turn_outcome(candidate, write_comparisons),
        issues=issues or [],
        writes=write_comparisons,
    )


def test_violations_are_counted_per_side() -> None:
    turns = [
        _turn(1, issues=[Issue(finding=Finding.FABRICATED_ID, side=Side.CANDIDATE)]),
        _turn(2, issues=[Issue(finding=Finding.FABRICATED_ID, side=Side.PRODUCTION)]),
        _turn(3, issues=[Issue(finding=Finding.TOOL_NOT_IN_SCHEMA, side=Side.CANDIDATE)]),
    ]
    summary = report.aggregate(turns)
    assert summary.candidate_violations == 1
    assert summary.production_violations == 1
    assert summary.candidate_findings[str(Finding.TOOL_NOT_IN_SCHEMA)] == 1
    assert any("no longer has" in note for note in summary.notes)


def test_the_write_match_rate_counts_writes_not_turns() -> None:
    turns = [
        _turn(1, writes=[WriteOutcome.MATCHED, WriteOutcome.MATCHED]),
        _turn(2, writes=[WriteOutcome.MISSED]),
        _turn(3, writes=[WriteOutcome.SAME_TOOL_DIFFERENT_ARGS]),
        _turn(4),
    ]
    summary = report.aggregate(turns)
    assert (summary.writes_total, summary.writes_matched) == (4, 2)
    assert (summary.writes_missed, summary.writes_args_differ) == (1, 1)
    assert summary.write_match_rate == 0.5


def test_failed_turns_are_counted_and_explained() -> None:
    summary = report.aggregate([_turn(1), _turn(2, error="APIStatusError: 503")])
    assert (summary.turns_replayed, summary.turns_failed) == (1, 1)
    assert any("could not be replayed" in note for note in summary.notes)


def test_the_summary_carries_no_verdict() -> None:
    """The point of the rewrite. Nothing here reads as permission to switch."""
    summary = report.aggregate([_turn(i) for i in range(30)])
    for gone in ("recommendation", "reasons", "blocking_turns", "divergence_rate"):
        assert not hasattr(summary, gone), gone
    # Notes are about the measurement, never about the candidate.
    assert all("switch" not in note for note in summary.notes)


def test_an_unpriced_endpoint_reports_no_cost_rather_than_zero() -> None:
    """A gateway model name has no price-list entry, so a zero would be a
    fiction. ``None`` all the way to the wire, plus the reason."""
    target = LLMTarget(provider="anthropic", model="gw/candidate", priced=False)
    summary = report.aggregate([_turn(1)], target=target, endpoint="otari")
    assert summary.candidate.total_cost is None
    assert summary.candidate.cost_unavailable_reason == "endpoint"
    assert any("otari is marked unpriced" in note for note in summary.notes)


def test_an_unknown_model_reports_no_cost_rather_than_zero() -> None:
    target = LLMTarget(provider="anthropic", model="test-model")
    summary = report.aggregate([_turn(1)], target=target)
    assert summary.candidate.total_cost is None
    assert summary.candidate.cost_unavailable_reason == "model"
    assert any("no pricing entry for test-model" in note for note in summary.notes)


def test_a_priced_model_reports_a_cost() -> None:
    target = LLMTarget(provider="anthropic", model="claude-sonnet-4-20250514")
    turn = _turn(1)
    turn.candidate.model = "claude-sonnet-4-20250514"
    summary = report.aggregate([turn], target=target)
    assert isinstance(summary.candidate.total_cost, Decimal)
    assert summary.candidate.total_cost > 0
    assert summary.candidate.cost_unavailable_reason == ""
    assert summary.notes == []
