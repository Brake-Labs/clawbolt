"""End-to-end orchestration of one model comparison run.

Model dispatch is mocked; the run's own machinery is not. What is being
tested is that the run writes its evidence as it goes, survives a turn that
fails, honors cancellation, and never executes a tool.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from backend.app.agent.core import AssembledPrompt
from backend.app.agent.messages import SystemMessage, UserMessage
from backend.app.agent.tools.base import Tool, ToolResult, ToolTags
from backend.app.models import ComparisonRun, ComparisonTurn, User
from backend.app.services.llm_service import LLMTarget
from backend.app.services.model_comparison.runner import (
    MAX_CONSECUTIVE_CALL_FAILURES,
    execute_run,
    mark_interrupted_runs,
)
from backend.app.services.model_comparison.sampling import ReplayFixture
from backend.app.services.model_comparison.types import (
    ModelCallResult,
    RecordedToolResult,
    ReplaySample,
    RunStatus,
    ToolCall,
    TurnOutcome,
)


def _make_run(
    db: Session,
    user_id: str,
    *,
    samples: int = 3,
    candidate_effort: str = "",
) -> int:
    run = ComparisonRun(
        user_id=user_id,
        incumbent_provider="anthropic",
        incumbent_model="incumbent",
        candidate_provider="anthropic",
        candidate_model="candidate",
        candidate_reasoning_effort=candidate_effort,
        requested_samples=samples,
        status=str(RunStatus.PENDING),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run.id


def _samples(count: int) -> list[ReplaySample]:
    return [
        ReplaySample(
            seq=i,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context=f"ask {i}",
            production_reply=f"answer {i}",
            production_tool_calls=(
                RecordedToolResult(name="lookup", arguments={"q": "a"}, result="ok"),
            ),
        )
        for i in range(1, count + 1)
    ]


def _assembled() -> AssembledPrompt:
    return AssembledPrompt(
        messages=[
            SystemMessage(content="system"),
            UserMessage(content="earlier ask"),
            UserMessage(content="current ask"),
        ],
        stable_system="system",
        dynamic_context="",
        system_prompt="system",
    )


def _patched_run(
    *,
    samples: list[ReplaySample],
    call_side_effect: object,
    tools_by_name: dict | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Patch the run's collaborators, leaving the orchestration real."""
    fixture = ReplayFixture(user=User(id="u"), rows=[])
    fixture.tools_by_name = tools_by_name or {}
    return (
        patch(
            "backend.app.services.model_comparison.runner.build_fixture",
            AsyncMock(return_value=fixture),
        ),
        patch(
            "backend.app.services.model_comparison.runner.select_samples",
            return_value=samples,
        ),
        patch(
            "backend.app.services.model_comparison.runner.assemble_for_sample",
            AsyncMock(return_value=_assembled()),
        ),
        patch(
            "backend.app.services.model_comparison.runner.call_model",
            AsyncMock(side_effect=call_side_effect),
        ),
    )


class _LookupParams(BaseModel):
    q: str


class _NoteParams(BaseModel):
    work_order_id: str
    body: str


def _lookup_tool(function: Callable[..., Awaitable[ToolResult]] | None = None) -> Tool:
    """A registered, non-mutating tool the model is allowed to call.

    Runs that call a tool need it present in ``tools_by_name``: an
    unregistered name is a genuine violation, which would put a finding on
    the report for the wrong reason.
    """

    async def _unused(**_kwargs: object) -> ToolResult:  # pragma: no cover
        raise AssertionError("a tool was executed during a comparison run")

    return Tool(
        name="lookup",
        description="lookup",
        function=function or _unused,
        params_model=_LookupParams,
        tags={ToolTags.READ_ONLY},
    )


def _note_tool(function: Callable[..., Awaitable[ToolResult]] | None = None) -> Tool:
    async def _unused(**_kwargs: object) -> ToolResult:  # pragma: no cover
        raise AssertionError("a tool was executed during a comparison run")

    return Tool(
        name="add_note",
        description="add_note",
        function=function or _unused,
        params_model=_NoteParams,
    )


def _result(text: str = "", tools: list[ToolCall] | None = None) -> ModelCallResult:
    return ModelCallResult(
        provider="anthropic",
        model="m",
        text=text,
        tool_calls=tools or [],
        stop_reason="end_turn",
        input_tokens=100,
        output_tokens=10,
    )


def _turns(db: Session, run_id: int) -> list[ComparisonTurn]:
    return list(
        db.execute(select(ComparisonTurn).where(ComparisonTurn.run_id == run_id)).scalars().all()
    )


async def test_run_writes_a_turn_row_per_sample_and_completes(
    db_session: Session, test_user: User
) -> None:
    run_id = _make_run(db_session, test_user.id, samples=3)
    matching = _result(tools=[ToolCall(name="lookup", arguments={"q": "a"})])

    patches = _patched_run(
        samples=_samples(3),
        call_side_effect=lambda *a, **k: matching,
        tools_by_name={"lookup": _lookup_tool()},
    )
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=2)

    db_session.expire_all()
    run = db_session.get(ComparisonRun, run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert run.progress_completed == 3
    assert run.progress_total == 3
    assert run.summary_json is not None
    assert run.summary_json["turns_replayed"] == 3
    # No verdict column, and no verdict in the summary. A short run has
    # nothing to say beyond what it measured.
    assert not hasattr(run, "recommendation")
    assert "recommendation" not in run.summary_json

    turns = _turns(db_session, run_id)
    assert len(turns) == 3
    assert {t.message_seq for t in turns} == {1, 2, 3}
    stored = json.loads(turns[0].candidate_tool_calls)
    assert stored[0]["name"] == "lookup"
    # What production did is on the row too, from the record rather than a
    # second call.
    recorded = json.loads(turns[0].production_tool_calls)
    assert recorded[0] == {
        "name": "lookup",
        "arguments": {"q": "a"},
        "result": "ok",
        "is_error": False,
    }


async def test_only_the_candidate_is_called(db_session: Session, test_user: User) -> None:
    """One provider call per turn, at the effort recorded on the run.

    The baseline is the recorded turn, so a second live call would be paying
    to re-sample a model whose answer is already in the transcript.
    """
    run_id = _make_run(db_session, test_user.id, samples=2, candidate_effort="high")
    seen: list[tuple[str, str]] = []

    async def record(*_args: object, **kwargs: Any) -> ModelCallResult:
        target: LLMTarget = kwargs["target"]
        seen.append((target.model, kwargs["reasoning_effort"]))
        return _result(text="ok")

    a, b, c, d = _patched_run(samples=_samples(2), call_side_effect=record)
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    assert seen == [("candidate", "high"), ("candidate", "high")]


async def test_the_write_comparison_reaches_the_turn_row(
    db_session: Session, test_user: User
) -> None:
    """The task outcome an operator is actually asking about."""
    run_id = _make_run(db_session, test_user.id, samples=1)
    samples = [
        ReplaySample(
            seq=1,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context="note the Oak St job",
            production_tool_calls=(
                RecordedToolResult(
                    name="add_note",
                    arguments={"work_order_id": "118601", "body": "done"},
                    result="ok",
                ),
            ),
        )
    ]
    reached = _result(
        tools=[ToolCall(name="add_note", arguments={"work_order_id": "118601", "body": "all set"})]
    )
    a, b, c, d = _patched_run(
        samples=samples,
        call_side_effect=lambda *x, **k: reached,
        tools_by_name={"add_note": _note_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    (row,) = _turns(db_session, run_id)
    assert row.outcome == str(TurnOutcome.WRITE_MATCHED)
    writes = json.loads(row.write_results)
    assert writes[0]["tool_name"] == "add_note"
    assert writes[0]["key_arguments"] == {"work_order_id": ["118601"]}

    run = db_session.get(ComparisonRun, run_id)
    assert run is not None and run.summary_json is not None
    assert run.summary_json["writes_matched"] == 1
    assert run.summary_json["write_match_rate"] == 1.0


async def test_production_is_checked_alongside_the_candidate(
    db_session: Session, test_user: User
) -> None:
    """The report has to be able to say the incumbent does this too."""
    run_id = _make_run(db_session, test_user.id, samples=1)
    samples = [
        ReplaySample(
            seq=1,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context="note it",
            production_tool_calls=(
                RecordedToolResult(
                    name="add_note",
                    arguments={"work_order_id": "999999", "body": "done"},
                    result="ok",
                ),
            ),
        )
    ]
    a, b, c, d = _patched_run(
        samples=samples,
        call_side_effect=lambda *x, **k: _result(text="on it"),
        tools_by_name={"add_note": _note_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    (row,) = _turns(db_session, run_id)
    findings = json.loads(row.findings)
    assert {"fabricated_id"} == {f["finding"] for f in findings}
    assert {"production"} == {f["side"] for f in findings}

    run = db_session.get(ComparisonRun, run_id)
    assert run is not None and run.summary_json is not None
    assert run.summary_json["production_violations"] == 1
    assert run.summary_json["candidate_violations"] == 0
    # And the console is told which checks production was asked at all.
    assert "unrequested_write" not in run.summary_json["production_checked_findings"]


async def test_a_failing_turn_does_not_abort_the_run(db_session: Session, test_user: User) -> None:
    run_id = _make_run(db_session, test_user.id, samples=3)
    calls = {"n": 0}

    async def flaky(*_args: object, **_kwargs: object) -> ModelCallResult:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("gateway blew up")
        return _result(text="fine")

    patches = _patched_run(samples=_samples(3), call_side_effect=None)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("backend.app.services.model_comparison.runner.call_model", flaky),
    ):
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    run = db_session.get(ComparisonRun, run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    turns = _turns(db_session, run_id)
    assert len(turns) == 3
    assert sum(1 for t in turns if t.candidate_error) == 1


async def test_cancelled_run_stops_and_keeps_the_turns_it_finished(
    db_session: Session, test_user: User
) -> None:
    run_id = _make_run(db_session, test_user.id, samples=4)

    async def cancel_after_first(*_args: object, **_kwargs: object) -> ModelCallResult:
        # Flip the row the way the cancel endpoint does, so the worker sees
        # it on its next turn.
        db_session.execute(
            update(ComparisonRun)
            .where(ComparisonRun.id == run_id)
            .values(status=str(RunStatus.CANCELLED))
        )
        db_session.commit()
        return _result(text="ok")

    patches = _patched_run(samples=_samples(4), call_side_effect=None)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("backend.app.services.model_comparison.runner.call_model", cancel_after_first),
        # The watcher caches the flag for a second so a run does not open a
        # session per turn to poll one column. Turns here finish in
        # microseconds, so without this the cancellation lands inside the
        # cache window and the run completes normally.
        patch("backend.app.services.model_comparison.runner._CANCEL_POLL_SECONDS", 0),
        pytest.raises(asyncio.CancelledError),
    ):
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    assert len(_turns(db_session, run_id)) < 4


async def test_a_run_deleted_mid_flight_unwinds_quietly(
    db_session: Session, test_user: User
) -> None:
    """Deleting a cancelled run while a worker is still inside a turn.

    This is the documented remedy for deleting an active run (cancel, then
    delete) and it races: cancellation is only checked at the top of a turn,
    so a worker can be inside a provider call when the row goes away. Its
    turn insert then hits the foreign key.

    The insert has to be swallowed rather than raised. ``_guarded`` turns any
    escaping exception into a logged stack trace plus a ``FAILED`` stamp, so
    raising would report a deliberate deletion as a crashed run, and on a
    monitored deployment would alert on it.
    """
    run_id = _make_run(db_session, test_user.id, samples=1)

    async def delete_the_run(*_args: object, **_kwargs: object) -> ModelCallResult:
        db_session.execute(delete(ComparisonRun).where(ComparisonRun.id == run_id))
        db_session.commit()
        return _result(text="ok")

    patches = _patched_run(samples=_samples(1), call_side_effect=None)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("backend.app.services.model_comparison.runner.call_model", delete_the_run),
    ):
        # No raise: the assertion is that nothing propagates.
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    assert db_session.get(ComparisonRun, run_id) is None
    assert _turns(db_session, run_id) == []


async def test_a_turn_that_cannot_be_replayed_is_marked_not_replayed(
    db_session: Session, test_user: User
) -> None:
    """A turn that fails to assemble must keep a failure marker.

    Otherwise the hardest failure is the least visible one: it carries a
    fabricated outcome and sorts below every turn that did run.
    """
    run_id = _make_run(db_session, test_user.id, samples=2)

    async def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("prompt could not be assembled")

    patches = _patched_run(samples=_samples(2), call_side_effect=None)
    with (
        patches[0],
        patches[1],
        patch("backend.app.services.model_comparison.runner.assemble_for_sample", explode),
        patches[3],
    ):
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    turns = _turns(db_session, run_id)
    assert len(turns) == 2
    for t in turns:
        assert t.outcome == str(TurnOutcome.NOT_REPLAYED)
        assert "call_failed" in t.findings
        assert "prompt could not be assembled" in t.candidate_error


async def test_progress_never_walks_backwards_under_concurrency(
    db_session: Session, test_user: User
) -> None:
    """Regression: the counter advanced under a lock but committed outside it,
    so a worker holding a lower count could land last."""
    run_id = _make_run(db_session, test_user.id, samples=8)
    result = _result(text="ok")
    patches = _patched_run(samples=_samples(8), call_side_effect=lambda *a, **k: result)
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=4)

    db_session.expire_all()
    run = db_session.get(ComparisonRun, run_id)
    assert run is not None
    assert run.progress_completed == 8


async def test_run_never_executes_a_tool(db_session: Session, test_user: User) -> None:
    """The safety property the whole design rests on.

    A replay reports what a model *would* do. If a tool ever ran, a
    comparison would text real customers and mutate real job records. The
    tripwire covers both the read the replay is allowed to continue through
    and the write it is not.
    """
    executed: list[str] = []

    async def tripwire(**_kwargs: object) -> ToolResult:
        executed.append("called")
        raise AssertionError("a tool was executed during a comparison run")

    tools = {"lookup": _lookup_tool(tripwire), "add_note": _note_tool(tripwire)}

    run_id = _make_run(db_session, test_user.id, samples=2)
    acting = _result(
        tools=[
            ToolCall(name="lookup", arguments={"q": "a"}),
            ToolCall(name="add_note", arguments={"work_order_id": "1", "body": "x"}),
        ]
    )
    patches = _patched_run(
        samples=_samples(2),
        call_side_effect=lambda *a, **k: acting,
        tools_by_name=tools,
    )
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=1)

    assert executed == []


async def test_a_replayed_lookup_is_fed_from_the_record_not_executed(
    db_session: Session, test_user: User
) -> None:
    """The continuation is the one place a replay produces a tool result.

    It produces it by copying a string out of the transcript. The tripwire
    proves no tool object was invoked to get it, and the stored row proves
    the string came from the recorded call.
    """
    executed: list[str] = []

    async def tripwire(**_kwargs: object) -> ToolResult:
        executed.append("called")
        raise AssertionError("a tool was executed during a comparison run")

    run_id = _make_run(db_session, test_user.id, samples=1)
    recorded = (RecordedToolResult(name="lookup", arguments={"q": "a"}, result="id 42"),)
    samples = [
        ReplaySample(
            seq=1,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context="note it on the job",
            production_tool_calls=recorded,
        )
    ]
    seen: list[Any] = []

    async def record(*_args: object, **kwargs: Any) -> ModelCallResult:
        seen.append(kwargs["recorded"])
        result = _result(tools=[ToolCall(name="lookup", arguments={"q": "b"})])
        result.replayed_lookups = list(recorded)
        return result

    a, b, c, d = _patched_run(
        samples=samples,
        call_side_effect=record,
        tools_by_name={"lookup": _lookup_tool(tripwire)},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    assert executed == []
    assert seen == [recorded]
    (row,) = _turns(db_session, run_id)
    assert json.loads(row.candidate_replayed_lookups) == [
        {"name": "lookup", "arguments": {"q": "a"}, "result": "id 42", "is_error": False}
    ]


async def test_each_completed_turn_touches_the_heartbeat(
    db_session: Session, test_user: User
) -> None:
    """The sweep's whole liveness signal.

    Without this write every run goes stale on a timer and the periodic
    sweep marks live runs interrupted, which is the bug the heartbeat was
    added to fix.
    """
    run_id = _make_run(db_session, test_user.id, samples=2)
    db_session.expire_all()
    assert db_session.get(ComparisonRun, run_id).heartbeat_at is None  # type: ignore[union-attr]

    a, b, c, d = _patched_run(samples=_samples(2), call_side_effect=lambda *x, **k: _result("ok"))
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    finished = db_session.get(ComparisonRun, run_id)
    assert finished is not None
    assert finished.heartbeat_at is not None


async def test_a_live_run_survives_another_process_booting(
    db_session: Session, test_user: User
) -> None:
    """A rolling deploy boots the new instance while the old one drains.

    An unconditional sweep marks the draining instance's run interrupted. It
    keeps calling the provider and later overwrites the row, and meanwhile
    both concurrency guards in ``start_run`` key off the active statuses, so
    a duplicate run can start for the same user.
    """
    run_id = _make_run(db_session, test_user.id)
    db_session.execute(
        update(ComparisonRun)
        .where(ComparisonRun.id == run_id)
        .values(status=str(RunStatus.RUNNING), heartbeat_at=datetime.now(UTC))
    )
    db_session.commit()

    await mark_interrupted_runs()

    db_session.expire_all()
    swept = db_session.get(ComparisonRun, run_id)
    assert swept is not None
    assert swept.status == str(RunStatus.RUNNING)


async def test_a_run_whose_process_died_is_still_swept(
    db_session: Session, test_user: User
) -> None:
    """The guard is a staleness check, not an amnesty."""
    run_id = _make_run(db_session, test_user.id)
    db_session.execute(
        update(ComparisonRun)
        .where(ComparisonRun.id == run_id)
        .values(
            status=str(RunStatus.RUNNING),
            heartbeat_at=datetime.now(UTC) - timedelta(hours=2),
        )
    )
    db_session.commit()

    await mark_interrupted_runs()

    db_session.expire_all()
    swept = db_session.get(ComparisonRun, run_id)
    assert swept is not None
    assert swept.status == str(RunStatus.INTERRUPTED)


async def test_a_run_that_never_started_a_turn_falls_back_to_created_at(
    db_session: Session, test_user: User
) -> None:
    """Covers a process that died between the insert and the first turn."""
    run_id = _make_run(db_session, test_user.id)
    db_session.execute(
        update(ComparisonRun)
        .where(ComparisonRun.id == run_id)
        .values(
            status=str(RunStatus.PENDING),
            heartbeat_at=None,
            created_at=datetime.now(UTC) - timedelta(hours=2),
        )
    )
    db_session.commit()

    await mark_interrupted_runs()

    db_session.expire_all()
    swept = db_session.get(ComparisonRun, run_id)
    assert swept is not None
    assert swept.status == str(RunStatus.INTERRUPTED)


async def test_user_with_no_turns_completes_with_a_note(
    db_session: Session, test_user: User
) -> None:
    run_id = _make_run(db_session, test_user.id, samples=10)
    patches = _patched_run(samples=[], call_side_effect=lambda *a, **k: _result())
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    run = db_session.get(ComparisonRun, run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    # A status, not a failure. ``error`` renders in a red banner on the
    # report, and this run did not fail: it had nothing to replay.
    assert run.error == ""
    assert run.summary_json is not None
    assert any("no replayable turns" in note for note in run.summary_json["notes"])


async def test_run_stops_after_consecutive_provider_failures(
    db_session: Session, test_user: User
) -> None:
    """A dead provider must not cost the whole sample budget.

    Every remaining turn would fail identically, so the run stops, keeps the
    turns it wrote, and says why.
    """
    run_id = _make_run(db_session, test_user.id, samples=20)
    failing = ModelCallResult(provider="anthropic", model="m", error="APIStatusError: 503")

    patches = _patched_run(
        samples=_samples(20),
        call_side_effect=lambda *a, **k: failing,
    )
    with patches[0], patches[1], patches[2], patches[3] as call:
        await execute_run(run_id, concurrency=1)

    run = db_session.get(ComparisonRun, run_id)
    assert run is not None
    assert run.status == str(RunStatus.FAILED)
    assert "consecutive provider failures" in run.error
    assert "503" in run.error

    # The three that failed are kept as evidence; the other seventeen are not
    # attempted, which is the point. One call per turn now, not two.
    assert len(_turns(db_session, run_id)) == MAX_CONSECUTIVE_CALL_FAILURES
    assert call.await_count == MAX_CONSECUTIVE_CALL_FAILURES

    # The report's banner reads the summary, so it has to carry the reason
    # as well as the column.
    summary = run.summary_json
    assert summary is not None
    assert "consecutive provider failures" in summary["notes"][0]


async def test_an_intermittent_failure_does_not_stop_the_run(
    db_session: Session, test_user: User
) -> None:
    """The count is consecutive, so a flaky call in a long run is survivable."""
    run_id = _make_run(db_session, test_user.id, samples=6)
    ok = _result(text="fine")
    bad = ModelCallResult(provider="anthropic", model="m", error="APIStatusError: 429")
    calls = iter([bad, ok, bad, ok, bad, ok])

    patches = _patched_run(samples=_samples(6), call_side_effect=lambda *a, **k: next(calls))
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=1)

    run = db_session.get(ComparisonRun, run_id)
    assert run is not None
    assert run.status == str(RunStatus.COMPLETED)
    assert run.error == ""
    assert len(_turns(db_session, run_id)) == 6
