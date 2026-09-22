"""End-to-end orchestration of one model-comparison run.

Model dispatch is mocked; the run's own machinery is not. What is being
tested is that the run writes its evidence as it goes, survives a turn that
fails, honors cancellation, and never executes a tool.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock, ToolUseBlock
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from backend.app.agent.core import AssembledPrompt
from backend.app.agent.dto import StoredMessage
from backend.app.agent.messages import SystemMessage, UserMessage
from backend.app.agent.observer import PURPOSE_AGENT_MAIN
from backend.app.agent.tools.base import Tool, ToolResult, ToolTags
from backend.app.models import LLMEvalRun, LLMEvalTurnResult, LLMUsageLog, User
from backend.app.services.llm_eval.judge import JudgeOutcome
from backend.app.services.llm_eval.metrics import MIN_TURNS_FOR_VERDICT
from backend.app.services.llm_eval.runner import (
    HARNESS_VERSION,
    MAX_CONSECUTIVE_CALL_FAILURES,
    execute_run,
    mark_interrupted_runs,
)
from backend.app.services.llm_eval.sampling import ReplayFixture, select_samples
from backend.app.services.llm_eval.types import (
    IncumbentSource,
    JudgeSkipReason,
    JudgeVerdict,
    ModelCallResult,
    Recommendation,
    RecordedToolResult,
    ReplaySample,
    RunStatus,
    SafetyFinding,
    Side,
    ToolCall,
)
from backend.app.services.llm_service import LLMTarget


def _make_run(
    db: Session,
    user_id: str,
    *,
    samples: int = 3,
    judge: bool = False,
    baseline_effort: str = "",
    candidate_effort: str = "",
    incumbent_source: IncumbentSource = IncumbentSource.REPLAY,
    candidate_model: str = "candidate",
) -> int:
    run = LLMEvalRun(
        user_id=user_id,
        baseline_reasoning_effort=baseline_effort,
        candidate_reasoning_effort=candidate_effort,
        baseline_provider="anthropic",
        baseline_model="incumbent",
        candidate_provider="anthropic",
        candidate_model=candidate_model,
        incumbent_source=str(incumbent_source),
        judge_provider="anthropic" if judge else "",
        judge_model="incumbent" if judge else "",
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
            historic_reply=f"answer {i}",
            historic_tool_names=["lookup"],
        )
        for i in range(1, count + 1)
    ]


def _assembled() -> AssembledPrompt:
    """A real assembled prompt: the judge renders its history into context."""
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
    assembled: AssembledPrompt | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Patch the run's collaborators, leaving the orchestration real."""
    fixture = ReplayFixture(user=User(id="u"), rows=[])
    fixture.tools_by_name = tools_by_name or {}
    return (
        patch(
            "backend.app.services.llm_eval.runner.build_fixture",
            AsyncMock(return_value=fixture),
        ),
        patch(
            "backend.app.services.llm_eval.runner.select_samples",
            return_value=samples,
        ),
        patch(
            "backend.app.services.llm_eval.runner.assemble_for_sample",
            AsyncMock(return_value=assembled or _assembled()),
        ),
        patch(
            "backend.app.services.llm_eval.runner.call_model",
            AsyncMock(side_effect=call_side_effect),
        ),
    )


class _LookupParams(BaseModel):
    q: str


def _lookup_tool(function: Callable[..., Awaitable[ToolResult]] | None = None) -> Tool:
    """A registered, non-mutating tool the models are allowed to call.

    Runs that call a tool need it present in ``tools_by_name``: an
    unregistered name is a genuine safety finding, which would sink the
    recommendation for the wrong reason.
    """

    async def _unused(**_kwargs: object) -> ToolResult:  # pragma: no cover
        raise AssertionError("a tool was executed during an evaluation")

    return Tool(
        name="lookup",
        description="lookup",
        function=function or _unused,
        params_model=_LookupParams,
        tags={ToolTags.READ_ONLY},
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


async def test_run_writes_a_turn_row_per_sample_and_completes(
    db_session: Session, test_user: User
) -> None:
    run_id = _make_run(db_session, test_user.id, samples=3)
    agreeing = _result(tools=[ToolCall(name="lookup", arguments={"q": "a"})])

    patches = _patched_run(
        samples=_samples(3),
        call_side_effect=lambda *a, **k: agreeing,
        tools_by_name={"lookup": _lookup_tool()},
    )
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=2)

    db_session.expire_all()
    run = db_session.get(LLMEvalRun, run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert run.progress_completed == 3
    assert run.progress_total == 3
    assert run.summary_json is not None
    assert run.summary_json["turns_completed"] == 3
    # Three matched turns is well under the minimum for a verdict, and a
    # short run must never read as permission to switch.
    assert run.recommendation == Recommendation.INCONCLUSIVE

    turns = (
        db_session.execute(select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id))
        .scalars()
        .all()
    )
    assert len(turns) == 3
    assert {t.message_seq for t in turns} == {1, 2, 3}
    stored = json.loads(turns[0].baseline_tool_calls)
    assert stored[0]["name"] == "lookup"


async def test_each_side_is_called_with_its_own_reasoning_effort(
    db_session: Session, test_user: User
) -> None:
    """The run's two recorded efforts must reach the two calls separately.

    Holding both sides to one value measures the value rather than the
    candidate, and an endpoint that spells reasoning differently can refuse
    the incumbent's spelling outright.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, baseline_effort="high", candidate_effort="none"
    )
    seen: list[tuple[str, str]] = []

    async def record(*_args: object, **kwargs: Any) -> ModelCallResult:
        target: LLMTarget = kwargs["target"]
        effort: str = kwargs["reasoning_effort"]
        seen.append((target.model, effort))
        return _result(text="ok")

    a, b, c, d = _patched_run(samples=_samples(1), call_side_effect=record)
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    assert sorted(seen) == [("candidate", "none"), ("incumbent", "high")]


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
        patch("backend.app.services.llm_eval.runner.call_model", flaky),
    ):
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    run = db_session.get(LLMEvalRun, run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    turns = (
        db_session.execute(select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id))
        .scalars()
        .all()
    )
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
            update(LLMEvalRun)
            .where(LLMEvalRun.id == run_id)
            .values(status=str(RunStatus.CANCELLED))
        )
        db_session.commit()
        return _result(text="ok")

    patches = _patched_run(samples=_samples(4), call_side_effect=None)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("backend.app.services.llm_eval.runner.call_model", cancel_after_first),
        # The watcher caches the flag for a second so a run does not open a
        # session per turn to poll one column. Turns here finish in
        # microseconds, so without this the cancellation lands inside the
        # cache window and the run completes normally.
        patch("backend.app.services.llm_eval.runner._CANCEL_POLL_SECONDS", 0),
        pytest.raises(asyncio.CancelledError),
    ):
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    turns = (
        db_session.execute(select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id))
        .scalars()
        .all()
    )
    assert len(turns) < 4


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
        db_session.execute(delete(LLMEvalRun).where(LLMEvalRun.id == run_id))
        db_session.commit()
        return _result(text="ok")

    patches = _patched_run(samples=_samples(1), call_side_effect=None)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("backend.app.services.llm_eval.runner.call_model", delete_the_run),
    ):
        # No raise: the assertion is that nothing propagates.
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    assert db_session.get(LLMEvalRun, run_id) is None
    assert (
        db_session.execute(
            select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id)
        ).first()
        is None
    )


async def test_a_turn_that_cannot_be_replayed_is_marked_not_compared(
    db_session: Session, test_user: User
) -> None:
    """A turn that fails to assemble must keep a failure marker.

    Otherwise the hardest failure is the least visible one: it carries a
    fabricated agreement value and sorts below every turn that did run.
    """
    run_id = _make_run(db_session, test_user.id, samples=2)

    async def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("prompt could not be assembled")

    patches = _patched_run(samples=_samples(2), call_side_effect=None)
    with (
        patches[0],
        patches[1],
        patch("backend.app.services.llm_eval.runner.assemble_for_sample", explode),
        patches[3],
    ):
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    turns = (
        db_session.execute(select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id))
        .scalars()
        .all()
    )
    assert len(turns) == 2
    for t in turns:
        assert t.agreement == "not_compared"
        assert "call_failed" in t.safety_issues
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
    run = db_session.get(LLMEvalRun, run_id)
    assert run is not None
    assert run.progress_completed == 8


async def test_run_never_executes_a_tool(db_session: Session, test_user: User) -> None:
    """The safety property the whole design rests on.

    A replay decides what a model *would* do. If a tool ever ran, an
    evaluation would text real customers and mutate real job records.
    """
    executed: list[str] = []

    async def tripwire(**_kwargs: object) -> ToolResult:
        executed.append("called")
        raise AssertionError("a tool was executed during an evaluation")

    tool = _lookup_tool(tripwire)

    run_id = _make_run(db_session, test_user.id, samples=2)
    acting = _result(tools=[ToolCall(name="lookup", arguments={"q": "anything"})])
    patches = _patched_run(
        samples=_samples(2),
        call_side_effect=lambda *a, **k: acting,
        tools_by_name={"lookup": tool},
    )
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=1)

    assert executed == []


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
    assert db_session.get(LLMEvalRun, run_id).heartbeat_at is None  # type: ignore[union-attr]

    a, b, c, d = _patched_run(samples=_samples(2), call_side_effect=lambda *x, **k: _result("ok"))
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    finished = db_session.get(LLMEvalRun, run_id)
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
        update(LLMEvalRun)
        .where(LLMEvalRun.id == run_id)
        .values(status=str(RunStatus.RUNNING), heartbeat_at=datetime.now(UTC))
    )
    db_session.commit()

    await mark_interrupted_runs()

    db_session.expire_all()
    swept = db_session.get(LLMEvalRun, run_id)
    assert swept is not None
    assert swept.status == str(RunStatus.RUNNING)


async def test_a_run_whose_process_died_is_still_swept(
    db_session: Session, test_user: User
) -> None:
    """The guard is a staleness check, not an amnesty."""
    run_id = _make_run(db_session, test_user.id)
    db_session.execute(
        update(LLMEvalRun)
        .where(LLMEvalRun.id == run_id)
        .values(
            status=str(RunStatus.RUNNING),
            heartbeat_at=datetime.now(UTC) - timedelta(hours=2),
        )
    )
    db_session.commit()

    await mark_interrupted_runs()

    db_session.expire_all()
    swept = db_session.get(LLMEvalRun, run_id)
    assert swept is not None
    assert swept.status == str(RunStatus.INTERRUPTED)


async def test_a_run_that_never_started_a_turn_falls_back_to_created_at(
    db_session: Session, test_user: User
) -> None:
    """Covers a process that died between the insert and the first turn."""
    run_id = _make_run(db_session, test_user.id)
    db_session.execute(
        update(LLMEvalRun)
        .where(LLMEvalRun.id == run_id)
        .values(
            status=str(RunStatus.PENDING),
            heartbeat_at=None,
            created_at=datetime.now(UTC) - timedelta(hours=2),
        )
    )
    db_session.commit()

    await mark_interrupted_runs()

    db_session.expire_all()
    swept = db_session.get(LLMEvalRun, run_id)
    assert swept is not None
    assert swept.status == str(RunStatus.INTERRUPTED)


async def test_user_with_no_turns_completes_as_inconclusive(
    db_session: Session, test_user: User
) -> None:
    run_id = _make_run(db_session, test_user.id, samples=10)
    patches = _patched_run(samples=[], call_side_effect=lambda *a, **k: _result())
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=1)

    db_session.expire_all()
    run = db_session.get(LLMEvalRun, run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert run.recommendation == Recommendation.INCONCLUSIVE
    # A status, not a failure. ``error`` renders in a red banner on the
    # report, and this run did not fail: it had nothing to replay.
    assert run.error == ""
    assert run.summary_json is not None
    assert any("no replayable turns" in r for r in run.summary_json["reasons"])


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

    run = db_session.get(LLMEvalRun, run_id)
    assert run is not None
    assert run.status == str(RunStatus.FAILED)
    assert "consecutive provider failures" in run.error
    assert "503" in run.error

    turns = (
        db_session.execute(select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id))
        .scalars()
        .all()
    )
    # The three that failed are kept as evidence; the other seventeen are not
    # attempted, which is the point.
    assert len(turns) == MAX_CONSECUTIVE_CALL_FAILURES
    assert call.await_count == MAX_CONSECUTIVE_CALL_FAILURES * 2

    # A run that stopped early cannot endorse a switch, and the banner reads
    # the summary rather than the column, so both have to say so.
    assert run.recommendation == str(Recommendation.INCONCLUSIVE)
    summary = run.summary_json
    assert summary is not None
    assert summary["recommendation"] == str(Recommendation.INCONCLUSIVE)
    assert "consecutive provider failures" in summary["reasons"][0]


async def test_an_intermittent_failure_does_not_stop_the_run(
    db_session: Session, test_user: User
) -> None:
    """The count is consecutive, so a flaky call in a long run is survivable."""
    run_id = _make_run(db_session, test_user.id, samples=6)
    ok = _result(text="fine")
    bad = ModelCallResult(provider="anthropic", model="m", error="APIStatusError: 429")
    # Two calls per turn: alternate whole turns rather than individual calls.
    outcomes = [bad, bad, ok, ok, bad, bad, ok, ok, bad, bad, ok, ok]
    calls = iter(outcomes)

    patches = _patched_run(samples=_samples(6), call_side_effect=lambda *a, **k: next(calls))
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=1)

    run = db_session.get(LLMEvalRun, run_id)
    assert run is not None
    assert run.status == str(RunStatus.COMPLETED)
    assert run.error == ""
    turns = (
        db_session.execute(select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id))
        .scalars()
        .all()
    )
    assert len(turns) == 6


# ---------------------------------------------------------------------------
# The judge gate is blocking findings, not any finding
# ---------------------------------------------------------------------------


async def test_a_non_blocking_finding_does_not_suppress_the_judge(
    db_session: Session, test_user: User
) -> None:
    """A retired tool name in the fixture is not a reason to skip adjudication.

    Both models copy the name out of the replayed history, only the candidate
    is inspected, and the finding is explicitly non-blocking. Skipping the
    judge on it left those turns at the top of the report wearing a red badge
    with nothing underneath to explain them.
    """
    run_id = _make_run(db_session, test_user.id, samples=1, judge=True)
    # ``retired`` is absent from the schema but present in the turn's history,
    # so it lands as UNRESOLVED_TOOL_NAME rather than UNKNOWN_TOOL.
    samples = [
        ReplaySample(
            seq=1,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context="price these",
            historic_tool_names=["retired"],
        )
    ]
    baseline = _result(tools=[ToolCall(name="lookup", arguments={"q": "a"})])
    candidate = _result(tools=[ToolCall(name="retired", arguments={})])
    calls = iter([baseline, candidate])

    judge = AsyncMock(return_value=JudgeOutcome(JudgeVerdict.CANDIDATE_WORSE, "worse because"))
    patches = _patched_run(
        samples=samples,
        call_side_effect=lambda *a, **k: next(calls),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch("backend.app.services.llm_eval.runner.judge_turn", judge),
    ):
        await execute_run(run_id, concurrency=1)

    row = db_session.execute(
        select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id)
    ).scalar_one()
    findings = [i["finding"] for i in json.loads(row.safety_issues)]
    assert findings == [str(SafetyFinding.UNRESOLVED_TOOL_NAME)]
    assert judge.await_count == 1
    assert row.judge_verdict == str(JudgeVerdict.CANDIDATE_WORSE)


async def test_a_turn_with_a_safety_finding_is_still_judged(
    db_session: Session, test_user: User
) -> None:
    """Regression: a turn with a finding skipped the judge as already disqualified.

    One finding no longer decides a run, so the judge's preference and its
    unsafe flags on that turn are evidence the recommendation still needs.
    The flags land as findings on the side the judge named, the incumbent's
    included.
    """
    run_id = _make_run(db_session, test_user.id, samples=1, judge=True)
    baseline = _result(tools=[ToolCall(name="lookup", arguments={"q": "a"})])
    candidate = _result(tools=[ToolCall(name="invented", arguments={})])
    calls = iter([baseline, candidate])

    judge = AsyncMock(
        return_value=JudgeOutcome(
            JudgeVerdict.CANDIDATE_WORSE, "invented a tool", frozenset({Side.BASELINE})
        )
    )
    patches = _patched_run(
        samples=_samples(1),
        call_side_effect=lambda *a, **k: next(calls),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch("backend.app.services.llm_eval.runner.judge_turn", judge),
    ):
        await execute_run(run_id, concurrency=1)

    row = db_session.execute(
        select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id)
    ).scalar_one()
    assert judge.await_count == 1
    assert row.judge_verdict == str(JudgeVerdict.CANDIDATE_WORSE)
    assert [(i["finding"], i["side"]) for i in json.loads(row.safety_issues)] == [
        (str(SafetyFinding.UNKNOWN_TOOL), "candidate"),
        (str(SafetyFinding.JUDGED_UNSAFE), "baseline"),
    ]


async def test_the_incumbents_findings_are_recorded_too(
    db_session: Session, test_user: User
) -> None:
    """Regression: ``check_safety`` only ever inspected the candidate."""
    run_id = _make_run(db_session, test_user.id, samples=1)
    baseline = _result(tools=[ToolCall(name="invented", arguments={})])
    candidate = _result(tools=[ToolCall(name="lookup", arguments={"q": "a"})])
    calls = iter([baseline, candidate])
    patches = _patched_run(
        samples=_samples(1),
        call_side_effect=lambda *a, **k: next(calls),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=1)

    row = db_session.execute(
        select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id)
    ).scalar_one()
    assert [(i["finding"], i["side"]) for i in json.loads(row.safety_issues)] == [
        (str(SafetyFinding.UNKNOWN_TOOL), "baseline")
    ]
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["baseline_safety_counts"] == {str(SafetyFinding.UNKNOWN_TOOL): 1}
    assert run.summary_json["safety_comparison"]["baseline_only"] == 1


async def test_the_summary_records_why_each_turn_went_unjudged(
    db_session: Session, test_user: User
) -> None:
    run_id = _make_run(db_session, test_user.id, samples=2, judge=True)
    agreeing = _result(tools=[ToolCall(name="lookup", arguments={"q": "a"})])

    patches = _patched_run(
        samples=_samples(2),
        call_side_effect=lambda *a, **k: agreeing,
        tools_by_name={"lookup": _lookup_tool()},
    )
    with patches[0], patches[1], patches[2], patches[3]:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    summary = run.summary_json
    assert summary is not None
    assert summary["judge_skip_counts"] == {str(JudgeSkipReason.IDENTICAL): 2}


# ---------------------------------------------------------------------------
# Lookups replayed from the live turn, and the judge's context
# ---------------------------------------------------------------------------


async def test_each_side_is_offered_the_live_turns_recorded_lookups(
    db_session: Session, test_user: User
) -> None:
    """The replay can only continue past a lookup the live turn made."""
    run_id = _make_run(db_session, test_user.id, samples=1)
    recorded = (RecordedToolResult(name="lookup", arguments={"q": "a"}, result="id 42"),)
    samples = [
        ReplaySample(
            seq=1,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context="note it on the job",
            historic_tool_names=["lookup"],
            historic_tool_results=recorded,
        )
    ]
    seen: list[Any] = []

    async def record(*_args: object, **kwargs: Any) -> ModelCallResult:
        seen.append(kwargs["recorded"])
        result = _result(tools=[ToolCall(name="lookup", arguments={"q": "b"})])
        result.replayed_lookups = list(recorded)
        return result

    a, b, c, d = _patched_run(
        samples=samples, call_side_effect=record, tools_by_name={"lookup": _lookup_tool()}
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    assert seen == [recorded, recorded]
    row = db_session.execute(
        select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id)
    ).scalar_one()
    stored = json.loads(row.candidate_replayed_lookups)
    assert stored == [
        {"name": "lookup", "arguments": {"q": "a"}, "result": "id 42", "is_error": False}
    ]


async def test_the_judge_is_given_the_conversation_and_the_turns_clock(
    db_session: Session, test_user: User
) -> None:
    """Regression: the judge saw only the user's latest message.

    It called correct answers made up when the fact came from an earlier
    turn, and resolved relative dates against no date at all.
    """
    run_id = _make_run(db_session, test_user.id, samples=1, judge=True)
    baseline = _result(tools=[ToolCall(name="lookup", arguments={"q": "a"})])
    candidate = _result(text="done")
    calls = iter([baseline, candidate])
    judge = AsyncMock(return_value=JudgeOutcome(JudgeVerdict.EQUIVALENT, ""))
    patches = _patched_run(
        samples=_samples(1),
        call_side_effect=lambda *a, **k: next(calls),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch("backend.app.services.llm_eval.runner.judge_turn", judge),
    ):
        await execute_run(run_id, concurrency=1)

    context = judge.await_args_list[-1].kwargs["context"]
    assert "earlier ask" in context.transcript
    assert "current ask" not in context.transcript
    assert "2026-05-01" in context.current_time


# ---------------------------------------------------------------------------
# Calibrating divergence with the incumbent against itself
# ---------------------------------------------------------------------------


def _calibration_run(
    db: Session,
    user_id: str,
    *,
    divergence: float,
    version: int,
    turns: int = 100,
    completed_at: datetime | None = None,
) -> None:
    db.add(
        LLMEvalRun(
            user_id=user_id,
            baseline_provider="anthropic",
            baseline_model="incumbent",
            candidate_provider="anthropic",
            candidate_model="incumbent",
            requested_samples=100,
            status=str(RunStatus.COMPLETED),
            completed_at=completed_at or datetime.now(UTC),
            summary_json={
                "harness_version": version,
                "divergence_rate": divergence,
                "turns_completed": turns,
            },
        )
    )
    db.commit()


async def test_a_calibration_run_sets_the_divergence_noise_floor(
    db_session: Session, test_user: User
) -> None:
    _calibration_run(db_session, test_user.id, divergence=0.9, version=HARNESS_VERSION - 1)
    _calibration_run(db_session, test_user.id, divergence=0.38, version=HARNESS_VERSION)
    run_id = _make_run(db_session, test_user.id, samples=1)
    a, b, c, d = _patched_run(samples=_samples(1), call_side_effect=lambda *a, **k: _result("ok"))
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["divergence_noise_floor"] == 0.38
    assert run.summary_json["divergence_threshold"] == 0.48
    assert run.summary_json["harness_version"] == HARNESS_VERSION


async def test_a_calibration_run_too_short_for_a_verdict_is_ignored(
    db_session: Session, test_user: User
) -> None:
    now = datetime.now(UTC)
    _calibration_run(
        db_session,
        test_user.id,
        divergence=0.38,
        version=HARNESS_VERSION,
        completed_at=now - timedelta(days=1),
    )
    _calibration_run(
        db_session, test_user.id, divergence=0.0, version=HARNESS_VERSION, turns=3, completed_at=now
    )
    run_id = _make_run(db_session, test_user.id, samples=1)
    a, b, c, d = _patched_run(samples=_samples(1), call_side_effect=lambda *a, **k: _result("ok"))
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["divergence_noise_floor"] == 0.38


async def test_only_a_short_calibration_run_leaves_divergence_uncalibrated(
    db_session: Session, test_user: User
) -> None:
    _calibration_run(db_session, test_user.id, divergence=0.1, version=HARNESS_VERSION, turns=3)
    run_id = _make_run(db_session, test_user.id, samples=1)
    a, b, c, d = _patched_run(samples=_samples(1), call_side_effect=lambda *a, **k: _result("ok"))
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["divergence_noise_floor"] is None


async def test_a_self_comparison_run_says_it_is_the_calibration(
    db_session: Session, test_user: User
) -> None:
    run = LLMEvalRun(
        user_id=test_user.id,
        baseline_provider="anthropic",
        baseline_model="incumbent",
        candidate_provider="anthropic",
        candidate_model="incumbent",
        requested_samples=1,
        status=str(RunStatus.PENDING),
    )
    db_session.add(run)
    db_session.commit()
    a, b, c, d = _patched_run(samples=_samples(1), call_side_effect=lambda *a, **k: _result("ok"))
    with a, b, c, d:
        await execute_run(run.id, concurrency=1)

    db_session.expire_all()
    stored = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run.id)).scalar_one()
    assert stored.summary_json is not None
    assert stored.summary_json["divergence_noise_floor"] is None
    assert any("against itself" in w for w in stored.summary_json["warnings"])


# ---------------------------------------------------------------------------
# Historic mode: the incumbent's decision, read rather than bought
# ---------------------------------------------------------------------------


def _recording_dispatch(seen: list[str]) -> Callable[..., Awaitable[ModelCallResult]]:
    """A ``call_model`` stand-in that logs which side asked for a decision.

    The point of the mode is not paying for the incumbent twice, so the
    assertion that matters is which models were called and how often.
    """

    async def dispatch(*_args: object, **kwargs: Any) -> ModelCallResult:
        target: LLMTarget = kwargs["target"]
        seen.append(target.model)
        return _result(text=f"{target.model} answered")

    return dispatch


def _turns_of(db: Session, run_id: int) -> list[LLMEvalTurnResult]:
    db.expire_all()
    rows = (
        db.execute(select(LLMEvalTurnResult).where(LLMEvalTurnResult.run_id == run_id))
        .scalars()
        .all()
    )
    return sorted(rows, key=lambda t: t.message_seq)


async def test_historic_mode_never_calls_the_incumbent(
    db_session: Session, test_user: User
) -> None:
    """Zero incumbent calls, and the decision comes from the recorded turn."""
    run_id = _make_run(
        db_session, test_user.id, samples=2, incumbent_source=IncumbentSource.HISTORIC
    )
    seen: list[str] = []
    a, b, c, _ = _patched_run(
        samples=_historic_samples(2),
        call_side_effect=None,
        tools_by_name={"lookup": _lookup_tool()},
    )
    with (
        a,
        b,
        c,
        patch("backend.app.services.llm_eval.runner.call_model", _recording_dispatch(seen)),
    ):
        await execute_run(run_id, concurrency=1)

    assert seen == ["candidate", "candidate"]
    assert [t.baseline_source for t in _turns_of(db_session, run_id)] == ["historic", "historic"]
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["incumbent_source"] == "historic"
    assert run.summary_json["incumbent_source_counts"] == {"historic": 2}


def _historic_samples(count: int) -> list[ReplaySample]:
    """Turns whose first decision was a single recorded lookup."""
    return [
        ReplaySample(
            seq=i,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context=f"ask {i}",
            historic_reply=f"here is what I found for {i}",
            historic_tool_names=["lookup"],
            historic_first_calls=(ToolCall(name="lookup", arguments={"q": f"a{i}"}),),
            historic_tool_results=(
                RecordedToolResult(name="lookup", arguments={"q": f"a{i}"}, result="found"),
            ),
        )
        for i in range(1, count + 1)
    ]


async def test_the_historic_side_is_the_turns_first_decision_not_its_reply(
    db_session: Session, test_user: User
) -> None:
    """The incumbent side must be the opening move, with its arguments.

    ``historic_reply`` is the prose the user saw after every tool round had
    run. Scoring it against the candidate's first decision compares a
    finished message with an opening move, and reads every lookup-then-act
    turn as the candidate acting where the incumbent only talked.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, incumbent_source=IncumbentSource.HISTORIC
    )
    a, b, c, d = _patched_run(
        samples=_historic_samples(1),
        call_side_effect=lambda *a, **k: _result(
            tools=[ToolCall(name="lookup", arguments={"q": "a1"})]
        ),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    turn = _turns_of(db_session, run_id)[0]
    assert json.loads(turn.baseline_tool_calls) == [{"name": "lookup", "arguments": {"q": "a1"}}]
    # The reply is kept as context for a human, not as the scored decision.
    assert turn.baseline_text == ""
    assert turn.historic_reply == "here is what I found for 1"
    # Same call, same arguments, so the candidate matched it.
    assert turn.agreement == "identical"


async def test_a_turn_that_only_replied_carries_its_prose_as_the_decision(
    db_session: Session, test_user: User
) -> None:
    """When the turn called nothing, its first decision was what it said."""
    run_id = _make_run(
        db_session, test_user.id, samples=1, incumbent_source=IncumbentSource.HISTORIC
    )
    sample = ReplaySample(
        seq=1,
        timestamp="2026-05-01T12:00:00+00:00",
        message_context="just checking in",
        historic_reply="all good here",
        historic_decision_text="all good here",
    )
    a, b, c, d = _patched_run(
        samples=[sample], call_side_effect=lambda *a, **k: _result(text="all good here")
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    turn = _turns_of(db_session, run_id)[0]
    assert turn.baseline_text == "all good here"
    assert json.loads(turn.baseline_tool_calls) == []
    assert turn.agreement == "both_replied"


async def test_an_unreconstructable_turn_leaves_the_comparison(
    db_session: Session, test_user: User
) -> None:
    """A turn with no readable decision is dropped, not guessed at.

    Reading it as "the agent did nothing" is the dangerous default: that is
    what exempts a candidate's write from ``unrequested_mutation`` on a turn
    production also wrote on.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=2, incumbent_source=IncumbentSource.HISTORIC
    )
    samples = _historic_samples(2)
    broken = ReplaySample(
        seq=2,
        timestamp="2026-05-01T12:00:00+00:00",
        message_context="ask 2",
        historic_decision_available=False,
    )
    a, b, c, d = _patched_run(
        samples=[samples[0], broken],
        call_side_effect=lambda *a, **k: _result(
            tools=[ToolCall(name="lookup", arguments={"q": "a1"})]
        ),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    turns = _turns_of(db_session, run_id)
    assert [t.baseline_source for t in turns] == ["historic", "unavailable"]
    assert turns[1].agreement == "not_compared"
    assert turns[1].judge_verdict == str(JudgeVerdict.NOT_JUDGED)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.baseline_turns_unavailable == 1
    summary = run.summary_json
    assert summary is not None
    assert summary["turns_incumbent_unavailable"] == 1
    # Neither compared nor failed: it is in no rate's denominator.
    assert summary["turns_completed"] == 1
    assert summary["turns_failed"] == 0
    assert summary["incumbent_source_counts"] == {"historic": 1, "unavailable": 1}
    assert any("no reconstructable incumbent decision" in w for w in summary["warnings"])


async def test_historic_mode_reports_the_unmeasured_incumbent_as_unavailable(
    db_session: Session, test_user: User
) -> None:
    """Nothing the incumbent was not asked may be reported as a zero.

    Its safety record, its tokens and its cost were never measured, so the
    safety comparison is marked incomparable and the cost unpriced rather
    than $0.0000, which a reader would take for a free model.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=2, incumbent_source=IncumbentSource.HISTORIC
    )
    a, b, c, d = _patched_run(
        samples=_historic_samples(2),
        call_side_effect=lambda *a, **k: _result(text="candidate answered"),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    summary = run.summary_json
    assert summary is not None
    assert summary["safety_comparison"]["comparable"] is False
    # Not the fabricated-ID tier. That check is deterministic against the
    # prompt text plus each side's own lookups, so the historic side really
    # is measured and its zero is a zero.
    assert summary["fabricated_id_comparison"]["comparable"] is True
    assert summary["baseline"]["pricing_available"] is False
    assert summary["baseline"]["pricing_unknown_reason"] == "not_replayed"
    assert summary["baseline"]["input_tokens"] == 0
    assert any("was not replayed" in warning for warning in summary["warnings"])
    # And no second warning blaming a missing price list for the same zero.
    assert not any("No pricing data for the incumbent" in w for w in summary["warnings"])


async def test_historic_mode_says_which_model_actually_answered(
    db_session: Session, test_user: User
) -> None:
    """A historic run compares against whoever answered, not whoever it names.

    The transcript does not record that per turn, so the usage log over the
    sampled window is the evidence. Saying nothing would present the run as
    a comparison against the configured incumbent, which it may not be.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, incumbent_source=IncumbentSource.HISTORIC
    )
    db_session.add_all(
        [
            LLMUsageLog(
                user_id=test_user.id,
                provider="anthropic",
                model="incumbent",
                purpose=PURPOSE_AGENT_MAIN,
                created_at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
            ),
            LLMUsageLog(
                user_id=test_user.id,
                provider="anthropic",
                model="a-model-nobody-named",
                purpose=PURPOSE_AGENT_MAIN,
                created_at=datetime(2026, 5, 1, 12, 5, tzinfo=UTC),
            ),
        ]
    )
    db_session.commit()

    a, b, c, d = _patched_run(
        samples=_historic_samples(1),
        call_side_effect=lambda *a, **k: _result(text="ok"),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.historic_other_config_calls == 1
    assert run.summary_json is not None
    assert any("a-model-nobody-named" in warning for warning in run.summary_json["warnings"])


async def test_historic_mode_never_clears_a_candidate_outright(
    db_session: Session, test_user: User
) -> None:
    """A clean run that never asked the incumbent is not permission to switch.

    Long enough for a verdict, no findings on either side, nothing diverging:
    in replay mode this is ``safe_to_switch``. Here the claim rests on a
    comparison that was never made, so the verdict stops one step short and
    says why.
    """
    turns = MIN_TURNS_FOR_VERDICT + 2
    samples = [
        ReplaySample(
            seq=i,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context=f"ask {i}",
            historic_reply="same answer",
        )
        for i in range(1, turns + 1)
    ]
    run_id = _make_run(
        db_session, test_user.id, samples=turns, incumbent_source=IncumbentSource.HISTORIC
    )
    a, b, c, d = _patched_run(
        samples=samples, call_side_effect=lambda *a, **k: _result(text="same answer")
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["divergence_rate"] == 0.0
    assert run.recommendation == Recommendation.SWITCH_WITH_MONITORING
    assert any("was not replayed" in reason for reason in run.summary_json["reasons"])


async def test_a_historic_run_is_not_a_divergence_calibration(
    db_session: Session, test_user: User
) -> None:
    """Both columns naming the incumbent is not enough to be a noise floor.

    A historic run's divergence is the candidate's distance from a months-old
    transcript, not the incumbent's disagreement with itself, and adopting it
    as a floor would excuse a genuinely divergent candidate later.
    """
    _calibration_run(db_session, test_user.id, divergence=0.38, version=HARNESS_VERSION)
    db_session.execute(update(LLMEvalRun).values(incumbent_source=str(IncumbentSource.HISTORIC)))
    db_session.commit()

    run_id = _make_run(db_session, test_user.id, samples=1)
    a, b, c, d = _patched_run(samples=_samples(1), call_side_effect=lambda *a, **k: _result("ok"))
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["divergence_noise_floor"] is None


async def test_a_replay_run_still_calls_both_sides(db_session: Session, test_user: User) -> None:
    """The escape hatch has to still work, and still record what it was."""
    run_id = _make_run(db_session, test_user.id, samples=2, incumbent_source=IncumbentSource.REPLAY)
    seen: list[str] = []
    a, b, c, _ = _patched_run(samples=_samples(2), call_side_effect=None)
    with (
        a,
        b,
        c,
        patch("backend.app.services.llm_eval.runner.call_model", _recording_dispatch(seen)),
    ):
        await execute_run(run_id, concurrency=1)

    assert sorted(seen) == ["candidate", "candidate", "incumbent", "incumbent"]
    assert [t.baseline_source for t in _turns_of(db_session, run_id)] == ["live", "live"]
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.baseline_turns_unavailable == 0
    assert run.summary_json is not None
    assert run.summary_json["incumbent_source"] == "replay"
    assert run.summary_json["incumbent_source_counts"] == {"live": 2}


# ---------------------------------------------------------------------------
# Historic mode against the real round loop
# ---------------------------------------------------------------------------
#
# The tests above stub ``call_model``, so they never exercise the thing that
# broke: the replay advances the candidate past a lookup the live turn also
# made and scores what comes next, while the historic side was read as the
# first recorded call. On a lookup-then-act turn those are different rounds,
# and a candidate that reproduced production exactly came out divergent.


class _FindParams(BaseModel):
    q: str


class _SendParams(BaseModel):
    invoice_id: str


async def _unused_tool(**_kwargs: object) -> ToolResult:  # pragma: no cover
    raise AssertionError("a tool was executed during an evaluation")


_QB_TOOLS = {
    "qb_find": Tool(
        name="qb_find",
        description="find",
        function=_unused_tool,
        params_model=_FindParams,
        tags={ToolTags.READ_ONLY},
    ),
    # Untagged, so mutating: never fed back, whichever side asks for it.
    "qb_send": Tool(
        name="qb_send", description="send", function=_unused_tool, params_model=_SendParams
    ),
}

_FIND_CALL = {"name": "qb_find", "input": {"q": "Acme Plumbing"}}
_SEND_CALL = {"name": "qb_send", "input": {"invoice_id": "1186"}}
_FOUND = "invoice 1186 for Acme Plumbing"


def _provider_response(*, text: str = "", tool: dict[str, Any] | None = None) -> MessageResponse:
    """What the provider hands back, parsed by the real dispatch path."""
    content: list[Any] = []
    if text:
        content.append(TextBlock(type="text", text=text))
    if tool is not None:
        content.append(
            ToolUseBlock(type="tool_use", id="toolu_1", name=tool["name"], input=tool["input"])
        )
    return MessageResponse(
        id="msg_1",
        content=content,
        model="candidate",
        role="assistant",
        stop_reason="tool_use" if tool else "end_turn",
        type="message",
        usage=MessageUsage(input_tokens=100, output_tokens=10),
    )


def _second_round(messages: list[dict[str, Any]]) -> bool:
    """Whether this request carries a replayed tool result, i.e. round two."""
    return any(
        isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
        for m in messages
    )


def _lookup_then(final: dict[str, Any]) -> Callable[..., Awaitable[MessageResponse]]:
    """A model that looks the invoice up, then does *final* with what it read."""

    async def dispatch(**kwargs: Any) -> MessageResponse:
        if _second_round(kwargs["messages"]):
            return _provider_response(**final)
        return _provider_response(tool=_FIND_CALL)

    return dispatch


def _recorded_turn(seq: int, *, interactions: list[dict[str, Any]], reply: str) -> ReplaySample:
    """One sampled turn, reconstructed from rows the way a real run does.

    Goes through ``select_samples`` rather than building the sample by hand,
    so the reconstruction under test is the one the run would use.
    """
    stamp = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    rows = [
        StoredMessage(
            seq=seq,
            direction="inbound",
            body="invoice Acme for the stalls",
            processed_context="invoice Acme for the stalls",
            timestamp=stamp.isoformat(),
        ),
        StoredMessage(
            seq=seq + 1,
            direction="outbound",
            body=reply,
            llm_reply_text=reply,
            tool_interactions_json=json.dumps(interactions),
            timestamp=(stamp + timedelta(seconds=9)).isoformat(),
        ),
    ]
    fixture = ReplayFixture(user=User(id="u"), rows=rows)
    fixture.tools_by_name = _QB_TOOLS
    return select_samples(fixture, 1)[0]


def _lookup_then_write_turn(seq: int = 1) -> ReplaySample:
    return _recorded_turn(
        seq,
        interactions=[
            {
                "tool_call_id": "t1",
                "name": "qb_find",
                "args": {"q": "Acme Plumbing"},
                "result": _FOUND,
            },
            {
                "tool_call_id": "t2",
                "name": "qb_send",
                "args": {"invoice_id": "1186"},
                "result": "ok",
            },
        ],
        reply="Sent it.",
    )


def _lookup_then_reply_turn(seq: int) -> ReplaySample:
    return _recorded_turn(
        seq,
        interactions=[
            {
                "tool_call_id": "t1",
                "name": "qb_find",
                "args": {"q": "Acme Plumbing"},
                "result": _FOUND,
            },
        ],
        reply="They owe 1186.",
    )


async def test_an_identical_candidate_on_a_lookup_then_act_turn_agrees(
    db_session: Session, test_user: User
) -> None:
    """The defect this mode shipped with, against the real round loop.

    The candidate looks the invoice up, is fed the result the live turn
    recorded, and sends the same invoice: production's turn, reproduced. The
    two sides must be scored on the same round of it, or the run reports the
    candidate as acting where the incumbent looked something up.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, incumbent_source=IncumbentSource.HISTORIC
    )
    dispatch = AsyncMock(side_effect=_lookup_then({"tool": _SEND_CALL}))
    a, b, c, _ = _patched_run(
        samples=[_lookup_then_write_turn()],
        call_side_effect=None,
        tools_by_name=_QB_TOOLS,
    )
    with a, b, c, patch("backend.app.services.llm_eval.execution.amessages", dispatch):
        await execute_run(run_id, concurrency=1)

    turn = _turns_of(db_session, run_id)[0]
    assert json.loads(turn.candidate_tool_calls) == [
        {"name": "qb_send", "arguments": {"invoice_id": "1186"}}
    ]
    assert json.loads(turn.baseline_tool_calls) == [
        {"name": "qb_send", "arguments": {"invoice_id": "1186"}}
    ]
    assert turn.agreement == "identical"
    # Both sides carry the lookup they made first, which is also what keeps
    # the two responses indistinguishable in the judge prompt.
    assert [item["name"] for item in json.loads(turn.baseline_replayed_lookups)] == ["qb_find"]
    assert [item["name"] for item in json.loads(turn.candidate_replayed_lookups)] == ["qb_find"]
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["agreement_counts"] == {"identical": 1}
    assert run.summary_json["silent_noop_rate"] == 0.0


async def test_an_identical_candidate_on_lookup_then_reply_turns_is_not_a_no_op(
    db_session: Session, test_user: User
) -> None:
    """The reproduction: twelve lookup-then-reply turns, an identical candidate.

    Every one of them came back ``replied_instead_of_acting``, because the
    candidate was scored on the reply it gave after the lookup and the
    incumbent on the lookup itself. That is a 100% silent-no-op rate and a
    ``do_not_switch`` against a candidate that did exactly what production
    did.
    """
    turns = 12
    samples = [_lookup_then_reply_turn(seq) for seq in range(1, turns * 2, 2)]
    run_id = _make_run(
        db_session, test_user.id, samples=turns, incumbent_source=IncumbentSource.HISTORIC
    )
    dispatch = AsyncMock(side_effect=_lookup_then({"text": "They owe 1186."}))
    a, b, c, _ = _patched_run(samples=samples, call_side_effect=None, tools_by_name=_QB_TOOLS)
    with a, b, c, patch("backend.app.services.llm_eval.execution.amessages", dispatch):
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    summary = run.summary_json
    assert summary is not None
    assert summary["agreement_counts"] == {"both_replied": turns}
    assert summary["silent_noop_rate"] == 0.0
    assert summary["divergence_rate"] == 0.0
    assert run.recommendation != Recommendation.DO_NOT_SWITCH


def _silent_noop_samples(turns: int) -> list[ReplaySample]:
    """Turns production answered with a lookup, end to end."""
    return [
        ReplaySample(
            seq=seq,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context=f"ask {seq}",
            historic_reply="looked it up",
            historic_tool_names=["lookup"],
            historic_first_calls=(ToolCall(name="lookup", arguments={"q": f"a{seq}"}),),
        )
        for seq in range(1, turns + 1)
    ]


async def test_historic_mode_rejects_a_candidate_it_really_did_compare(
    db_session: Session, test_user: User
) -> None:
    """The default mode has to be able to say no.

    Every turn here is a silent no-op the judge did not excuse. Both sides
    of that are real decisions taken at the same point in the turn, so the
    evidence is as good as a replayed run's, and gating the block on the
    mode left ``do_not_switch`` unreachable by default.
    """
    turns = MIN_TURNS_FOR_VERDICT + 2
    run_id = _make_run(
        db_session, test_user.id, samples=turns, incumbent_source=IncumbentSource.HISTORIC
    )
    a, b, c, d = _patched_run(
        samples=_silent_noop_samples(turns),
        call_side_effect=lambda *a, **k: _result(text="Here is what I think."),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    summary = run.summary_json
    assert summary is not None
    assert summary["silent_noop_blocking_rate"] == 1.0
    assert summary["silent_noop_comparable"] is True
    assert summary["blocking_withheld"] == []
    assert run.recommendation == Recommendation.DO_NOT_SWITCH
    assert any("where acting was the better call" in reason for reason in summary["reasons"])


async def test_a_withheld_block_lands_on_inconclusive_not_on_an_endorsement(
    db_session: Session, test_user: User
) -> None:
    """Over the confounder ceiling, the run has no answer.

    A third of this sample had no reconstructable incumbent decision, so the
    turns the no-op rate was measured on may not represent the turns it was
    drawn from. The evidence still belongs in the reasons; what it must not
    do is arrive under a verdict that reads as permission to switch.
    """
    turns = MIN_TURNS_FOR_VERDICT + 2
    samples = _silent_noop_samples(turns)
    samples += [
        ReplaySample(
            seq=seq,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context=f"ask {seq}",
            historic_decision_available=False,
        )
        for seq in range(turns + 1, turns + 12)
    ]
    run_id = _make_run(
        db_session, test_user.id, samples=len(samples), incumbent_source=IncumbentSource.HISTORIC
    )
    a, b, c, d = _patched_run(
        samples=samples,
        call_side_effect=lambda *a, **k: _result(text="Here is what I think."),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    summary = run.summary_json
    assert summary is not None
    assert summary["turns_incumbent_unavailable"] == 11
    assert summary["silent_noop_comparable"] is False
    assert summary["blocking_withheld"]
    assert run.recommendation == Recommendation.INCONCLUSIVE
    assert any("where acting was the better call" in reason for reason in summary["reasons"])


async def test_a_judged_unsafe_flag_is_never_filed_against_an_unmeasured_incumbent(
    db_session: Session, test_user: User
) -> None:
    """The incumbent was not measured, so it carries no findings at all.

    The judge is shown a decision read out of the transcript; a flag on it
    describes what production did that day, under that day's prompt, not the
    model this run names. Recording it as the incumbent's finding puts a
    measurement in the column every other part of this mode reports as
    unmeasured.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, judge=True, incumbent_source=IncumbentSource.HISTORIC
    )
    a, b, c, d = _patched_run(
        samples=_historic_samples(1),
        call_side_effect=lambda *a, **k: _result(text="I'd rather not."),
        tools_by_name={"lookup": _lookup_tool()},
    )
    outcome = JudgeOutcome(
        JudgeVerdict.CANDIDATE_BETTER, "the other one texts the wrong customer", frozenset(Side)
    )
    with (
        a,
        b,
        c,
        d,
        patch(
            "backend.app.services.llm_eval.runner.judge_turn",
            AsyncMock(return_value=outcome),
        ),
    ):
        await execute_run(run_id, concurrency=1)

    turn = _turns_of(db_session, run_id)[0]
    sides = {issue["side"] for issue in json.loads(turn.safety_issues)}
    assert sides == {"candidate"}
    # The judge's reasoning is still on the turn, so the flag is readable.
    assert "wrong customer" in turn.judge_rationale
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["baseline_safety_counts"] == {}


async def test_the_judge_is_not_told_which_side_came_from_the_transcript(
    db_session: Session, test_user: User
) -> None:
    """The run passes the blinding flag through on a historic turn."""
    run_id = _make_run(
        db_session, test_user.id, samples=1, judge=True, incumbent_source=IncumbentSource.HISTORIC
    )
    judge = AsyncMock(return_value=JudgeOutcome(JudgeVerdict.EQUIVALENT, "same"))
    a, b, c, d = _patched_run(
        samples=_historic_samples(1),
        call_side_effect=lambda *a, **k: _result(text="I'll look."),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d, patch("backend.app.services.llm_eval.runner.judge_turn", judge):
        await execute_run(run_id, concurrency=1)

    assert judge.await_args_list[-1].kwargs["historic_side_shown"] is True


# ---------------------------------------------------------------------------
# Shapes the judge cannot be shown blind
# ---------------------------------------------------------------------------
#
# ``_historic_first_decision`` returns exactly one call and never prose
# beside it, so two shapes identify the recorded side however carefully
# ``judge._describe`` blinds the rest. The judge defaults to the incumbent
# model, so both are self-preference channels. The turn is withheld, which
# also keeps the confounded evidence out of the judge tier.


def _flattened_historic_sample() -> list[ReplaySample]:
    """A turn whose record holds several calls in one flat list."""
    sample = _historic_samples(1)[0]
    return [replace(sample, historic_calls_flattened=True)]


async def test_the_judge_is_withheld_on_a_flattened_turn(
    db_session: Session, test_user: User
) -> None:
    """The record lost its round boundaries, so the divergence may be that.

    Letting the turn through and discounting the tier afterwards was the old
    shape, and it left a run whose every judged divergence was one of these
    passing a guard that divided them by the whole sample.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, judge=True, incumbent_source=IncumbentSource.HISTORIC
    )
    judge = AsyncMock(return_value=JudgeOutcome(JudgeVerdict.CANDIDATE_WORSE, "worse"))
    a, b, c, d = _patched_run(
        samples=_flattened_historic_sample(),
        call_side_effect=lambda *a, **k: _result(text="I'll answer instead."),
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d, patch("backend.app.services.llm_eval.runner.judge_turn", judge):
        await execute_run(run_id, concurrency=1)

    judge.assert_not_awaited()
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["judge_skip_counts"] == {str(JudgeSkipReason.FLATTENED_ROUNDS): 1}


async def test_the_judge_is_withheld_when_the_candidate_batches_its_calls(
    db_session: Session, test_user: User
) -> None:
    """Two calls in one response can only be the candidate's.

    The record is walked call by call and yields one, so there is no second
    call to show on that side and no way to render the difference away.
    Trimming the candidate's would score a decision it did not make.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, judge=True, incumbent_source=IncumbentSource.HISTORIC
    )
    judge = AsyncMock(return_value=JudgeOutcome(JudgeVerdict.CANDIDATE_WORSE, "worse"))
    batched = _result(
        tools=[
            ToolCall(name="lookup", arguments={"q": "a1"}),
            ToolCall(name="lookup", arguments={"q": "a2"}),
        ]
    )
    a, b, c, d = _patched_run(
        samples=_historic_samples(1),
        call_side_effect=lambda *a, **k: batched,
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d, patch("backend.app.services.llm_eval.runner.judge_turn", judge):
        await execute_run(run_id, concurrency=1)

    judge.assert_not_awaited()
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["judge_skip_counts"] == {str(JudgeSkipReason.UNBLINDABLE_SHAPE): 1}
    assert any("several tools in one response" in w for w in run.summary_json["warnings"])


async def test_a_replayed_run_judges_a_batched_candidate(
    db_session: Session, test_user: User
) -> None:
    """Both sides were elicited, so either may carry two calls and neither is marked."""
    run_id = _make_run(db_session, test_user.id, samples=1, judge=True)
    judge = AsyncMock(return_value=JudgeOutcome(JudgeVerdict.EQUIVALENT, "same"))
    responses = [
        _result(tools=[ToolCall(name="lookup", arguments={"q": "a"})]),
        _result(
            tools=[
                ToolCall(name="lookup", arguments={"q": "a"}),
                ToolCall(name="lookup", arguments={"q": "b"}),
            ]
        ),
    ]
    a, b, c, d = _patched_run(
        samples=_samples(1),
        call_side_effect=responses,
        tools_by_name={"lookup": _lookup_tool()},
    )
    with a, b, c, d, patch("backend.app.services.llm_eval.runner.judge_turn", judge):
        await execute_run(run_id, concurrency=1)

    judge.assert_awaited_once()
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    assert run.summary_json["judge_skip_counts"] == {}


# ---------------------------------------------------------------------------
# Fabricated IDs are checked on the recorded side too
# ---------------------------------------------------------------------------


class _InvoiceParams(BaseModel):
    invoice_id: str


def _invoice_tool() -> Tool:
    """A mutating tool with an ID-shaped argument, untagged so it is a write."""

    async def _unused(**_kwargs: object) -> ToolResult:  # pragma: no cover
        raise AssertionError("a tool was executed during an evaluation")

    return Tool(
        name="send_invoice",
        description="send an invoice",
        function=_unused,
        params_model=_InvoiceParams,
    )


def _invoice_sample(invoice_id: str) -> list[ReplaySample]:
    """A turn whose recorded decision wrote to *invoice_id*."""
    return [
        ReplaySample(
            seq=1,
            timestamp="2026-05-01T12:00:00+00:00",
            message_context="current ask",
            historic_reply="sent it",
            historic_tool_names=["send_invoice"],
            historic_first_calls=(
                ToolCall(name="send_invoice", arguments={"invoice_id": invoice_id}),
            ),
        )
    ]


async def test_the_recorded_side_is_checked_for_fabricated_ids(
    db_session: Session, test_user: User
) -> None:
    """Both sides invent the same ID, so the comparison is parity, not an excess.

    Checking only the candidate made this run read as a candidate that writes
    to record ids against an incumbent that never does. The check needs no
    second model and no current schema: it asks whether the ID is in the
    prompt that turn was assembled from.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, incumbent_source=IncumbentSource.HISTORIC
    )
    a, b, c, d = _patched_run(
        samples=_invoice_sample("77771"),
        call_side_effect=lambda *a, **k: _result(
            tools=[ToolCall(name="send_invoice", arguments={"invoice_id": "77771"})]
        ),
        tools_by_name={"send_invoice": _invoice_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    turn = _turns_of(db_session, run_id)[0]
    fabricated = {
        issue["side"]
        for issue in json.loads(turn.safety_issues)
        if issue["finding"] == str(SafetyFinding.FABRICATED_ID)
    }
    assert fabricated == {"candidate", "baseline"}
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    comparison = run.summary_json["fabricated_id_comparison"]
    assert comparison["comparable"] is True
    assert comparison["candidate_only"] == 0
    assert comparison["baseline_only"] == 0


async def test_only_the_fabricated_id_check_runs_on_the_recorded_side(
    db_session: Session, test_user: User
) -> None:
    """The rest would be artifacts of the reading.

    The recorded call's own tool name is what ``UNREQUESTED_MUTATION``
    exempts, so it can never raise one, and validating that day's arguments
    against today's schema would reject calls production accepted. Only the
    deterministic check is run, and only its finding is recorded.
    """
    run_id = _make_run(
        db_session, test_user.id, samples=1, incumbent_source=IncumbentSource.HISTORIC
    )
    a, b, c, d = _patched_run(
        samples=_invoice_sample("77771"),
        call_side_effect=lambda *a, **k: _result(text="I'll check first."),
        tools_by_name={"send_invoice": _invoice_tool()},
    )
    with a, b, c, d:
        await execute_run(run_id, concurrency=1)

    turn = _turns_of(db_session, run_id)[0]
    baseline_findings = {
        issue["finding"] for issue in json.loads(turn.safety_issues) if issue["side"] == "baseline"
    }
    assert baseline_findings == {str(SafetyFinding.FABRICATED_ID)}
    run = db_session.execute(select(LLMEvalRun).where(LLMEvalRun.id == run_id)).scalar_one()
    assert run.summary_json is not None
    # It counts on the recorded side of the broader tier too, which is real
    # rather than a measured zero. That tier still reports rather than
    # decides, because the checks it would compare were not run there.
    assert run.summary_json["safety_comparison"]["comparable"] is False
    assert run.summary_json["safety_comparison"]["baseline_turns"] == 1
