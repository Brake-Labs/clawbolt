"""Orchestration for a single model comparison run.

A run replays N of the user's most recent turns through one model, the
candidate, and records what it decided beside what production actually did.
There is no second live call: the baseline is the record, which is both free
and the only baseline the operator cares about. Per-turn rows are written as
they complete rather than in one batch at the end, so a process restart
halfway through leaves behind the evidence it already gathered.

Turn failures are recorded, not raised. One provider hiccup on turn 40 must
not discard the 39 turns before it. Sustained failure is different: a dead
provider, a rejected key, or an exhausted quota fails every remaining turn
the same way, so after ``MAX_CONSECUTIVE_CALL_FAILURES`` turns in a row come
back with an error the run stops, keeps what it gathered, and says why rather
than spending the rest of the samples proving the point.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

from sqlalchemy import CursorResult, func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend.app.config import settings
from backend.app.database import db_session_async
from backend.app.models import ComparisonRun, ComparisonTurn, User
from backend.app.services.llm_endpoints import resolve_target
from backend.app.services.llm_service import LLMTarget
from backend.app.services.model_comparison import checks, report
from backend.app.services.model_comparison.execution import call_model
from backend.app.services.model_comparison.production_usage import (
    ProductionUsage,
    read_production_usage,
)
from backend.app.services.model_comparison.sampling import (
    ReplayFixture,
    assemble_for_sample,
    build_fixture,
    sample_clock,
    select_samples,
)
from backend.app.services.model_comparison.types import (
    PRODUCTION_CHECKED,
    Finding,
    Issue,
    ModelCallResult,
    RecordedToolResult,
    ReplaySample,
    RunStatus,
    ToolCall,
    TurnOutcome,
    TurnReport,
)

logger = logging.getLogger(__name__)

# ``asyncio.create_task`` holds only a weak reference to the task it spawns,
# so without a strong reference here the GC can collect a still-running run.
# Tasks drop themselves on completion.
_pending_tasks: set[asyncio.Task[None]] = set()


def launch_run(run_id: int, *, concurrency: int) -> None:
    """Start *run_id* on a background task and return immediately."""
    task = asyncio.create_task(_guarded(run_id, concurrency), name=f"model-comparison-{run_id}")
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


async def _guarded(run_id: int, concurrency: int) -> None:
    """Run *run_id*, recording any unhandled failure onto the row itself.

    Without this the only trace of a crashed run would be a log line and a
    row stuck at ``running`` forever.
    """
    try:
        await execute_run(run_id, concurrency=concurrency)
    except asyncio.CancelledError:
        await _finish(run_id, RunStatus.CANCELLED, error="run cancelled")
        raise
    except Exception as exc:
        logger.exception("Model comparison run %d failed", run_id)
        await _finish(run_id, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


async def _load_run(run_id: int) -> ComparisonRun | None:
    async with db_session_async() as db:
        return (
            await db.execute(select(ComparisonRun).where(ComparisonRun.id == run_id))
        ).scalar_one_or_none()


# How stale a cancellation check may be. Turns run concurrently and can each
# finish in well under a second, so checking per turn opened a fresh session
# per turn to read one column that changes at most once per run. A second of
# lag on a job measured in minutes is not worth that.
_CANCEL_POLL_SECONDS = 1.0


# How long a run may go without touching ``heartbeat_at`` before the sweep
# treats it as abandoned. Generously above the slowest plausible single turn:
# the cost of waiting is a stale row for a few minutes, and the cost of being
# wrong is marking a live run interrupted.
_HEARTBEAT_STALE_AFTER = timedelta(minutes=15)

# Consecutive turns whose provider calls failed before the run gives up. A
# dead provider, a bad key, or an exhausted quota fails every remaining turn
# identically, and a 200-turn run would make 200 more calls discovering that.
# Counted consecutively rather than in total so one flaky call in a long run
# does not end it: any turn that comes back resets the count.
MAX_CONSECUTIVE_CALL_FAILURES = 3


class _CancellationWatcher:
    """Caches the cancelled flag for a run across closely-spaced checks."""

    def __init__(self, run_id: int) -> None:
        self._run_id = run_id
        self._cancelled = False
        self._checked_at = 0.0
        self._lock = asyncio.Lock()

    async def is_cancelled(self) -> bool:
        if self._cancelled:
            # Latching: a run never un-cancels, so once seen there is nothing
            # left to ask the database.
            return True
        async with self._lock:
            now = time.monotonic()
            if now - self._checked_at < _CANCEL_POLL_SECONDS:
                return self._cancelled
            async with db_session_async() as db:
                status = (
                    await db.execute(
                        select(ComparisonRun.status).where(ComparisonRun.id == self._run_id)
                    )
                ).scalar_one_or_none()
            self._checked_at = now
            self._cancelled = status == RunStatus.CANCELLED
            return self._cancelled


async def _finish(
    run_id: int,
    status: RunStatus,
    *,
    error: str = "",
    summary: dict | None = None,
) -> None:
    async with db_session_async() as db:
        values: dict = {
            "status": str(status),
            "completed_at": datetime.now(UTC),
        }
        if error:
            values["error"] = error
        if summary is not None:
            values["summary_json"] = summary
        await db.execute(update(ComparisonRun).where(ComparisonRun.id == run_id).values(**values))
        await db.commit()


def _serialize_calls(calls: list[ToolCall]) -> str:
    return json.dumps(
        [{"name": c.name, "arguments": c.arguments} for c in calls],
        default=str,
    )


def _serialize_recorded(items: tuple[RecordedToolResult, ...] | list[RecordedToolResult]) -> str:
    """Recorded calls, results included.

    Results are kept so the report can show what production read, and what a
    replay fed back before the candidate decided. They are the live turn's own
    recorded results, already in the user's history, and the column is
    encrypted like the rest.
    """
    return json.dumps(
        [
            {
                "name": item.name,
                "arguments": item.arguments,
                "result": item.result,
                "is_error": item.is_error,
            }
            for item in items
        ],
        default=str,
    )


def _turn_row(run_id: int, turn: TurnReport) -> ComparisonTurn:
    sample = turn.sample
    cand = turn.candidate
    return ComparisonTurn(
        run_id=run_id,
        message_seq=sample.seq,
        message_timestamp=sample.timestamp,
        user_message=sample.user_text,
        production_reply=sample.production_reply,
        production_tool_calls=_serialize_recorded(sample.production_tool_calls),
        candidate_text=cand.text,
        candidate_tool_calls=_serialize_calls(cand.tool_calls),
        candidate_replayed_lookups=_serialize_recorded(cand.replayed_lookups),
        candidate_stop_reason=cand.stop_reason or "",
        candidate_input_tokens=cand.input_tokens,
        candidate_output_tokens=cand.output_tokens,
        candidate_cache_read_tokens=cand.cache_read_input_tokens,
        candidate_cache_creation_tokens=cand.cache_creation_input_tokens,
        candidate_latency_ms=cand.latency_ms,
        candidate_error=cand.error,
        outcome=str(turn.outcome),
        write_results=json.dumps(
            [
                {
                    "tool_name": write.tool_name,
                    "outcome": str(write.outcome),
                    "key_arguments": write.key_arguments,
                    "candidate_arguments": write.candidate_arguments,
                    "record_ids": write.record_ids,
                    "differing_arguments": list(write.differing_arguments),
                }
                for write in turn.writes
            ],
            default=str,
        ),
        findings=json.dumps(
            [
                {
                    "finding": str(issue.finding),
                    "tool_name": issue.tool_name,
                    "detail": issue.detail,
                    "side": str(issue.side),
                }
                for issue in turn.issues
            ]
        ),
    )


async def _replay_turn(
    fixture: ReplayFixture,
    sample: ReplaySample,
    *,
    target: LLMTarget,
    reasoning_effort: str,
) -> TurnReport:
    """Replay one turn through the candidate and describe what came back."""
    assembled = await assemble_for_sample(fixture, sample)

    # The candidate may continue through lookups the live turn also made, fed
    # the recorded results; nothing is executed. See ``call_model``.
    candidate = await call_model(
        assembled,
        fixture.tool_schemas,
        target=target,
        reasoning_effort=reasoning_effort,
        tools_by_name=fixture.tools_by_name,
        recorded=sample.production_tool_calls,
    )

    seen = checks.prompt_text(assembled.messages)
    issues = [
        *checks.check_candidate(
            candidate,
            fixture.tools_by_name,
            production_calls=sample.production_tool_calls,
            seen=seen,
        ),
        *checks.check_production(sample.production_tool_calls, fixture.tools_by_name, seen=seen),
    ]
    writes = report.compare_writes(sample, candidate, fixture.tools_by_name)
    return TurnReport(
        sample=sample,
        candidate=candidate,
        outcome=report.turn_outcome(candidate, writes, production_acted=sample.production_acted),
        issues=issues,
        writes=writes,
    )


async def _production_window(user_id: str, samples: Sequence[ReplaySample]) -> ProductionUsage:
    """What the user's live loop billed across the days the samples span.

    Bounded by the first and last sampled turn's own timestamps rather than
    by a fixed lookback, so the two sides of the cost tile describe the same
    stretch of the user's history.

    A sample whose stored timestamp will not parse is skipped rather than
    widening the window to wall time: a bad row must not turn "these forty
    turns" into "everything since the epoch". With none of them parseable
    there is no window and the usage comes back empty, which the console
    renders as "not available".
    """
    stamps = sorted(filter(None, (sample_clock(s) for s in samples)))
    if not stamps:
        logger.info(
            "No parseable sample timestamps for user %s; skipping production usage", user_id
        )
        return ProductionUsage()
    try:
        async with db_session_async() as db:
            return await read_production_usage(db, user_id, start=stamps[0], end=stamps[-1])
    except SQLAlchemyError:
        # A caveat on the cost tile is not worth failing a finished run for.
        logger.exception("Could not read production usage for user %s", user_id)
        return ProductionUsage()


async def execute_run(run_id: int, *, concurrency: int) -> None:
    """Replay the configured turns and write the run's summary."""
    run = await _load_run(run_id)
    if run is None:
        logger.warning("Model comparison run %d disappeared before it started", run_id)
        return

    async with db_session_async() as db:
        user = (await db.execute(select(User).where(User.id == run.user_id))).scalar_one_or_none()
        if user is None:
            await _finish(run_id, RunStatus.FAILED, error="user no longer exists")
            return
        await db.execute(
            update(ComparisonRun)
            .where(ComparisonRun.id == run_id)
            .values(status=str(RunStatus.RUNNING), started_at=datetime.now(UTC))
        )
        await db.commit()

    # The run knows its sample count, so the transcript read is bounded to the
    # tail it can reach rather than decrypting the user's whole history.
    fixture = await build_fixture(user, sample_limit=run.requested_samples)
    samples = select_samples(fixture, run.requested_samples)
    if not samples:
        # Not an ``error``: the report renders that in a red banner, and a
        # run that completed normally against an empty history did not fail.
        empty = report.aggregate([])
        empty.notes = ["This user has no replayable turns."]
        await _finish(run_id, RunStatus.COMPLETED, summary=_summary_payload(empty))
        return

    async with db_session_async() as db:
        await db.execute(
            update(ComparisonRun)
            .where(ComparisonRun.id == run_id)
            .values(progress_total=len(samples))
        )
        await db.commit()

    target = await resolve_target(
        endpoint=run.candidate_endpoint,
        provider=run.candidate_provider,
        model=run.candidate_model,
        api_base=settings.llm_api_base,
    )

    cancellation = _CancellationWatcher(run_id)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    turns: list[TurnReport] = []
    completed = 0
    consecutive_failures = 0
    breaker_error = ""
    lock = asyncio.Lock()

    async def worker(sample: ReplaySample) -> None:
        nonlocal completed, consecutive_failures, breaker_error
        async with semaphore:
            if await cancellation.is_cancelled():
                raise asyncio.CancelledError
            if breaker_error:
                # The provider is failing every call. Return rather than
                # raise: the turns that already landed are the run's evidence
                # and ``gather`` must not discard the bookkeeping for them.
                return
            try:
                turn = await _replay_turn(
                    fixture,
                    sample,
                    target=target,
                    reasoning_effort=run.candidate_reasoning_effort,
                )
            except Exception as exc:
                logger.exception("Replay of seq=%d failed in run %d", sample.seq, run_id)
                # A turn that could not even be assembled still belongs in
                # the report, as a failure rather than a silent omission.
                detail = f"{type(exc).__name__}: {exc}"
                turn = TurnReport(
                    sample=sample,
                    candidate=ModelCallResult(
                        provider=run.candidate_provider,
                        model=run.candidate_model,
                        error=detail,
                    ),
                    outcome=TurnOutcome.NOT_REPLAYED,
                    issues=[Issue(finding=Finding.CALL_FAILED, detail=detail)],
                )
        failure = turn.candidate.error
        async with lock:
            turns.append(turn)
            completed += 1
            if failure:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_CALL_FAILURES and not breaker_error:
                    breaker_error = (
                        f"stopped after {consecutive_failures} consecutive provider "
                        f"failures: {failure}"
                    )
                    logger.warning("Model comparison run %d %s", run_id, breaker_error)
            else:
                consecutive_failures = 0
        try:
            async with db_session_async() as db:
                db.add(_turn_row(run_id, turn))
                # Incremented in SQL rather than written from the Python counter.
                # The counter advances under the lock but the commit happens
                # outside it, so a worker holding a lower count could land last
                # and walk progress backwards. This adds exactly one per committed
                # turn regardless of the order the workers reach the database.
                await db.execute(
                    update(ComparisonRun)
                    .where(ComparisonRun.id == run_id)
                    .values(
                        progress_completed=ComparisonRun.progress_completed + 1,
                        heartbeat_at=datetime.now(UTC),
                    )
                )
                await db.commit()
        except IntegrityError:
            # The run was deleted while this turn was in flight. Reachable on
            # the documented remedy for deleting an active run: cancel, then
            # delete. Cancellation is only checked at the top of the turn and
            # the latch may be a second stale, so a worker can still be inside
            # a provider call when the row goes away.
            #
            # Nothing is lost by returning: the turn's own row is gone with
            # its parent, which is what the operator asked for. Raising would
            # reach ``_guarded``, which would log a stack trace and stamp a
            # deleted row as FAILED, turning a deliberate deletion into what
            # reads like a crashed run.
            logger.info(
                "Model comparison run %d was deleted mid-flight; discarding turn seq=%d",
                run_id,
                sample.seq,
            )
            return

    try:
        await asyncio.gather(*(worker(s) for s in samples))
    except asyncio.CancelledError:
        logger.info("Model comparison run %d cancelled after %d turns", run_id, completed)
        raise

    turns.sort(key=lambda t: t.sample.seq)
    summary = report.aggregate(
        turns,
        target=target,
        endpoint=run.candidate_endpoint,
        production=await _production_window(run.user_id, samples),
    )
    if breaker_error:
        # The evidence gathered before the provider went down is kept and
        # still readable. It is a fraction of what was asked for, which the
        # note and the run's own error both say.
        summary.notes.insert(0, breaker_error)
        await _finish(
            run_id, RunStatus.FAILED, summary=_summary_payload(summary), error=breaker_error
        )
        logger.info(
            "Model comparison run %d abandoned after %d turn(s): %s",
            run_id,
            completed,
            breaker_error,
        )
        return
    await _finish(run_id, RunStatus.COMPLETED, summary=_summary_payload(summary))
    logger.info(
        "Model comparison run %d complete: %d turns, %d candidate violation(s), "
        "%d of %d writes matched",
        run_id,
        summary.turns_replayed,
        summary.candidate_violations,
        summary.writes_matched,
        summary.writes_total,
    )


def _summary_payload(summary: report.RunSummary) -> dict:
    """Freeze the summary into the JSON stored on the run row.

    Stored rather than recomputed so a later change to what the checks look
    for never silently rewrites the numbers on a run an operator already read.
    """
    totals = summary.candidate
    live = summary.production
    return {
        "turns_total": summary.turns_total,
        "turns_replayed": summary.turns_replayed,
        "turns_failed": summary.turns_failed,
        "outcome_counts": summary.outcome_counts,
        "candidate_findings": summary.candidate_findings,
        "production_findings": summary.production_findings,
        "candidate_violations": summary.candidate_violations,
        "production_violations": summary.production_violations,
        "production_checked_findings": sorted(str(f) for f in PRODUCTION_CHECKED),
        "writes_total": summary.writes_total,
        "writes_matched": summary.writes_matched,
        "writes_same_record": summary.writes_same_record,
        "writes_args_differ": summary.writes_args_differ,
        "writes_missed": summary.writes_missed,
        "writes_not_reached": summary.writes_not_reached,
        "writes_measured": summary.writes_measured,
        "write_match_rate": round(summary.write_match_rate, 4),
        "candidate": {
            "provider": totals.provider,
            "model": totals.model,
            "input_tokens": totals.input_tokens,
            "output_tokens": totals.output_tokens,
            "cache_read_tokens": totals.cache_read_tokens,
            "cache_creation_tokens": totals.cache_creation_tokens,
            "billed_prompt_tokens": totals.billed_prompt_tokens,
            # ``None``, never "0.000000". A zero here reads as a measurement,
            # and for a gateway model name it never is one. The latencies are
            # ``None`` for the same reason when no call came back.
            "total_cost_usd": None if totals.total_cost is None else str(totals.total_cost),
            "cost_unavailable_reason": totals.cost_unavailable_reason,
            "latency_p50_ms": _rounded(totals.percentile_latency_ms(0.50)),
            "latency_p95_ms": _rounded(totals.percentile_latency_ms(0.95)),
        },
        "production": {
            "calls": live.calls,
            "input_tokens": live.input_tokens,
            "output_tokens": live.output_tokens,
            "cache_read_tokens": live.cache_read_tokens,
            "cache_creation_tokens": live.cache_creation_tokens,
            "billed_prompt_tokens": live.billed_prompt_tokens,
            "total_cost_usd": None if live.total_cost is None else str(live.total_cost),
            "unpriced_calls": live.unpriced_calls,
            "window_start": live.window_start,
            "window_end": live.window_end,
        },
        "notes": summary.notes,
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


async def mark_interrupted_runs() -> None:
    """Flag runs whose worker has gone quiet.

    "Still ``running``" is not on its own evidence of an abandoned run: a
    rolling deploy boots the new instance while the old one drains, and more
    than one process can serve the app, so an unconditional sweep marks a
    live run interrupted. The run keeps calling the provider and later
    overwrites the row, and in the meantime both concurrency guards in
    ``start_run`` key off ``ACTIVE_STATUSES`` and would let a duplicate run
    start for the user.

    A run touches ``heartbeat_at`` as each turn lands, so only rows quiet for
    longer than ``_HEARTBEAT_STALE_AFTER`` are swept. A row that never got a
    heartbeat falls back to ``created_at``, which covers a process that died
    between the insert and the first turn.

    Run periodically, not only at boot: staleness takes time to establish, so
    a boot-only sweep misses every run whose process died less than
    ``_HEARTBEAT_STALE_AFTER`` before that boot, which is the ordinary
    crash-and-restart. Nothing would then clear the row until the next boot,
    days later, while it 409s that user's runs and holds a slot against
    ``model_comparison_max_concurrent_runs``. See :class:`InterruptedRunSweeper`.
    """
    cutoff = datetime.now(UTC) - _HEARTBEAT_STALE_AFTER
    async with db_session_async() as db:
        result = await db.execute(
            update(ComparisonRun)
            .where(
                ComparisonRun.status.in_([str(RunStatus.RUNNING), str(RunStatus.PENDING)]),
                func.coalesce(ComparisonRun.heartbeat_at, ComparisonRun.created_at) < cutoff,
            )
            .values(
                status=str(RunStatus.INTERRUPTED),
                completed_at=datetime.now(UTC),
                error="interrupted by a server restart",
            )
        )
        await db.commit()
    count = cast("CursorResult[object]", result).rowcount or 0
    if count:
        logger.warning("Marked %d in-flight model comparison run(s) as interrupted", count)


class InterruptedRunSweeper:
    """Periodically closes out runs whose worker stopped touching them.

    Modelled on ``HeartbeatScheduler``: one task, started and stopped by the
    lifespan. The interval only bounds how long a dead run stays visible as
    in-flight, so it is well under ``_HEARTBEAT_STALE_AFTER`` and far above
    anything that would matter for load.
    """

    def __init__(self, interval_seconds: float = 300.0) -> None:
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the sweep loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._run())

    def stop(self) -> None:
        """Cancel the sweep loop."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
                await mark_interrupted_runs()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed sweep must not kill the loop: the next tick is a
                # cheap retry, and the alternative is silently never sweeping
                # again for the life of the process.
                logger.exception("Interrupted-run sweep failed; continuing")


interrupted_run_sweeper = InterruptedRunSweeper()
