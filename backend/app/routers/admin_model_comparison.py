"""Admin endpoints for the model comparison report.

An operator picks a user and a candidate model; the replay runs that user's
most recent turns through the candidate and the report lays each decision
beside what production actually did. The comparison itself lives in
``backend.app.services.model_comparison``; this module owns authorization,
request validation, job lifecycle, and serialization.

Consent-gated in the same sense as ``/admin/shared-data``: a run reads the
user's real conversations and the report renders them back to an admin, so
both require ``User.data_sharing_consent``. Content is PII-redacted on the
way out, exactly as it is there. The redaction runs *after* the checks, so
findings are computed on the real values and only the human-readable
drill-down is masked.

Endpoints:

- ``POST /admin/model-comparison/users/{user_id}/runs`` starts a run.
- ``GET  /admin/model-comparison/runs`` lists runs, newest first, across every
  user or one of them (``?user_id=``).
- ``GET  /admin/model-comparison/runs/{run_id}`` returns a run plus its turns,
  the ones worth reading first.
- ``GET  /admin/model-comparison/runs/{run_id}/progress`` returns counters
  only, for the console's poll. Unaudited; see the handler.
- ``POST /admin/model-comparison/runs/{run_id}/cancel`` stops an in-flight run.
- ``DELETE /admin/model-comparison/runs/{run_id}`` discards a run and its turns.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError
from sqlalchemy import desc, select
from sqlalchemy import func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.admin_dep import get_current_admin
from backend.app.config import settings
from backend.app.database import get_async_db
from backend.app.models import ComparisonRun, ComparisonTurn, Subscription, User
from backend.app.query_helpers import count_rows, fetch_all, iso_or_none
from backend.app.schemas.model_comparison import (
    ComparisonFinding,
    ComparisonReportResponse,
    ComparisonRunCreate,
    ComparisonRunItem,
    ComparisonRunListResponse,
    ComparisonRunProgress,
    ComparisonSummary,
    ComparisonToolCall,
    ComparisonTurnItem,
    ComparisonWrite,
)
from backend.app.services.admin_audit import AdminAction, AdminAuditContext, audit_admin
from backend.app.services.llm_endpoints import UnknownLLMEndpointError, resolve_target
from backend.app.services.model_comparison import launch_run
from backend.app.services.model_comparison.types import (
    HARD_VIOLATIONS,
    RunStatus,
    Side,
    TurnOutcome,
)
from backend.app.services.pii_redaction import redact_pii, redact_pii_recursive

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/model-comparison", tags=["admin"])

ACTIVE_STATUSES = (str(RunStatus.PENDING), str(RunStatus.RUNNING))

# Largest page ``list_runs`` will serve. Reported on the response so the
# console can clamp to it instead of growing past it into a 422.
MAX_RUN_PAGE_SIZE = 100


async def _consenting_user(user_id: str, db: AsyncSession) -> User:
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.data_sharing_consent:
        raise HTTPException(
            status_code=403,
            detail="User has not consented to data sharing.",
        )
    return user


async def _effective_models(user_id: str, db: AsyncSession) -> tuple[str, str, str]:
    """Resolve the (endpoint, provider, model) this user's loop runs on today.

    Mirrors ``ClawboltAgent._resolve_target``, including its pairing rule:
    the model falls through to the global default on its own, but endpoint
    and provider are inherited together, so a user pinned to a bare provider
    does not keep the global endpoint.

    Recorded on the run as a label. Nothing is sent here.
    """
    sub = (
        await db.execute(select(Subscription).where(Subscription.user_id == user_id))
    ).scalar_one_or_none()
    model = (sub.llm_model_override if sub else "") or settings.llm_model
    endpoint = sub.llm_endpoint_override if sub else ""
    provider = sub.llm_provider_override if sub else ""
    if endpoint or provider:
        return endpoint, provider, model
    return settings.llm_endpoint, settings.llm_provider, model


def _summary_of(run: ComparisonRun) -> ComparisonSummary | None:
    """The run's frozen summary, or None when it cannot be read back.

    Every field on ``ComparisonSummary`` is required, which is what keeps the
    generated frontend types free of ``?? 0``. The cost is that a summary
    written by an older shape of the code cannot be validated, and raising
    here would 500 the whole listing rather than one row: an operator could
    not then reach the console to delete the stranded run. ``summary`` is
    already nullable on the wire and the console renders "This run has no
    summary", so degrading to that is the honest answer.
    """
    if not run.summary_json:
        return None
    try:
        return ComparisonSummary.model_validate(run.summary_json)
    except ValidationError:
        logger.warning(
            "Comparison run %s has a summary this build cannot read; serving it without one",
            run.public_id,
        )
        return None


def _run_item(
    run: ComparisonRun, *, user_email: str = "", user_consented: bool = True
) -> ComparisonRunItem:
    return ComparisonRunItem(
        id=run.public_id,
        user_email=user_email,
        user_consented=user_consented,
        user_id=run.user_id,
        incumbent_endpoint=run.incumbent_endpoint,
        incumbent_provider=run.incumbent_provider,
        incumbent_model=run.incumbent_model,
        candidate_endpoint=run.candidate_endpoint,
        candidate_provider=run.candidate_provider,
        candidate_model=run.candidate_model,
        candidate_reasoning_effort=run.candidate_reasoning_effort,
        requested_samples=run.requested_samples,
        status=run.status,
        progress_completed=run.progress_completed,
        progress_total=run.progress_total,
        error=run.error,
        created_at=run.created_at.isoformat(),
        started_at=iso_or_none(run.started_at),
        completed_at=iso_or_none(run.completed_at),
        summary=_summary_of(run),
    )


def _load_json(raw: str, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _entries(raw: str) -> list[dict[str, Any]]:
    """A stored JSON list of objects, tolerating a row that is neither."""
    loaded = _load_json(raw, [])
    if not isinstance(loaded, list):
        return []
    return [entry for entry in loaded if isinstance(entry, dict)]


# A recorded tool result can be a whole document. The drill-down needs enough
# to see what the model read, not the payload.
_MAX_RESULT_CHARS = 2000


def _tool_calls(raw: str) -> list[ComparisonToolCall]:
    return [
        ComparisonToolCall(
            name=str(entry.get("name", "")),
            arguments=redact_pii_recursive(entry.get("arguments") or {}),
            result=redact_pii(str(entry.get("result", ""))[:_MAX_RESULT_CHARS]),
            is_error=bool(entry.get("is_error", False)),
        )
        for entry in _entries(raw)
    ]


def _side(entry: dict[str, Any]) -> Literal["production", "candidate"]:
    return "production" if entry.get("side") == Side.PRODUCTION else "candidate"


def _findings(raw: str) -> list[ComparisonFinding]:
    return [
        ComparisonFinding(
            finding=str(entry.get("finding", "")),
            tool_name=str(entry.get("tool_name", "")),
            detail=redact_pii(str(entry.get("detail", ""))),
            violation=entry.get("finding") in HARD_VIOLATIONS,
            side=_side(entry),
        )
        for entry in _entries(raw)
    ]


def _writes(raw: str) -> list[ComparisonWrite]:
    return [
        ComparisonWrite(
            tool_name=str(entry.get("tool_name", "")),
            outcome=str(entry.get("outcome", "")),
            key_arguments=redact_pii_recursive(entry.get("key_arguments") or {}),
            candidate_arguments=(
                redact_pii_recursive(entry["candidate_arguments"])
                if isinstance(entry.get("candidate_arguments"), dict)
                else None
            ),
            # Record IDs are the join between the two calls, not content, so
            # they are served as recorded. Redacting them would leave the
            # card unable to say the two writes hit the same job.
            record_ids={
                str(path): [str(value) for value in values]
                for path, values in (entry.get("record_ids") or {}).items()
                if isinstance(values, list)
            },
            differing_arguments=[
                str(name) for name in (entry.get("differing_arguments") or []) if name
            ],
        )
        for entry in _entries(raw)
    ]


def _candidate_violation(turn: ComparisonTurn) -> bool:
    """Whether the candidate has a finding that counts as a violation.

    ``TOOL_NOT_IN_SCHEMA`` and ``CALL_FAILED`` are recorded on the turn but
    are not something a model did, and a production-side finding is the
    comparison rather than evidence against the candidate.
    """
    return any(
        entry.get("finding") in HARD_VIOLATIONS and _side(entry) == Side.CANDIDATE
        for entry in _entries(turn.findings)
    )


def _turn_item(turn: ComparisonTurn) -> ComparisonTurnItem:
    return ComparisonTurnItem(
        message_seq=turn.message_seq,
        message_timestamp=turn.message_timestamp,
        user_message=redact_pii(turn.user_message),
        production_reply=redact_pii(turn.production_reply),
        production_tool_calls=_tool_calls(turn.production_tool_calls),
        candidate_text=redact_pii(turn.candidate_text),
        candidate_tool_calls=_tool_calls(turn.candidate_tool_calls),
        candidate_replayed_lookups=_tool_calls(turn.candidate_replayed_lookups),
        candidate_stop_reason=turn.candidate_stop_reason,
        candidate_input_tokens=turn.candidate_input_tokens,
        candidate_output_tokens=turn.candidate_output_tokens,
        candidate_cache_read_tokens=turn.candidate_cache_read_tokens,
        candidate_cache_creation_tokens=turn.candidate_cache_creation_tokens,
        candidate_latency_ms=turn.candidate_latency_ms,
        candidate_error=turn.candidate_error,
        outcome=turn.outcome,
        writes=_writes(turn.write_results),
        findings=_findings(turn.findings),
    )


# Drill-down ordering. The point of the report is the handful of turns an
# operator acts on, so they come first and a hundred turns where the candidate
# did what production did sit below them.
#
# A candidate that answered a turn with nothing at all is first: it is the
# hardest failure on the page and the one that reads cleanest everywhere
# else, since a turn with no output has nothing to flag. Then a write
# production made and the candidate did not reach, then the two
# different-arguments buckets, with "right record" below "wrong record".
_OUTCOME_PRIORITY = {
    str(TurnOutcome.NO_CANDIDATE_OUTPUT): 1,
    str(TurnOutcome.WRITE_MISSED): 2,
    str(TurnOutcome.WRITE_ARGS_DIFFER): 3,
    str(TurnOutcome.WRITE_SAME_RECORD): 4,
    str(TurnOutcome.NOT_REPLAYED): 5,
    str(TurnOutcome.REPLAY_INCOMPLETE): 6,
    str(TurnOutcome.WRITE_MATCHED): 7,
    str(TurnOutcome.NO_WRITE): 8,
}


def _turn_sort_key(turn: ComparisonTurn) -> tuple[int, int, int, int]:
    """Rank turns by how much they should change the reader's mind.

    The candidate's violations first, then how the turn's writes came out,
    then any other finding (production's included, which is the "it does this
    too" evidence), then the newest turn.
    """
    has_violation = 0 if _candidate_violation(turn) else 1
    has_other_finding = 0 if _entries(turn.findings) else 1
    return (
        has_violation,
        _OUTCOME_PRIORITY.get(turn.outcome, len(_OUTCOME_PRIORITY) + 1),
        has_other_finding,
        -turn.message_seq,
    )


@router.post("/users/{user_id}/runs", response_model=ComparisonRunItem, status_code=201)
async def start_run(
    user_id: str,
    payload: ComparisonRunCreate,
    ctx: AdminAuditContext = Depends(audit_admin(AdminAction.START_MODEL_COMPARISON_RUN)),
    db: AsyncSession = Depends(get_async_db),
) -> ComparisonRunItem:
    """Queue a comparison run and return the row immediately.

    The run executes on a background task; poll the returned ``id`` for
    progress. One active run per user: a second concurrent replay of the
    same history would double the provider load for no extra information.
    """
    user = await _consenting_user(user_id, db)
    ctx.target_user_id = user.id

    if payload.sample_count > settings.model_comparison_max_samples:
        raise HTTPException(
            status_code=422,
            detail=f"sample_count exceeds the configured maximum of "
            f"{settings.model_comparison_max_samples}",
        )

    active = (
        await db.execute(
            select(ComparisonRun.id)
            .where(ComparisonRun.user_id == user_id)
            .where(ComparisonRun.status.in_(ACTIVE_STATUSES))
        )
    ).first()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail="A comparison is already running for this user.",
        )

    # Runs compete with live inbound traffic for the same provider rate limit,
    # so the per-user guard above is not enough on its own.
    running_total = await count_rows(
        db, ComparisonRun.id, ComparisonRun.status.in_(ACTIVE_STATUSES)
    )
    if running_total >= settings.model_comparison_max_concurrent_runs:
        raise HTTPException(
            status_code=429,
            detail=(
                f"{running_total} comparison(s) already running; the limit is "
                f"{settings.model_comparison_max_concurrent_runs}. Try again when one finishes."
            ),
        )

    incumbent_endpoint, incumbent_provider, incumbent_model = await _effective_models(user_id, db)

    # A candidate has to name somewhere to send the call. Neither half is
    # required on its own (an endpoint carries its own dialect), but with
    # both empty ``resolve_target`` yields a target with no provider and
    # any-llm refuses every turn, which is the same wasted budget the
    # endpoint check below exists to avoid.
    if not (payload.candidate_endpoint or payload.candidate_provider):
        raise HTTPException(
            status_code=422,
            detail="candidate: name either an endpoint or a provider.",
        )

    # Resolve before the run row exists. An endpoint that does not exist is a
    # 422 the operator can act on, not a run that starts, fails every turn,
    # and reports nothing after spending the budget.
    try:
        await resolve_target(
            endpoint=payload.candidate_endpoint,
            provider=payload.candidate_provider,
            model="probe",
            api_base=settings.llm_api_base,
        )
    except UnknownLLMEndpointError as exc:
        raise HTTPException(status_code=422, detail=f"candidate: {exc}") from exc

    # An unset effort means "whatever the deployment runs at". Frozen onto
    # the row at creation rather than read at call time, so a mid-run change
    # to the global setting cannot silently redefine what was measured.
    effort = payload.candidate_reasoning_effort or settings.reasoning_effort

    run = ComparisonRun(
        user_id=user_id,
        created_by_admin_id=ctx.admin_user_id,
        incumbent_endpoint=incumbent_endpoint,
        incumbent_provider=incumbent_provider,
        incumbent_model=incumbent_model,
        candidate_endpoint=payload.candidate_endpoint,
        candidate_provider=payload.candidate_provider,
        candidate_model=payload.candidate_model,
        candidate_reasoning_effort=effort,
        requested_samples=payload.sample_count,
        status=str(RunStatus.PENDING),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    launch_run(run.id, concurrency=settings.model_comparison_concurrency)
    logger.info(
        "Started model comparison run %d for user %s: %s/%s (%s) over %d turns",
        run.id,
        user_id,
        payload.candidate_endpoint or payload.candidate_provider,
        payload.candidate_model,
        effort,
        payload.sample_count,
    )
    return _run_item(run)


@router.get("/runs", response_model=ComparisonRunListResponse)
async def list_runs(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=MAX_RUN_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    ctx: AdminAuditContext = Depends(audit_admin(AdminAction.VIEW_MODEL_COMPARISON_RUNS)),
    db: AsyncSession = Depends(get_async_db),
) -> ComparisonRunListResponse:
    """Comparison runs, newest first, across every user or one of them.

    Unfiltered by default so the console can answer "what has been compared
    lately", which is how an operator finds a run again weeks later without
    remembering whose it was. ``user_id`` narrows it to one user for the run
    form beside it.

    No consent gate here, unlike the report: a row is run metadata (which
    model, how many turns, what it cost), not the user's conversations. Each
    row carries ``user_consented`` so the console can show that a run's
    evidence is no longer readable rather than offering a link that 403s.
    """
    query = select(ComparisonRun).order_by(desc(ComparisonRun.created_at))
    total_query = select(sa_func.count()).select_from(ComparisonRun)
    if user_id is not None:
        # Existence still matters: a typo'd id should 404 rather than quietly
        # return an empty list that reads as "this user has never been run".
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        ctx.target_user_id = user.id
        query = query.where(ComparisonRun.user_id == user_id)
        total_query = total_query.where(ComparisonRun.user_id == user_id)

    runs = await fetch_all(db, query.limit(limit).offset(offset))
    total = await db.scalar(total_query) or 0

    # One query for the identities on this page rather than one per row.
    owner_ids = {r.user_id for r in runs}
    emails: dict[str, str] = {}
    consented: dict[str, bool] = {}
    if owner_ids:
        rows = (
            await db.execute(
                select(User.id, User.data_sharing_consent, Subscription.email)
                .outerjoin(Subscription, Subscription.user_id == User.id)
                .where(User.id.in_(owner_ids))
            )
        ).all()
        for owner_id, consent, email in rows:
            emails[owner_id] = email or ""
            consented[owner_id] = bool(consent)

    return ComparisonRunListResponse(
        runs=[
            _run_item(
                r,
                user_email=emails.get(r.user_id, ""),
                user_consented=consented.get(r.user_id, False),
            )
            for r in runs
        ],
        total=total,
        max_samples=settings.model_comparison_max_samples,
        max_page_size=MAX_RUN_PAGE_SIZE,
    )


@router.get("/runs/{run_id}", response_model=ComparisonReportResponse)
async def get_report(
    run_id: str,
    limit: int = Query(default=10, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    ctx: AdminAuditContext = Depends(audit_admin(AdminAction.VIEW_MODEL_COMPARISON_REPORT)),
    db: AsyncSession = Depends(get_async_db),
) -> ComparisonReportResponse:
    """Return a run and a page of its turns, the ones worth reading first.

    The default page is ten because that is the part of the report anyone
    acts on. The rest is available on request rather than shipped by default:
    every text column on a turn is envelope-encrypted and then PII-redacted,
    so serializing a 200-turn run whole is over a thousand decrypts for a
    single page view. The ordering is what makes a page worth reading, so the
    sort runs across the whole run and the page is taken from the result, not
    the other way round.
    """
    run = (
        await db.execute(select(ComparisonRun).where(ComparisonRun.public_id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    await _consenting_user(run.user_id, db)
    ctx.target_user_id = run.user_id

    turns = (
        (await db.execute(select(ComparisonTurn).where(ComparisonTurn.run_id == run.id)))
        .scalars()
        .all()
    )
    ordered = sorted(turns, key=_turn_sort_key)
    page = ordered[offset : offset + limit]
    return ComparisonReportResponse(
        run=_run_item(run),
        turns=[_turn_item(t) for t in page],
        total_turns=len(ordered),
    )


@router.get("/runs/{run_id}/progress", response_model=ComparisonRunProgress)
async def get_run_progress(
    run_id: str,
    _admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_db),
) -> ComparisonRunProgress:
    """Report how far a run has got.

    Deliberately not audited, and deliberately not consent-gated: it returns
    counters and a status, never a turn, a model output, or an email. The
    console polls this every couple of seconds while a run is in flight and
    fetches the audited report only when there is something new to read. The
    audited endpoints were being polled at the same cadence, which buried a
    single human read under hundreds of audit rows.
    """
    run = (
        await db.execute(select(ComparisonRun).where(ComparisonRun.public_id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return ComparisonRunProgress(
        id=run.public_id,
        status=run.status,
        progress_completed=run.progress_completed,
        progress_total=run.progress_total,
    )


@router.post("/runs/{run_id}/cancel", response_model=ComparisonRunItem)
async def cancel_run(
    run_id: str,
    ctx: AdminAuditContext = Depends(audit_admin(AdminAction.CANCEL_MODEL_COMPARISON_RUN)),
    db: AsyncSession = Depends(get_async_db),
) -> ComparisonRunItem:
    """Ask an in-flight run to stop.

    Flips the status; the worker checks it between turns and unwinds. Turns
    already written stay, so a cancelled run keeps whatever evidence it had
    gathered.
    """
    run = (
        await db.execute(select(ComparisonRun).where(ComparisonRun.public_id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    ctx.target_user_id = run.user_id
    if run.status not in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail=f"Run is already {run.status}.")
    run.status = str(RunStatus.CANCELLED)
    await db.commit()
    await db.refresh(run)
    return _run_item(run)


@router.delete("/runs/{run_id}", status_code=204)
async def delete_run(
    run_id: str,
    ctx: AdminAuditContext = Depends(audit_admin(AdminAction.DELETE_MODEL_COMPARISON_RUN)),
    db: AsyncSession = Depends(get_async_db),
) -> None:
    """Discard a run and every turn it recorded.

    Runs accumulate, and a report is only as good as the checks that produced
    it, so a change to what the replay looks for strands every earlier run.
    Leaving them listed is worse than losing them, because the console sorts
    newest-first and nothing on a stale row says it was measured by
    since-changed code.

    Not consent-gated, unlike the report. A run belonging to a user who has
    since withdrawn consent is the run most worth removing, and a gate here
    would pin it in the list permanently.

    Refuses while the run is still going. Its workers are mid-flight against
    a paid provider, and deleting under them throws that spend away for a
    result nobody asked to abandon, so stopping the run is a decision the
    operator makes explicitly: cancel first, then delete. A worker that is
    already inside a turn when the row goes away unwinds quietly; see the
    ``IntegrityError`` branch in ``model_comparison.runner``.

    The turns go with the run through ``model_comparison_turns.run_id``'s
    ``ON DELETE CASCADE``. The audit row this request writes survives, and is
    the only remaining evidence the run was ever here.
    """
    run = (
        await db.execute(select(ComparisonRun).where(ComparisonRun.public_id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    ctx.target_user_id = run.user_id
    if run.status in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Run is still {run.status}. Cancel it before deleting.",
        )
    await db.delete(run)
    await db.commit()
