"""Admin endpoints for the model comparison report.

Covers the things that gate real behavior: the consent requirement, the
server-side resolution of the user's current model (an operator cannot claim
the candidate was weighed against something else), the one-run-per-user
guard, cancellation, and the report's worst-first ordering and PII redaction.

``launch_run`` is patched throughout. Letting it fire would start a real
background comparison run against real providers.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.admin_dep import get_current_admin
from backend.app.config import settings
from backend.app.database import db_session_async
from backend.app.models import (
    AdminAuditLog,
    ComparisonRun,
    ComparisonTurn,
    LLMEndpoint,
    Subscription,
    User,
)
from backend.app.services.admin_audit import AdminAction
from backend.app.services.llm_endpoints import reset_llm_endpoint_cache
from backend.app.services.model_comparison.types import RunStatus, TurnOutcome

BASE = "/api/admin/model-comparison"


@pytest.fixture()
def admin_client(client: TestClient, test_user: User) -> Generator[TestClient]:
    from tests.multi_user.conftest import MULTI_USER_APP as app

    app.dependency_overrides[get_current_admin] = lambda: test_user
    yield client
    app.dependency_overrides.pop(get_current_admin, None)


@pytest.fixture()
def consenting_user(db_session: Session, test_user: User) -> User:
    user = db_session.get(User, test_user.id)
    assert user is not None
    user.data_sharing_consent = True
    db_session.commit()
    return test_user


@pytest.fixture()
def _launch() -> Generator[MagicMock]:
    with patch("backend.app.routers.admin_model_comparison.launch_run") as mock:
        yield mock


@pytest.fixture(autouse=True)
def _global_model() -> Generator[None]:
    with (
        patch.object(settings, "llm_provider", "anthropic"),
        patch.object(settings, "llm_model", "incumbent-model"),
    ):
        yield


def _payload(**overrides: object) -> dict:
    body = {
        "candidate_provider": "anthropic",
        "candidate_model": "candidate-model",
        "sample_count": 50,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Consent gate
# ---------------------------------------------------------------------------


def test_starting_a_run_requires_consent(
    admin_client: TestClient, test_user: User, _launch: MagicMock
) -> None:
    response = admin_client.post(f"{BASE}/users/{test_user.id}/runs", json=_payload())
    assert response.status_code == 403
    assert "consent" in response.json()["detail"].lower()
    _launch.assert_not_called()


def test_unknown_user_is_404(admin_client: TestClient, _launch: MagicMock) -> None:
    response = admin_client.post(f"{BASE}/users/{uuid.uuid4()}/runs", json=_payload())
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Starting a run
# ---------------------------------------------------------------------------


def test_start_run_creates_a_pending_row_and_launches(
    admin_client: TestClient, consenting_user: User, db_session: Session, _launch: MagicMock
) -> None:
    response = admin_client.post(f"{BASE}/users/{consenting_user.id}/runs", json=_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == RunStatus.PENDING
    assert body["candidate_model"] == "candidate-model"
    assert body["requested_samples"] == 50
    _launch.assert_called_once()

    # ``body["id"]`` is the public id, so the row is found by that column.
    row = db_session.execute(
        select(ComparisonRun).filter_by(public_id=body["id"])
    ).scalar_one_or_none()
    assert row is not None
    assert row.user_id == consenting_user.id


def test_the_incumbent_label_comes_from_the_server_not_the_client(
    admin_client: TestClient, consenting_user: User, _launch: MagicMock
) -> None:
    """The report says what the candidate would be replacing.

    Nothing is sent there, but a client-supplied label would let a report
    claim the candidate was weighed against a model this user never ran on.
    """
    response = admin_client.post(
        f"{BASE}/users/{consenting_user.id}/runs",
        json=_payload(incumbent_model="something-else"),
    )
    assert response.status_code == 201
    assert response.json()["incumbent_model"] == "incumbent-model"


def test_reasoning_effort_defaults_to_the_deployment_setting_and_is_frozen(
    admin_client: TestClient, consenting_user: User, db_session: Session, _launch: MagicMock
) -> None:
    """An unset effort resolves at creation, not at call time.

    Reading the global when each turn fires would let a mid-run change to the
    setting redefine what the finished report claims to have measured.
    """
    with patch.object(settings, "reasoning_effort", "medium"):
        response = admin_client.post(f"{BASE}/users/{consenting_user.id}/runs", json=_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["candidate_reasoning_effort"] == "medium"

    row = db_session.execute(select(ComparisonRun).filter_by(public_id=body["id"])).scalar_one()
    assert row.candidate_reasoning_effort == "medium"


def test_an_explicit_effort_overrides_the_deployment_setting(
    admin_client: TestClient, consenting_user: User, _launch: MagicMock
) -> None:
    """Only the candidate has one: nothing is sent to the incumbent."""
    response = admin_client.post(
        f"{BASE}/users/{consenting_user.id}/runs",
        json=_payload(candidate_reasoning_effort="none"),
    )
    assert response.status_code == 201
    assert response.json()["candidate_reasoning_effort"] == "none"


def test_an_unknown_reasoning_effort_is_rejected(
    admin_client: TestClient, consenting_user: User, _launch: MagicMock
) -> None:
    response = admin_client.post(
        f"{BASE}/users/{consenting_user.id}/runs",
        json=_payload(candidate_reasoning_effort="maximum"),
    )
    assert response.status_code == 422
    _launch.assert_not_called()


def test_a_candidate_endpoint_that_does_not_exist_is_rejected_before_launching(
    admin_client: TestClient, consenting_user: User, _launch: MagicMock
) -> None:
    """Caught at creation rather than by the circuit breaker.

    Left to the runner, every turn would fail against the missing endpoint
    and the run would spend part of its sample budget before reporting
    ``inconclusive``.
    """
    response = admin_client.post(
        f"{BASE}/users/{consenting_user.id}/runs",
        json=_payload(candidate_endpoint="nonexistent"),
    )
    assert response.status_code == 422
    assert "nonexistent" in response.json()["detail"]
    _launch.assert_not_called()


def test_a_candidate_with_neither_endpoint_nor_provider_is_rejected(
    admin_client: TestClient, consenting_user: User, _launch: MagicMock
) -> None:
    """A candidate has to name somewhere to send the call.

    Either half is enough on its own, but with both empty the resolved
    target has no provider, any-llm refuses every turn, and the run burns
    its baseline budget before reporting ``inconclusive``.
    """
    response = admin_client.post(
        f"{BASE}/users/{consenting_user.id}/runs",
        json=_payload(candidate_provider="", candidate_endpoint=""),
    )
    assert response.status_code == 422
    _launch.assert_not_called()


def test_the_candidate_endpoint_is_recorded_on_the_run(
    admin_client: TestClient, consenting_user: User, _launch: MagicMock
) -> None:
    async def _create() -> None:
        async with db_session_async() as db:
            db.add(LLMEndpoint(name="otari", dialect="anthropic", base_url="https://gw.test"))
            await db.commit()
        reset_llm_endpoint_cache()

    asyncio.run(_create())
    try:
        response = admin_client.post(
            f"{BASE}/users/{consenting_user.id}/runs",
            json=_payload(candidate_endpoint="otari", candidate_provider=""),
        )
        assert response.status_code == 201
        assert response.json()["candidate_endpoint"] == "otari"
    finally:
        reset_llm_endpoint_cache()


def test_the_incumbent_label_prefers_the_users_subscription_override(
    admin_client: TestClient,
    consenting_user: User,
    db_session: Session,
    _launch: MagicMock,
) -> None:
    db_session.add(
        Subscription(
            user_id=consenting_user.id,
            role="user",
            plan="free",
            status="active",
            llm_model_override="pinned-model",
        )
    )
    db_session.commit()

    response = admin_client.post(f"{BASE}/users/{consenting_user.id}/runs", json=_payload())
    assert response.status_code == 201
    assert response.json()["incumbent_model"] == "pinned-model"


def test_sample_count_above_the_cap_is_rejected(
    admin_client: TestClient, consenting_user: User, _launch: MagicMock
) -> None:
    with patch.object(settings, "model_comparison_max_samples", 10):
        response = admin_client.post(
            f"{BASE}/users/{consenting_user.id}/runs", json=_payload(sample_count=500)
        )
    assert response.status_code == 422
    _launch.assert_not_called()


def test_second_concurrent_run_for_the_same_user_conflicts(
    admin_client: TestClient, consenting_user: User, _launch: MagicMock
) -> None:
    first = admin_client.post(f"{BASE}/users/{consenting_user.id}/runs", json=_payload())
    assert first.status_code == 201
    second = admin_client.post(f"{BASE}/users/{consenting_user.id}/runs", json=_payload())
    assert second.status_code == 409
    assert _launch.call_count == 1


def test_global_concurrency_cap_returns_429(
    admin_client: TestClient, consenting_user: User, db_session: Session, _launch: MagicMock
) -> None:
    """Runs share the provider rate limit with live traffic, so the per-user
    guard is not enough on its own."""
    other = User(id=str(uuid.uuid4()), user_id=f"google_{uuid.uuid4().hex[:8]}")
    db_session.add(other)
    db_session.commit()
    db_session.add(
        ComparisonRun(
            user_id=other.id,
            incumbent_provider="anthropic",
            incumbent_model="incumbent-model",
            candidate_provider="anthropic",
            candidate_model="candidate-model",
            requested_samples=10,
            status=str(RunStatus.RUNNING),
        )
    )
    db_session.commit()

    with patch.object(settings, "model_comparison_max_concurrent_runs", 1):
        response = admin_client.post(f"{BASE}/users/{consenting_user.id}/runs", json=_payload())
    assert response.status_code == 429
    assert "limit is 1" in response.json()["detail"]
    _launch.assert_not_called()


# ---------------------------------------------------------------------------
# Listing, reporting, cancelling
# ---------------------------------------------------------------------------


def _make_run(db: Session, user_id: str, **overrides: object) -> ComparisonRun:
    run = ComparisonRun(
        user_id=user_id,
        incumbent_provider="anthropic",
        incumbent_model="incumbent-model",
        candidate_provider="anthropic",
        candidate_model="candidate-model",
        requested_samples=10,
        status=str(RunStatus.COMPLETED),
    )
    for key, value in overrides.items():
        setattr(run, key, value)
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def test_list_runs_spans_every_user_by_default(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """Unfiltered is the useful default: it is how a run is found again.

    An operator looking for last week's evaluation does not necessarily
    remember which user it was against.
    """
    other = User(id=str(uuid.uuid4()), user_id="other-user")
    db_session.add(other)
    db_session.commit()
    # The email lives on Subscription, and the base fixture user has no row,
    # so both sides of the join are covered: one with an email, one without.
    db_session.add(Subscription(user_id=other.id, email="other@example.com"))
    db_session.commit()
    _make_run(db_session, consenting_user.id)
    _make_run(db_session, other.id)

    body = admin_client.get(f"{BASE}/runs").json()
    assert body["total"] == 2
    owners = {r["user_id"] for r in body["runs"]}
    assert owners == {consenting_user.id, other.id}
    # The listing names whose run each row is, or the table cannot be read.
    # A user with no subscription row reports an empty email rather than
    # dropping out of the listing.
    assert {r["user_email"] for r in body["runs"]} == {"other@example.com", ""}


def test_list_runs_filters_to_one_user(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    other = User(id=str(uuid.uuid4()), user_id="other-user")
    db_session.add(other)
    db_session.commit()
    _make_run(db_session, consenting_user.id)
    _make_run(db_session, other.id)

    body = admin_client.get(f"{BASE}/runs?user_id={consenting_user.id}").json()
    assert body["total"] == 1
    assert body["runs"][0]["user_id"] == consenting_user.id


def test_the_list_reports_the_page_ceiling_it_enforces(
    admin_client: TestClient, consenting_user: User
) -> None:
    """On the wire for the same reason ``max_samples`` is.

    The console grows its page size as the operator asks for more rows. With
    no ceiling to clamp to it walks past the server's and gets a 422, and
    since the poll closes over that size, every later tick fails too.
    """
    response = admin_client.get(f"{BASE}/runs")
    assert response.status_code == 200
    ceiling = response.json()["max_page_size"]
    assert ceiling >= 25

    assert admin_client.get(f"{BASE}/runs?limit={ceiling}").status_code == 200
    assert admin_client.get(f"{BASE}/runs?limit={ceiling + 1}").status_code == 422


def test_progress_reports_counters_without_writing_an_audit_row(
    admin_client: TestClient, consenting_user: User, db_session: Session, _launch: MagicMock
) -> None:
    """The console polls this every couple of seconds while a run is in flight.

    Auditing it would bury one human read under hundreds of rows, so it is
    exempt, which is only defensible because it returns no conversation
    content and no email.
    """
    created = admin_client.post(f"{BASE}/users/{consenting_user.id}/runs", json=_payload()).json()

    before = db_session.query(AdminAuditLog).count()
    response = admin_client.get(f"{BASE}/runs/{created['id']}/progress")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["id"]
    assert set(body) == {
        "id",
        "status",
        "progress_completed",
        "progress_total",
    }

    db_session.commit()
    assert db_session.query(AdminAuditLog).count() == before


def test_progress_for_an_unknown_run_is_404(admin_client: TestClient) -> None:
    assert admin_client.get(f"{BASE}/runs/{uuid.uuid4()}/progress").status_code == 404


def test_list_runs_404s_an_unknown_user_filter(admin_client: TestClient) -> None:
    """An empty page would read as "never evaluated" rather than "no such user"."""
    assert admin_client.get(f"{BASE}/runs?user_id={uuid.uuid4()}").status_code == 404


def test_list_runs_pages(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    for _ in range(3):
        _make_run(db_session, consenting_user.id)

    body = admin_client.get(f"{BASE}/runs?limit=2").json()
    assert len(body["runs"]) == 2
    assert body["total"] == 3
    assert len(admin_client.get(f"{BASE}/runs?limit=2&offset=2").json()["runs"]) == 1


def test_list_runs_keeps_a_run_whose_user_withdrew_consent_and_says_so(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """A row is run metadata, not conversation content, so it survives.

    The report does not: it is consent-gated, which is why the row carries the
    flag rather than the console offering a link that 403s.
    """
    run = _make_run(db_session, consenting_user.id)
    user = db_session.get(User, consenting_user.id)
    assert user is not None
    user.data_sharing_consent = False
    db_session.commit()

    body = admin_client.get(f"{BASE}/runs").json()
    assert [r["id"] for r in body["runs"]] == [run.public_id]
    assert body["runs"][0]["user_consented"] is False
    # And the evidence really is refused.
    assert admin_client.get(f"{BASE}/runs/{run.public_id}").status_code == 403


def test_a_run_is_addressed_by_a_public_id_not_its_row_id(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """The report URL is pasted and bookmarked, so it must not be a row counter.

    The integer primary key stays internal: the worker and the turn rows use
    it, and it must not be reachable from the API.
    """
    run = _make_run(db_session, consenting_user.id)
    assert admin_client.get(f"{BASE}/runs/{run.id}").status_code == 404

    body = admin_client.get(f"{BASE}/runs/{run.public_id}").json()
    assert body["run"]["id"] == run.public_id
    assert body["run"]["id"] != str(run.id)


def test_list_runs_reports_the_bounds_the_run_form_has_to_respect(
    admin_client: TestClient, consenting_user: User
) -> None:
    """The sample control cannot hold its own ceiling.

    ``MODEL_COMPARISON_MAX_SAMPLES`` is configurable and ``start_run`` enforces it, so
    a client guessing the cap offers sizes the API rejects with a bare 422.
    """
    with patch.object(settings, "model_comparison_max_samples", 40):
        response = admin_client.get(f"{BASE}/runs?user_id={consenting_user.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["max_samples"] == 40
    # And nothing resembling a verdict threshold, because there is no verdict.
    assert "min_turns_for_verdict" not in body


def test_report_orders_the_most_concerning_turns_first(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    run = _make_run(db_session, consenting_user.id)
    db_session.add_all(
        [
            ComparisonTurn(
                run_id=run.id,
                message_seq=1,
                user_message="matched turn",
                outcome=str(TurnOutcome.NO_WRITE),
            ),
            ComparisonTurn(
                run_id=run.id,
                message_seq=2,
                user_message="same write, different arguments",
                outcome=str(TurnOutcome.WRITE_ARGS_DIFFER),
            ),
            ComparisonTurn(
                run_id=run.id,
                message_seq=3,
                user_message="unsafe turn",
                outcome=str(TurnOutcome.WRITE_MISSED),
                findings=json.dumps([{"finding": "unknown_tool", "tool_name": "nope"}]),
            ),
            ComparisonTurn(
                run_id=run.id,
                message_seq=4,
                user_message="did not make the write",
                outcome=str(TurnOutcome.WRITE_MISSED),
            ),
        ]
    )
    db_session.commit()

    response = admin_client.get(f"{BASE}/runs/{run.public_id}")
    assert response.status_code == 200
    order = [t["message_seq"] for t in response.json()["turns"]]
    # The violation first, then the write the candidate never made, then the
    # one it made differently, with the quiet turn last.
    assert order == [3, 4, 2, 1]


def test_report_pages_turns_and_reports_the_total(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """Every text column on a turn is decrypted and redacted per request, so a
    run's evidence is paged rather than shipped whole."""
    run = _make_run(db_session, consenting_user.id)
    db_session.add_all(
        [
            ComparisonTurn(
                run_id=run.id,
                message_seq=i,
                user_message=f"turn {i}",
                outcome=str(TurnOutcome.NO_WRITE),
            )
            for i in range(1, 8)
        ]
    )
    db_session.commit()

    first = admin_client.get(f"{BASE}/runs/{run.public_id}?limit=3").json()
    assert len(first["turns"]) == 3
    assert first["total_turns"] == 7

    second = admin_client.get(f"{BASE}/runs/{run.public_id}?limit=3&offset=3").json()
    assert len(second["turns"]) == 3
    # Pages must not overlap, or "show more" would repeat turns.
    assert {t["message_seq"] for t in first["turns"]}.isdisjoint(
        t["message_seq"] for t in second["turns"]
    )


def test_report_orders_before_paging(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """The first page has to be the worst turns, not an arbitrary three."""
    run = _make_run(db_session, consenting_user.id)
    rows = [
        ComparisonTurn(
            run_id=run.id,
            message_seq=i,
            user_message=f"clean {i}",
            outcome=str(TurnOutcome.NO_WRITE),
        )
        for i in range(1, 7)
    ]
    rows.append(
        ComparisonTurn(
            run_id=run.id,
            message_seq=99,
            user_message="the bad one",
            outcome=str(TurnOutcome.WRITE_MISSED),
            findings=json.dumps([{"finding": "unknown_tool", "tool_name": "nope"}]),
        )
    )
    db_session.add_all(rows)
    db_session.commit()

    page = admin_client.get(f"{BASE}/runs/{run.public_id}?limit=1").json()
    assert page["turns"][0]["message_seq"] == 99


def test_report_redacts_pii_in_message_bodies(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    run = _make_run(db_session, consenting_user.id)
    db_session.add(
        ComparisonTurn(
            run_id=run.id,
            message_seq=1,
            user_message="call me at +15555550123",
            outcome=str(TurnOutcome.NO_WRITE),
            candidate_tool_calls=json.dumps(
                [{"name": "send_message", "arguments": {"to": "jane.doe@example.com"}}]
            ),
        )
    )
    db_session.commit()

    turn = admin_client.get(f"{BASE}/runs/{run.public_id}").json()["turns"][0]
    assert "+15555550123" not in turn["user_message"]
    assert "[PHONE]" in turn["user_message"]
    assert turn["candidate_tool_calls"][0]["arguments"]["to"] == "[EMAIL]"


def test_report_for_unknown_run_is_404(admin_client: TestClient) -> None:
    assert admin_client.get(f"{BASE}/runs/{uuid.uuid4()}").status_code == 404


def test_cancel_flips_an_active_run(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    run = _make_run(db_session, consenting_user.id, status=str(RunStatus.RUNNING))
    response = admin_client.post(f"{BASE}/runs/{run.public_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == RunStatus.CANCELLED


def test_cancelling_a_finished_run_conflicts(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    run = _make_run(db_session, consenting_user.id, status=str(RunStatus.COMPLETED))
    assert admin_client.post(f"{BASE}/runs/{run.public_id}/cancel").status_code == 409


def test_delete_removes_the_run_and_its_turns(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """The turn rows go with the run, via the FK's ON DELETE CASCADE.

    Worth asserting rather than assuming: nothing in Python deletes them, so
    a migration that recreated the constraint without the cascade would leave
    orphaned turns behind and this is the only thing that would notice.
    """
    run = _make_run(db_session, consenting_user.id)
    db_session.add(
        ComparisonTurn(
            run_id=run.id,
            message_seq=1,
            user_message="evidence",
            outcome=str(TurnOutcome.NO_WRITE),
        )
    )
    db_session.commit()
    run_pk = run.id

    assert admin_client.delete(f"{BASE}/runs/{run.public_id}").status_code == 204

    db_session.expire_all()
    assert db_session.get(ComparisonRun, run_pk) is None
    assert db_session.query(ComparisonTurn).filter(ComparisonTurn.run_id == run_pk).count() == 0


def test_delete_leaves_other_runs_alone(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    doomed = _make_run(db_session, consenting_user.id)
    keeper = _make_run(db_session, consenting_user.id)

    assert admin_client.delete(f"{BASE}/runs/{doomed.public_id}").status_code == 204

    db_session.expire_all()
    assert db_session.get(ComparisonRun, keeper.id) is not None


@pytest.mark.parametrize("status", [RunStatus.PENDING, RunStatus.RUNNING])
def test_deleting_an_active_run_conflicts(
    admin_client: TestClient,
    consenting_user: User,
    db_session: Session,
    status: RunStatus,
) -> None:
    """Stopping a run is a decision the operator has to make explicitly.

    Its workers are mid-flight against a paid provider, so deleting under
    them throws that spend away for a result nobody asked to abandon.
    """
    run = _make_run(db_session, consenting_user.id, status=str(status))
    response = admin_client.delete(f"{BASE}/runs/{run.public_id}")
    assert response.status_code == 409
    assert "Cancel it before deleting" in response.json()["detail"]

    db_session.expire_all()
    assert db_session.get(ComparisonRun, run.id) is not None


def test_delete_is_not_consent_gated(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """A withdrawn-consent run is the one most worth removing.

    Its report already 403s, so gating the delete too would pin an
    unreadable row in the list with no way to clear it.
    """
    run = _make_run(db_session, consenting_user.id)
    user = db_session.get(User, consenting_user.id)
    assert user is not None
    user.data_sharing_consent = False
    db_session.commit()

    assert admin_client.get(f"{BASE}/runs/{run.public_id}").status_code == 403
    assert admin_client.delete(f"{BASE}/runs/{run.public_id}").status_code == 204


def test_deleting_an_unknown_run_is_404(admin_client: TestClient) -> None:
    assert admin_client.delete(f"{BASE}/runs/{uuid.uuid4()}").status_code == 404


def test_delete_writes_an_audit_row(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """The audit row is the only record left once the run is gone."""
    run = _make_run(db_session, consenting_user.id)
    assert admin_client.delete(f"{BASE}/runs/{run.public_id}").status_code == 204

    row = (
        db_session.query(AdminAuditLog)
        .filter(AdminAuditLog.action == AdminAction.DELETE_MODEL_COMPARISON_RUN)
        .one()
    )
    assert row.target_user_id == consenting_user.id


# ---------------------------------------------------------------------------
# Report ordering and accounting
# ---------------------------------------------------------------------------


def test_report_ranks_violations_above_fixture_artifacts(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """A fixture note must not outrank the turn an operator came to read.

    ``tool_not_in_schema`` is a property of the replayed history, but keying
    the sort on "has any finding" put five of them on the first screen and
    pushed a genuine unrequested write below the fold.
    """
    run = _make_run(db_session, consenting_user.id)
    db_session.add_all(
        [
            ComparisonTurn(
                run_id=run.id,
                message_seq=1,
                user_message="retired tool in the history",
                outcome=str(TurnOutcome.NO_WRITE),
                findings=json.dumps([{"finding": "tool_not_in_schema", "tool_name": "retired"}]),
            ),
            ComparisonTurn(
                run_id=run.id,
                message_seq=2,
                user_message="wrote something nobody asked for",
                outcome=str(TurnOutcome.NO_WRITE),
                findings=json.dumps([{"finding": "unrequested_write", "tool_name": "qb_update"}]),
            ),
            ComparisonTurn(
                run_id=run.id,
                message_seq=3,
                user_message="never made the write",
                outcome=str(TurnOutcome.WRITE_MISSED),
            ),
            ComparisonTurn(
                run_id=run.id,
                message_seq=4,
                user_message="quiet turn",
                outcome=str(TurnOutcome.NO_WRITE),
            ),
        ]
    )
    db_session.commit()

    response = admin_client.get(f"{BASE}/runs/{run.public_id}")
    assert response.status_code == 200
    assert [t["message_seq"] for t in response.json()["turns"]] == [2, 3, 1, 4]


def test_report_says_whose_finding_each_one_is(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """Production is checked too; its findings must not read as the candidate's."""
    run = _make_run(db_session, consenting_user.id)
    db_session.add_all(
        [
            ComparisonTurn(
                run_id=run.id,
                message_seq=1,
                user_message="the live turn guessed",
                outcome=str(TurnOutcome.WRITE_MATCHED),
                findings=json.dumps(
                    [{"finding": "fabricated_id", "tool_name": "add_note", "side": "production"}]
                ),
            ),
            ComparisonTurn(
                run_id=run.id,
                message_seq=2,
                user_message="the candidate guessed",
                outcome=str(TurnOutcome.WRITE_MATCHED),
                findings=json.dumps(
                    [{"finding": "fabricated_id", "tool_name": "add_note", "side": "candidate"}]
                ),
            ),
        ]
    )
    db_session.commit()

    body = admin_client.get(f"{BASE}/runs/{run.public_id}").json()
    turns = body["turns"]
    # The candidate's ranks first: production's is the comparison, not
    # evidence against the candidate.
    assert [t["message_seq"] for t in turns] == [2, 1]
    assert turns[0]["findings"][0]["side"] == "candidate"
    assert turns[0]["findings"][0]["violation"] is True
    assert turns[1]["findings"][0]["side"] == "production"


def test_report_shows_what_production_did_beside_the_candidate(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """The baseline is the record, so it has to be on the turn."""
    run = _make_run(db_session, consenting_user.id)
    db_session.add(
        ComparisonTurn(
            run_id=run.id,
            message_seq=1,
            user_message="note the job at 12 Oak St",
            outcome=str(TurnOutcome.WRITE_MATCHED),
            production_reply="Noted.",
            production_tool_calls=json.dumps(
                [
                    {
                        "name": "add_note",
                        "arguments": {"work_order_id": "71002"},
                        "result": "ok",
                        "is_error": False,
                    }
                ]
            ),
            candidate_replayed_lookups=json.dumps(
                [
                    {
                        "name": "search",
                        "arguments": {"q": "12 Oak St"},
                        "result": "work order 71002",
                        "is_error": False,
                    }
                ]
            ),
            write_results=json.dumps(
                [
                    {
                        "tool_name": "add_note",
                        "outcome": "matched",
                        "key_arguments": {"work_order_id": ["71002"]},
                        "candidate_arguments": {"work_order_id": "71002"},
                    }
                ]
            ),
        )
    )
    db_session.commit()

    turn = admin_client.get(f"{BASE}/runs/{run.public_id}").json()["turns"][0]
    assert turn["production_reply"] == "Noted."
    assert turn["production_tool_calls"][0]["name"] == "add_note"
    assert turn["production_tool_calls"][0]["result"] == "ok"
    (lookup,) = turn["candidate_replayed_lookups"]
    assert lookup["result"] == "work order 71002"
    assert turn["writes"][0]["outcome"] == "matched"


def test_a_summary_without_a_cost_serves_null_not_zero(
    admin_client: TestClient, consenting_user: User, db_session: Session
) -> None:
    """A zero reads as a measurement, and for a gateway model name it is not."""
    run = _make_run(
        db_session,
        consenting_user.id,
        summary_json={
            "turns_total": 1,
            "turns_replayed": 1,
            "turns_failed": 0,
            "outcome_counts": {"no_write": 1},
            "candidate_findings": {},
            "production_findings": {},
            "candidate_violations": 0,
            "production_violations": 0,
            "production_checked_findings": ["fabricated_id"],
            "writes_total": 0,
            "writes_matched": 0,
            "writes_args_differ": 0,
            "writes_missed": 0,
            "write_match_rate": 0.0,
            "candidate": {
                "provider": "anthropic",
                "model": "gw/candidate",
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "billed_prompt_tokens": 10,
                "total_cost_usd": None,
                "cost_unavailable_reason": "endpoint",
                "latency_p50_ms": 1.0,
                "latency_p95_ms": 2.0,
            },
            "notes": ["Cost is not available: endpoint otari is marked unpriced."],
        },
    )

    summary = admin_client.get(f"{BASE}/runs/{run.public_id}").json()["run"]["summary"]
    assert summary["candidate"]["total_cost_usd"] is None
    assert summary["candidate"]["cost_unavailable_reason"] == "endpoint"
    assert "recommendation" not in summary
