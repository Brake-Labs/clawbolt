"""Regression: reconnecting Google Calendar must resync the saved calendar config.

Production incident: calendars saved from one Google account kept 404ing on
every read and write after the user reconnected Google Calendar (the operator
had swapped OAuth clients, and the reconnect may have used another account).
The saved ``calendar_configs`` rows were never refreshed, so the agent kept
targeting ids the new connection could not see.

These tests drive the real OAuth callback path (``oauth_service`` singleton,
where the calendar integration registers its post-connect hook) with Google
mocked at the HTTP boundary, and assert on the rows left behind.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select

# Imported for its side effect: registers the calendar tool factory and its
# post-connect hook on ``oauth_service``.
import backend.app.integrations.calendar.factory  # noqa: F401
from backend.app.database import db_session_async
from backend.app.integrations.calendar.provider import CalendarInfo
from backend.app.integrations.calendar.service import GoogleCalendarService
from backend.app.models import CalendarConfig, User
from backend.app.services.oauth import OAuthConfig, oauth_service

OLD_PRIMARY = "old-owner@example.com"
OLD_CREW = "crew-old@group.calendar.google.com"
OLD_JOBS = "jobs-old@group.calendar.google.com"
HOLIDAYS = "en.usa#holiday@group.v.calendar.google.com"
NEW_PRIMARY = "new-owner@example.com"

_CALENDAR_CONFIG = OAuthConfig(
    integration="google_calendar",
    client_id="cid",
    client_secret="csec",
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    use_pkce=False,
)


@pytest_asyncio.fixture()
async def user() -> User:
    async with db_session_async() as db:
        u = User(user_id="calendar-resync-user", onboarding_complete=True)
        db.add(u)
        await db.commit()
        await db.refresh(u)
        db.expunge(u)
    return u


async def _save_april_config(user_id: str) -> None:
    """The config saved from the original account: two crew calendars, the
    primary (by its email id), and public holidays, with user choices."""
    async with db_session_async() as db:
        for cid, name, role, disabled, primary in (
            (OLD_PRIMARY, "Personal", "owner", "", True),
            (OLD_CREW, "Crew", "owner", json.dumps(["calendar_delete_event"]), False),
            (OLD_JOBS, "Jobs", "owner", "", False),
            (HOLIDAYS, "Holidays", "reader", json.dumps(["calendar_list_events"]), False),
        ):
            db.add(
                CalendarConfig(
                    user_id=user_id,
                    provider="google_calendar",
                    calendar_id=cid,
                    display_name=name,
                    access_role=role,
                    disabled_tools=disabled,
                    is_primary=primary,
                )
            )
        await db.commit()


async def _rows(user_id: str) -> dict[str, CalendarConfig]:
    async with db_session_async() as db:
        rows = (
            (
                await db.execute(
                    select(CalendarConfig).filter_by(user_id=user_id, provider="google_calendar")
                )
            )
            .scalars()
            .all()
        )
    return {r.calendar_id: r for r in rows}


def _token_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"access_token": "new-at", "refresh_token": "new-rt", "expires_in": 3600},
        request=httpx.Request("POST", "https://oauth2.googleapis.com/token"),
    )


def _primary_calendar_response(email: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"id": email},
        request=httpx.Request("GET", "https://www.googleapis.com/calendar/v3/x"),
    )


@pytest.fixture()
def oauth_http() -> Iterator[MagicMock]:
    """Mock the OAuth service's HTTP client: token exchange and account lookup."""
    client = MagicMock()
    client.post = AsyncMock(return_value=_token_response())
    client.get = AsyncMock(return_value=_primary_calendar_response(NEW_PRIMARY))
    with patch.object(oauth_service, "_get_http", return_value=client):
        yield client


async def _complete_connect(user_id: str, live: list[CalendarInfo] | Exception) -> None:
    url = oauth_service.get_authorization_url(_CALENDAR_CONFIG, user_id)
    state = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["state"][0]
    list_mock = (
        AsyncMock(side_effect=live) if isinstance(live, Exception) else AsyncMock(return_value=live)
    )
    with (
        patch("backend.app.services.oauth.get_oauth_config", return_value=_CALENDAR_CONFIG),
        patch.object(GoogleCalendarService, "list_calendars", list_mock),
    ):
        await oauth_service.handle_callback(state, "auth-code")


async def test_reconnect_with_other_account_drops_unreachable_and_enables_new_primary(
    user: User, oauth_http: MagicMock
) -> None:
    """The incident: saved owned calendars are not visible to the new connection.

    They must be removed (not left to 404), the still-visible holidays calendar
    kept with the user's per-calendar choices, and the new account's primary
    enabled so the user is not left with nothing usable.
    """
    await _save_april_config(user.id)

    await _complete_connect(
        user.id,
        [
            CalendarInfo(id=NEW_PRIMARY, summary="Work", primary=True, access_role="owner"),
            CalendarInfo(id=HOLIDAYS, summary="Holidays in United States", access_role="reader"),
        ],
    )

    rows = await _rows(user.id)
    assert set(rows) == {HOLIDAYS, NEW_PRIMARY}
    # The user's per-calendar choice on the surviving calendar is preserved.
    assert json.loads(rows[HOLIDAYS].disabled_tools) == ["calendar_list_events"]
    assert rows[HOLIDAYS].is_primary is False
    assert rows[NEW_PRIMARY].is_primary is True
    assert rows[NEW_PRIMARY].access_role == "owner"


async def test_reconnect_same_account_keeps_choices_and_refreshes_roles(
    user: User, oauth_http: MagicMock
) -> None:
    """Every saved calendar still visible: nothing removed or added, user
    choices intact, role and name refreshed from Google."""
    await _save_april_config(user.id)
    oauth_http.get.return_value = _primary_calendar_response(OLD_PRIMARY)

    await _complete_connect(
        user.id,
        [
            CalendarInfo(id=OLD_PRIMARY, summary="Personal", primary=True, access_role="owner"),
            CalendarInfo(id=OLD_CREW, summary="Crew (renamed)", access_role="reader"),
            CalendarInfo(id=OLD_JOBS, summary="Jobs", access_role="writer"),
            CalendarInfo(id=HOLIDAYS, summary="Holidays", access_role="reader"),
            CalendarInfo(id="unpicked@group.calendar.google.com", summary="Unpicked"),
        ],
    )

    rows = await _rows(user.id)
    assert set(rows) == {OLD_PRIMARY, OLD_CREW, OLD_JOBS, HOLIDAYS}
    assert rows[OLD_CREW].access_role == "reader"
    assert rows[OLD_CREW].display_name == "Crew (renamed)"
    assert json.loads(rows[OLD_CREW].disabled_tools) == ["calendar_delete_event"]
    assert rows[OLD_JOBS].access_role == "writer"
    assert rows[OLD_PRIMARY].is_primary is True


async def test_resync_failure_does_not_break_connect(user: User, oauth_http: MagicMock) -> None:
    """A Calendar API failure during resync is logged; the token is still stored
    and the saved rows are left as they were."""
    await _save_april_config(user.id)

    await _complete_connect(user.id, httpx.ConnectError("boom"))

    token = await oauth_service.load_token_uncached(user.id, "google_calendar")
    assert token is not None
    assert token.access_token == "new-at"
    assert set(await _rows(user.id)) == {OLD_PRIMARY, OLD_CREW, OLD_JOBS, HOLIDAYS}
