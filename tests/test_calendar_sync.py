"""Unit tests for the calendar resync plan and the Calendar API calls it relies on."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.integrations.calendar.provider import CalendarInfo
from backend.app.integrations.calendar.service import (
    CalendarAvailabilityError,
    CalendarListTruncatedError,
    CalendarNotVisibleError,
    GoogleCalendarService,
)
from backend.app.integrations.calendar.sync import plan_calendar_resync

PRIMARY = CalendarInfo(id="owner@example.com", summary="Owner", primary=True, access_role="owner")
CREW = CalendarInfo(id="crew@group.calendar.google.com", summary="Crew", access_role="writer")


def test_plan_keeps_visible_and_removes_missing() -> None:
    plan = plan_calendar_resync(
        [PRIMARY.id, CREW.id, "gone@group.calendar.google.com"], [PRIMARY, CREW]
    )
    assert plan is not None
    assert set(plan.updated) == {PRIMARY.id, CREW.id}
    assert plan.removed == ["gone@group.calendar.google.com"]
    # Primary survived, so nothing is added.
    assert plan.added_primary is None
    assert plan.primary_id == PRIMARY.id


def test_plan_adds_primary_when_stale_rows_dropped() -> None:
    plan = plan_calendar_resync(["gone@example.com", CREW.id], [PRIMARY, CREW])
    assert plan is not None
    assert plan.removed == ["gone@example.com"]
    assert plan.added_primary == PRIMARY


def test_plan_adds_primary_on_fresh_connect() -> None:
    plan = plan_calendar_resync([], [PRIMARY, CREW])
    assert plan is not None
    assert plan.added_primary == PRIMARY
    assert plan.removed == []


def test_plan_respects_primary_left_off_when_nothing_dropped() -> None:
    """Same account, user deliberately enabled only a crew calendar."""
    plan = plan_calendar_resync([CREW.id], [PRIMARY, CREW])
    assert plan is not None
    assert plan.added_primary is None
    assert plan.removed == []


def test_plan_treats_primary_alias_as_always_visible() -> None:
    plan = plan_calendar_resync(["primary"], [PRIMARY])
    assert plan is not None
    assert plan.removed == []
    assert plan.added_primary is None


def test_plan_refuses_listing_without_primary() -> None:
    """Every account has a primary calendar; a listing without one is not
    trusted enough to delete saved rows against."""
    assert plan_calendar_resync([CREW.id], []) is None
    assert plan_calendar_resync([CREW.id], [CREW]) is None


@pytest.fixture()
def service() -> GoogleCalendarService:
    return GoogleCalendarService(access_token="at")


async def test_list_calendars_follows_pages_and_shows_hidden(
    service: GoogleCalendarService,
) -> None:
    pages = [
        {"items": [{"id": "a@example.com", "primary": True}], "nextPageToken": "p2"},
        {"items": [{"id": "b@example.com", "accessRole": "reader"}]},
    ]
    with patch.object(service, "_request", new_callable=AsyncMock, side_effect=pages) as req:
        cals = await service.list_calendars(show_hidden=True)

    assert [c.id for c in cals] == ["a@example.com", "b@example.com"]
    assert req.call_count == 2
    assert req.call_args_list[0].kwargs["params"] == {"maxResults": "250", "showHidden": "true"}
    assert req.call_args_list[1].kwargs["params"] == {
        "maxResults": "250",
        "showHidden": "true",
        "pageToken": "p2",
    }


async def test_list_calendars_raises_instead_of_truncating(
    service: GoogleCalendarService,
) -> None:
    """A listing cut off at the page cap must not look complete: the resync
    would delete every saved calendar on the pages it never read."""
    page = {"items": [{"id": "a@example.com", "primary": True}], "nextPageToken": "more"}
    with (
        patch.object(service, "_request", new_callable=AsyncMock, return_value=page),
        pytest.raises(CalendarListTruncatedError),
    ):
        await service.list_calendars(show_hidden=True)


async def test_check_availability_raises_when_calendar_not_visible(
    service: GoogleCalendarService,
) -> None:
    """freeBusy answers 200 with a per-calendar notFound error; that must not
    read as 'free'."""
    body = {
        "calendars": {"gone@example.com": {"errors": [{"domain": "global", "reason": "notFound"}]}}
    }
    with (
        patch.object(service, "_request", new_callable=AsyncMock, return_value=body),
        pytest.raises(CalendarNotVisibleError),
    ):
        await service.check_availability(
            "gone@example.com", datetime(2026, 3, 25, tzinfo=UTC), datetime(2026, 3, 26, tzinfo=UTC)
        )


async def test_check_availability_raises_on_other_per_calendar_errors(
    service: GoogleCalendarService,
) -> None:
    """Any per-calendar freeBusy error comes with no busy data, so it must not
    read as 'free' either."""
    body = {
        "calendars": {
            "a@example.com": {"errors": [{"domain": "global", "reason": "internalError"}]}
        }
    }
    with (
        patch.object(service, "_request", new_callable=AsyncMock, return_value=body),
        pytest.raises(CalendarAvailabilityError),
    ):
        await service.check_availability(
            "a@example.com", datetime(2026, 3, 25, tzinfo=UTC), datetime(2026, 3, 26, tzinfo=UTC)
        )
