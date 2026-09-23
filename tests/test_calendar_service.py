"""Tests for GoogleCalendarService."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.integrations.calendar.provider import (
    CalendarEventCreate,
    CalendarEventUpdate,
)
from backend.app.integrations.calendar.service import GoogleCalendarService


def _mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
) -> httpx.Response:
    """Create a mock httpx response."""
    return httpx.Response(
        status_code,
        json=json_data or {},
        request=httpx.Request("GET", "https://example.com"),
    )


@pytest.fixture()
def service() -> GoogleCalendarService:
    return GoogleCalendarService(access_token="test-access-token")


# ---------------------------------------------------------------------------
# list_events
# ---------------------------------------------------------------------------


async def test_list_events_returns_events(service: GoogleCalendarService) -> None:
    """Should parse Google Calendar event items into CalendarEventData."""
    api_response = {
        "items": [
            {
                "id": "abc123",
                "summary": "Job: Smith Remodel",
                "description": "Kitchen work",
                "start": {"dateTime": "2026-03-25T09:00:00+00:00"},
                "end": {"dateTime": "2026-03-25T17:00:00+00:00"},
                "location": "123 Oak St",
                "status": "confirmed",
            }
        ]
    }

    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        events = await service.list_events(
            "primary",
            datetime(2026, 3, 25, tzinfo=UTC),
            datetime(2026, 3, 26, tzinfo=UTC),
        )

    assert len(events) == 1
    assert events[0].id == "abc123"
    assert events[0].title == "Job: Smith Remodel"
    assert events[0].location == "123 Oak St"


async def test_list_events_empty(service: GoogleCalendarService) -> None:
    """Should return empty list when no events."""
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = {"items": []}
        events = await service.list_events(
            "primary",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        )

    assert events == []


async def test_list_events_encodes_calendar_id(service: GoogleCalendarService) -> None:
    """Calendar IDs with '#' or '@' must be percent-encoded in the URL path."""
    cal_id = "en.usa#holiday@group.v.calendar.google.com"
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = {"items": []}
        await service.list_events(
            cal_id,
            datetime(2026, 3, 25, tzinfo=UTC),
            datetime(2026, 3, 26, tzinfo=UTC),
        )

    path = mock_req.call_args[0][1]
    assert "#" not in path
    assert "en.usa%23holiday%40group.v.calendar.google.com" in path


async def test_list_events_skips_malformed_events(service: GoogleCalendarService) -> None:
    """Should skip malformed events instead of crashing the entire list."""
    api_response = {
        "items": [
            {
                "id": "good-event",
                "summary": "Valid Event",
                "start": {"dateTime": "2026-03-25T09:00:00+00:00"},
                "end": {"dateTime": "2026-03-25T17:00:00+00:00"},
                "status": "confirmed",
            },
            {
                "id": "bad-event",
                "summary": "Missing start/end",
                "start": {},
                "end": {},
            },
        ]
    }

    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        events = await service.list_events(
            "primary",
            datetime(2026, 3, 25, tzinfo=UTC),
            datetime(2026, 3, 26, tzinfo=UTC),
        )

    assert len(events) == 1
    assert events[0].id == "good-event"


async def test_list_events_api_error(service: GoogleCalendarService) -> None:
    """Should propagate API errors."""
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("GET", "https://example.com"),
            response=_mock_response(500),
        )
        with pytest.raises(httpx.HTTPStatusError):
            await service.list_events(
                "primary",
                datetime(2026, 3, 25, tzinfo=UTC),
                datetime(2026, 3, 26, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# all-day event parsing
# ---------------------------------------------------------------------------


async def test_list_events_all_day_returns_tz_aware(service: GoogleCalendarService) -> None:
    """All-day events should have timezone-aware datetimes (not naive)."""
    api_response = {
        "items": [
            {
                "id": "all-day-1",
                "summary": "Day Off",
                "start": {"date": "2026-03-25"},
                "end": {"date": "2026-03-26"},
                "status": "confirmed",
            }
        ]
    }

    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        events = await service.list_events(
            "primary",
            datetime(2026, 3, 25, tzinfo=UTC),
            datetime(2026, 3, 27, tzinfo=UTC),
        )

    assert len(events) == 1
    assert events[0].all_day is True
    assert events[0].start.tzinfo is not None
    assert events[0].end.tzinfo is not None


# ---------------------------------------------------------------------------
# create_event
# ---------------------------------------------------------------------------


async def test_create_event_success(service: GoogleCalendarService) -> None:
    """Should create an event and return parsed data."""
    api_response = {
        "id": "new-event-id",
        "summary": "Job: Test",
        "start": {"dateTime": "2026-03-28T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-28T17:00:00+00:00"},
        "status": "confirmed",
    }

    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        event = await service.create_event(
            "primary",
            CalendarEventCreate(
                title="Job: Test",
                start=datetime(2026, 3, 28, 9, 0, tzinfo=UTC),
                end=datetime(2026, 3, 28, 17, 0, tzinfo=UTC),
            ),
        )

    assert event.id == "new-event-id"
    assert event.title == "Job: Test"


async def test_create_event_no_reminder_omits_field(service: GoogleCalendarService) -> None:
    """Without reminder_minutes_before, the request body must omit `reminders`.

    Google then applies the user's default Calendar reminders. Sending an
    explicit `reminders.useDefault=False` with no overrides would silence
    notifications entirely, which is not what the legacy callers expect.
    """
    api_response = {
        "id": "x",
        "summary": "T",
        "start": {"dateTime": "2026-03-28T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-28T17:00:00+00:00"},
        "status": "confirmed",
    }
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        await service.create_event(
            "primary",
            CalendarEventCreate(
                title="T",
                start=datetime(2026, 3, 28, 9, 0, tzinfo=UTC),
                end=datetime(2026, 3, 28, 17, 0, tzinfo=UTC),
            ),
        )
        body = mock_req.call_args.kwargs["json"]
        assert "reminders" not in body


async def test_create_event_with_reminder_at_start(service: GoogleCalendarService) -> None:
    """reminder_minutes_before=0 must produce a popup override at exact start (#1067)."""
    api_response = {
        "id": "x",
        "summary": "T",
        "start": {"dateTime": "2026-03-28T14:00:00+00:00"},
        "end": {"dateTime": "2026-03-28T14:05:00+00:00"},
        "status": "confirmed",
    }
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        await service.create_event(
            "primary",
            CalendarEventCreate(
                title="Call client",
                start=datetime(2026, 3, 28, 14, 0, tzinfo=UTC),
                end=datetime(2026, 3, 28, 14, 5, tzinfo=UTC),
                reminder_minutes_before=0,
            ),
        )
        body = mock_req.call_args.kwargs["json"]
        assert body["reminders"] == {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": 0}],
        }


async def test_create_event_with_reminder_minutes_before(
    service: GoogleCalendarService,
) -> None:
    """Positive reminder_minutes_before must produce a popup override N min ahead."""
    api_response = {
        "id": "x",
        "summary": "T",
        "start": {"dateTime": "2026-03-28T14:00:00+00:00"},
        "end": {"dateTime": "2026-03-28T15:00:00+00:00"},
        "status": "confirmed",
    }
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        await service.create_event(
            "primary",
            CalendarEventCreate(
                title="T",
                start=datetime(2026, 3, 28, 14, 0, tzinfo=UTC),
                end=datetime(2026, 3, 28, 15, 0, tzinfo=UTC),
                reminder_minutes_before=15,
            ),
        )
        body = mock_req.call_args.kwargs["json"]
        assert body["reminders"]["overrides"][0]["minutes"] == 15
        assert body["reminders"]["useDefault"] is False


async def test_create_event_api_error(service: GoogleCalendarService) -> None:
    """Should propagate API errors on create."""
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=httpx.Request("POST", "https://example.com"),
            response=_mock_response(400),
        )
        with pytest.raises(httpx.HTTPStatusError):
            await service.create_event(
                "primary",
                CalendarEventCreate(
                    title="Test",
                    start=datetime(2026, 3, 28, 9, 0, tzinfo=UTC),
                    end=datetime(2026, 3, 28, 17, 0, tzinfo=UTC),
                ),
            )


# ---------------------------------------------------------------------------
# update_event
# ---------------------------------------------------------------------------


async def test_update_event_success(service: GoogleCalendarService) -> None:
    """Should update an event and return parsed data."""
    api_response = {
        "id": "evt-001",
        "summary": "Updated Title",
        "start": {"dateTime": "2026-03-25T10:00:00+00:00"},
        "end": {"dateTime": "2026-03-25T18:00:00+00:00"},
        "status": "confirmed",
    }

    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        event = await service.update_event(
            "primary",
            "evt-001",
            CalendarEventUpdate(title="Updated Title"),
        )

    assert event.title == "Updated Title"
    mock_req.assert_called_once()
    call_args = mock_req.call_args
    assert call_args[0][0] == "PATCH"
    assert "evt-001" in call_args[0][1]


async def test_update_event_not_found(service: GoogleCalendarService) -> None:
    """Should raise on 404."""
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=httpx.Request("PATCH", "https://example.com"),
            response=_mock_response(404),
        )
        with pytest.raises(httpx.HTTPStatusError):
            await service.update_event(
                "primary",
                "nonexistent",
                CalendarEventUpdate(title="X"),
            )


# ---------------------------------------------------------------------------
# delete_event
# ---------------------------------------------------------------------------


async def test_delete_event_success(service: GoogleCalendarService) -> None:
    """Should delete without error."""
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = None  # 204
        await service.delete_event("primary", "evt-001")

    mock_req.assert_called_once()
    assert "evt-001" in mock_req.call_args[0][1]


async def test_delete_event_not_found(service: GoogleCalendarService) -> None:
    """Should raise on 404."""
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=httpx.Request("DELETE", "https://example.com"),
            response=_mock_response(404),
        )
        with pytest.raises(httpx.HTTPStatusError):
            await service.delete_event("primary", "nonexistent")


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------


async def test_check_availability_busy(service: GoogleCalendarService) -> None:
    """Should return busy slots."""
    api_response = {
        "calendars": {
            "primary": {
                "busy": [
                    {
                        "start": "2026-03-25T09:00:00+00:00",
                        "end": "2026-03-25T17:00:00+00:00",
                    }
                ]
            }
        }
    }

    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        slots = await service.check_availability(
            "primary",
            datetime(2026, 3, 25, tzinfo=UTC),
            datetime(2026, 3, 26, tzinfo=UTC),
        )

    assert len(slots) == 1
    assert slots[0].start.hour == 9


async def test_check_availability_email_key(service: GoogleCalendarService) -> None:
    """Should find busy slots even when Google returns email as key instead of 'primary'."""
    api_response = {
        "calendars": {
            "user@gmail.com": {
                "busy": [
                    {
                        "start": "2026-03-25T09:00:00+00:00",
                        "end": "2026-03-25T17:00:00+00:00",
                    }
                ]
            }
        }
    }

    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        slots = await service.check_availability(
            "primary",
            datetime(2026, 3, 25, tzinfo=UTC),
            datetime(2026, 3, 26, tzinfo=UTC),
        )

    assert len(slots) == 1
    assert slots[0].start.hour == 9


async def test_check_availability_free(service: GoogleCalendarService) -> None:
    """Should return empty list when free."""
    api_response = {"calendars": {"primary": {"busy": []}}}

    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = api_response
        slots = await service.check_availability(
            "primary",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        )

    assert slots == []


async def test_check_availability_api_error(service: GoogleCalendarService) -> None:
    """Should propagate API errors."""
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.HTTPStatusError(
            "Error",
            request=httpx.Request("POST", "https://example.com"),
            response=_mock_response(500),
        )
        with pytest.raises(httpx.HTTPStatusError):
            await service.check_availability(
                "primary",
                datetime(2026, 3, 25, tzinfo=UTC),
                datetime(2026, 3, 26, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"


def _events_client(*responses: httpx.Response) -> tuple[Any, AsyncMock]:
    """Patch target and client that answer ``request`` with *responses* in order.

    Each call's Authorization header is recorded on ``mock_client.sent_auth``
    at call time, since the service reuses one headers dict across the retry.
    """
    mock_client = AsyncMock()
    queue = list(responses)
    sent_auth: list[str] = []

    async def _request(*args: Any, **kwargs: Any) -> httpx.Response:
        sent_auth.append(kwargs["headers"]["Authorization"])
        return queue.pop(0)

    mock_client.request.side_effect = _request
    mock_client.sent_auth = sent_auth
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=mock_client), mock_client


def _resp(status: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", _EVENTS_URL))


async def _list(svc: GoogleCalendarService) -> list[Any]:
    return await svc.list_events(
        "primary", datetime(2026, 3, 25, tzinfo=UTC), datetime(2026, 3, 26, tzinfo=UTC)
    )


async def test_401_refreshes_through_the_hook_and_retries() -> None:
    refresh = AsyncMock(return_value="fresh-token")
    svc = GoogleCalendarService(access_token="expired-token", refresh_access_token=refresh)
    patcher, client = _events_client(_resp(401, {}), _resp(200, {"items": []}))

    with patcher:
        assert await _list(svc) == []

    refresh.assert_awaited_once_with("expired-token")
    assert client.sent_auth == ["Bearer expired-token", "Bearer fresh-token"]
    # Later calls on the same service reuse the fresh token.
    assert svc._access_token == "fresh-token"


async def test_no_refresh_without_a_401() -> None:
    refresh = AsyncMock(return_value="fresh-token")
    svc = GoogleCalendarService(access_token="tok", refresh_access_token=refresh)
    patcher, _ = _events_client(_resp(200, {"items": []}))

    with patcher:
        await _list(svc)

    refresh.assert_not_awaited()


async def test_401_persisting_after_refresh_raises() -> None:
    svc = GoogleCalendarService(
        access_token="old", refresh_access_token=AsyncMock(return_value="new")
    )
    patcher, _ = _events_client(_resp(401, {}), _resp(401, {}))

    with patcher, pytest.raises(httpx.HTTPStatusError) as exc_info:
        await _list(svc)
    assert exc_info.value.response.status_code == 401


async def test_401_stands_when_no_refresh_could_run() -> None:
    refresh = AsyncMock(return_value=None)
    svc = GoogleCalendarService(access_token="old", refresh_access_token=refresh)
    patcher, client = _events_client(_resp(401, {}))

    with patcher, pytest.raises(httpx.HTTPStatusError):
        await _list(svc)
    assert client.request.await_count == 1


async def test_refresh_failure_propagates() -> None:
    """The hook's own errors (ReconnectRequired, a token endpoint 5xx) reach the caller."""
    boom = RuntimeError("token endpoint down")
    svc = GoogleCalendarService(
        access_token="old", refresh_access_token=AsyncMock(side_effect=boom)
    )
    patcher, _ = _events_client(_resp(401, {}))

    with patcher, pytest.raises(RuntimeError):
        await _list(svc)


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


async def test_timeout_handling(service: GoogleCalendarService) -> None:
    """Should propagate timeout exceptions."""
    with patch.object(service, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.TimeoutException("Connection timeout")
        with pytest.raises(httpx.TimeoutException):
            await service.list_events(
                "primary",
                datetime(2026, 3, 25, tzinfo=UTC),
                datetime(2026, 3, 26, tzinfo=UTC),
            )
