"""Google Calendar REST API client using httpx.

Follows the same pattern as the QuickBooks service: a reactive 401 retry
through the shared, locked OAuth refresh, and no dependency on
google-api-python-client.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from backend.app.integrations.calendar.provider import (
    BusySlot,
    CalendarEventCreate,
    CalendarEventData,
    CalendarEventUpdate,
    CalendarInfo,
)
from backend.app.services.oauth import TokenRefreshUnavailable

logger = logging.getLogger(__name__)

GOOGLE_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
# calendarList page size. Google defaults to 100 and caps it at 250.
_CALENDAR_LIST_PAGE_SIZE = 250

# Upper bound on calendarList pages followed in one listing (5000 calendars),
# which keeps a misbehaving pagination loop finite. Hitting it raises rather
# than returning a truncated list, because the reconnect resync deletes saved
# calendars missing from the listing.
_MAX_CALENDAR_LIST_PAGES = 20


class CalendarListTruncatedError(Exception):
    """calendarList still had pages left after ``_MAX_CALENDAR_LIST_PAGES``."""


class CalendarNotVisibleError(Exception):
    """The calendar id is not visible to the connected Google account.

    Raised where Google reports this inside a successful response (freeBusy)
    rather than as an HTTP 404, so callers can treat both the same way.
    """

    def __init__(self, calendar_id: str) -> None:
        super().__init__("Calendar not visible to the connected account")
        self.calendar_id = calendar_id


class CalendarAvailabilityError(Exception):
    """freeBusy reported a per-calendar error other than notFound.

    The busy list is empty in that case too, so it must not read as free.
    """


def _encode_cal_id(calendar_id: str) -> str:
    """URL-encode a calendar ID for use in API paths.

    Google Calendar IDs can contain ``#`` and ``@`` which must be
    percent-encoded when embedded in URL paths (e.g.
    ``en.usa#holiday@group.v.calendar.google.com``).
    """
    return quote(calendar_id, safe="")


class GoogleCalendarService:
    """Google Calendar REST API client."""

    def __init__(
        self,
        access_token: str,
        refresh_access_token: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> None:
        """``refresh_access_token`` is called with the access token Google just
        answered 401 to and returns a fresh one, or None when no refresh could
        run. It owns the OAuth side (locking, persistence, retiring a dead
        grant) and raises ``ReconnectRequired`` when the grant is dead.
        """
        self._access_token = access_token
        self._refresh_access_token = refresh_access_token

    @property
    def provider_name(self) -> str:
        return "google_calendar"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Make an authenticated request to the Google Calendar API.

        Returns the parsed JSON body, or None for 204 responses. On a 401 it
        refreshes the token once and retries. Raises ``ReconnectRequired``
        when the refresh finds the grant dead, and ``TokenRefreshUnavailable``
        when no refresh could run (the lock was contended). A 401 that
        persists after a refresh that did run is raised as the
        ``HTTPStatusError`` it is.
        """
        url = f"{GOOGLE_CALENDAR_API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(method, url, headers=headers, json=json, params=params)

            if resp.status_code == 401 and self._refresh_access_token is not None:
                new_token = await self._refresh_access_token(self._access_token)
                if not new_token:
                    raise TokenRefreshUnavailable(
                        "Google Calendar rejected the access token and no refresh could run",
                        request=resp.request,
                        response=resp,
                    )
                self._access_token = new_token
                headers["Authorization"] = f"Bearer {new_token}"
                resp = await client.request(method, url, headers=headers, json=json, params=params)

            resp.raise_for_status()

            if resp.status_code == 204:
                return None
            return resp.json()

    # -- Public API -----------------------------------------------------------

    async def list_calendars(self, *, show_hidden: bool = False) -> list[CalendarInfo]:
        """List calendars visible to the authenticated user.

        Follows ``nextPageToken`` so an account with more calendars than one
        page holds is listed in full: callers that compare saved config
        against this list (the reconnect resync) would otherwise treat the
        calendars on later pages as unreachable. *show_hidden* includes
        calendars the user hid in Google Calendar's sidebar, which are still
        readable and writable through the API. Raises
        ``CalendarListTruncatedError`` rather than return a partial list.
        """
        calendars: list[CalendarInfo] = []
        page_token = ""
        for _ in range(_MAX_CALENDAR_LIST_PAGES):
            params: dict[str, str] = {"maxResults": str(_CALENDAR_LIST_PAGE_SIZE)}
            if show_hidden:
                params["showHidden"] = "true"
            if page_token:
                params["pageToken"] = page_token
            data = await self._request("GET", "/users/me/calendarList", params=params)
            body = data or {}
            calendars.extend(
                CalendarInfo(
                    id=item.get("id", ""),
                    summary=item.get("summary", ""),
                    primary=item.get("primary", False),
                    access_role=item.get("accessRole", ""),
                )
                for item in body.get("items", [])
            )
            page_token = body.get("nextPageToken", "")
            if not isinstance(page_token, str) or not page_token:
                return calendars
        raise CalendarListTruncatedError(f"calendarList exceeded {_MAX_CALENDAR_LIST_PAGES} pages")

    async def list_events(
        self,
        calendar_id: str,
        time_min: datetime,
        time_max: datetime,
    ) -> list[CalendarEventData]:
        """List events in a calendar within a time range."""
        params = {
            "timeMin": _to_rfc3339(time_min),
            "timeMax": _to_rfc3339(time_max),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": "250",
        }
        data = await self._request(
            "GET", f"/calendars/{_encode_cal_id(calendar_id)}/events", params=params
        )
        items = (data or {}).get("items", [])
        events: list[CalendarEventData] = []
        for item in items:
            try:
                events.append(_parse_event(item))
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed calendar event %s: %s", item.get("id"), exc)
        return events

    async def create_event(
        self,
        calendar_id: str,
        event: CalendarEventCreate,
    ) -> CalendarEventData:
        """Create a new event on a calendar."""
        body = _build_event_body(event)
        data = await self._request(
            "POST", f"/calendars/{_encode_cal_id(calendar_id)}/events", json=body
        )
        return _parse_event(data or {})

    async def update_event(
        self,
        calendar_id: str,
        event_id: str,
        updates: CalendarEventUpdate,
    ) -> CalendarEventData:
        """Update an existing event (PATCH semantics)."""
        body: dict[str, Any] = {}
        if updates.title is not None:
            body["summary"] = updates.title
        if updates.description is not None:
            body["description"] = updates.description
        if updates.location is not None:
            body["location"] = updates.location
        if updates.start is not None:
            body["start"] = {"dateTime": _to_rfc3339(updates.start)}
        if updates.end is not None:
            body["end"] = {"dateTime": _to_rfc3339(updates.end)}

        data = await self._request(
            "PATCH",
            f"/calendars/{_encode_cal_id(calendar_id)}/events/{event_id}",
            json=body,
        )
        return _parse_event(data or {})

    async def delete_event(
        self,
        calendar_id: str,
        event_id: str,
    ) -> None:
        """Delete an event from a calendar."""
        await self._request("DELETE", f"/calendars/{_encode_cal_id(calendar_id)}/events/{event_id}")

    async def check_availability(
        self,
        calendar_id: str,
        time_min: datetime,
        time_max: datetime,
    ) -> list[BusySlot]:
        """Check free/busy information for a calendar."""
        body = {
            "timeMin": _to_rfc3339(time_min),
            "timeMax": _to_rfc3339(time_max),
            "items": [{"id": calendar_id}],
        }
        data = await self._request("POST", "/freeBusy", json=body)
        calendars = (data or {}).get("calendars", {})
        # Google may return the resolved email as key instead of "primary",
        # so collect busy slots from all calendars in the response (we only
        # requested one).
        busy_list: list[dict[str, str]] = []
        for cal_data in calendars.values():
            # freeBusy answers 200 even when the calendar is not visible to
            # this account, and reports it per calendar as
            # ``errors: [{"reason": "notFound"}]`` with an empty busy list.
            # Reading that as "free" would tell the user an unreachable
            # calendar has no conflicts.
            errors = cal_data.get("errors") or []
            if any(isinstance(e, dict) and e.get("reason") == "notFound" for e in errors):
                raise CalendarNotVisibleError(calendar_id)
            # Any other reason (internalError, or one Google adds later) also
            # comes with no busy data.
            if errors:
                raise CalendarAvailabilityError(
                    "Google could not report free/busy for this calendar right now"
                )
            busy_list.extend(cal_data.get("busy", []))
        return [
            BusySlot(
                start=datetime.fromisoformat(slot["start"]),
                end=datetime.fromisoformat(slot["end"]),
            )
            for slot in busy_list
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_rfc3339(dt: datetime) -> str:
    """Convert a datetime to RFC 3339 format for the Google API."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _parse_event(item: dict[str, Any]) -> CalendarEventData:
    """Parse a Google Calendar API event item into a CalendarEventData."""
    start_raw = item.get("start", {})
    end_raw = item.get("end", {})

    all_day = "date" in start_raw and "dateTime" not in start_raw

    if all_day:
        start = datetime.fromisoformat(start_raw["date"]).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_raw.get("date", start_raw["date"])).replace(tzinfo=UTC)
    else:
        start = datetime.fromisoformat(start_raw.get("dateTime", ""))
        end = datetime.fromisoformat(end_raw.get("dateTime", ""))

    return CalendarEventData(
        id=item.get("id", ""),
        title=item.get("summary", "(No title)"),
        description=item.get("description", ""),
        start=start,
        end=end,
        location=item.get("location", ""),
        all_day=all_day,
        status=item.get("status", "confirmed"),
    )


def _build_event_body(event: CalendarEventCreate) -> dict[str, Any]:
    """Build a Google Calendar API event body from a CalendarEventCreate DTO."""
    body: dict[str, Any] = {
        "summary": event.title,
    }
    if event.description:
        body["description"] = event.description
    if event.location:
        body["location"] = event.location

    if event.all_day:
        body["start"] = {"date": event.start.strftime("%Y-%m-%d")}
        body["end"] = {"date": event.end.strftime("%Y-%m-%d")}
    else:
        body["start"] = {"dateTime": _to_rfc3339(event.start)}
        body["end"] = {"dateTime": _to_rfc3339(event.end)}

    # When reminder_minutes_before is set (including 0 for fire-at-start),
    # request a single popup reminder. Leave the field off entirely so the
    # user's default Calendar reminders apply. The Calendar API caps minutes
    # at 40320 (4 weeks); validation lives on the Pydantic model.
    if event.reminder_minutes_before is not None:
        body["reminders"] = {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": event.reminder_minutes_before},
            ],
        }

    return body
