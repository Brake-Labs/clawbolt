"""Tests for Google Calendar tools."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.app.agent.approval import PermissionLevel
from backend.app.agent.tool_errors import build_error_hint
from backend.app.agent.tools.base import Tool, ToolErrorKind
from backend.app.agent.tools.names import ToolName
from backend.app.agent.tools.registry import ToolContext
from backend.app.integrations.calendar.factory import (
    _calendar_factory,
    _calendar_not_visible_result,
    _format_event_list,
    _handle_http_error,
    _parse_dt,
    _resolve_tz,
    create_calendar_tools,
)
from backend.app.integrations.calendar.provider import (
    BusySlot,
    CalendarEventCreate,
    CalendarEventData,
    CalendarEventUpdate,
    CalendarInfo,
)
from backend.app.models import User
from backend.app.services.pii_redaction import redact_pii
from tests.mocks.google_calendar import MockGoogleCalendarService

# Tools that are disabled to make a calendar "read-only".
_WRITE_TOOLS = [
    ToolName.CALENDAR_CREATE_EVENT,
    ToolName.CALENDAR_UPDATE_EVENT,
    ToolName.CALENDAR_DELETE_EVENT,
]

# Default enabled calendars for tests: both mock calendars enabled with full access.
_DEFAULT_ENABLED: list[tuple[str, str, list[str], str]] = [
    ("primary", "Personal", [], "owner"),
    ("jobs@example.com", "Jobs", [], "writer"),
]


@pytest.fixture()
def cal_service() -> MockGoogleCalendarService:
    return MockGoogleCalendarService()


@pytest.fixture()
def cal_tools(cal_service: MockGoogleCalendarService) -> list[Tool]:
    return create_calendar_tools(cal_service, enabled_calendars=_DEFAULT_ENABLED)


def _get_tool(tools: list[Tool], name: str) -> Tool:
    for t in tools:
        if t.name == name:
            return t
    msg = f"Tool {name} not found"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


async def test_factory_returns_empty_when_not_configured() -> None:
    """_calendar_factory should return [] when client_id/secret are empty."""
    ctx = MagicMock(spec=ToolContext)
    user = MagicMock(spec=User)
    user.id = "1"
    ctx.user = user

    with patch("backend.app.integrations.calendar.factory.settings") as mock_settings:
        mock_settings.google_calendar_client_id = ""
        mock_settings.google_calendar_client_secret = ""
        assert await _calendar_factory(ctx) == []


async def test_factory_returns_empty_when_not_connected() -> None:
    """_calendar_factory should return [] when user has no OAuth token."""
    ctx = MagicMock(spec=ToolContext)
    user = MagicMock(spec=User)
    user.id = "1"
    ctx.user = user

    with (
        patch("backend.app.integrations.calendar.factory.settings") as mock_settings,
        patch("backend.app.integrations.calendar.factory.oauth_service") as mock_oauth,
    ):
        mock_settings.google_calendar_client_id = "test-id"
        mock_settings.google_calendar_client_secret = "test-secret"
        mock_oauth.get_valid_token = AsyncMock(return_value=None)

        tools = await _calendar_factory(ctx)

    assert tools == []


async def test_factory_returns_6_tools_when_configured() -> None:
    """_calendar_factory should return 6 tools when configured and connected."""
    ctx = MagicMock(spec=ToolContext)
    user = MagicMock(spec=User)
    user.id = "1"
    ctx.user = user

    mock_token = MagicMock()
    mock_token.access_token = "test-access"
    mock_token.refresh_token = "test-refresh"
    mock_token.expires_at = 9999999999.0

    with (
        patch("backend.app.integrations.calendar.factory.settings") as mock_settings,
        patch("backend.app.integrations.calendar.factory.oauth_service") as mock_oauth,
        patch(
            "backend.app.integrations.calendar.factory._get_enabled_calendars",
            new_callable=AsyncMock,
            return_value=[("primary", "Primary", [], "owner")],
        ),
        patch(
            "backend.app.integrations.calendar.factory._get_primary_calendar_id",
            new_callable=AsyncMock,
            return_value="",
        ),
    ):
        mock_settings.google_calendar_client_id = "test-id"
        mock_settings.google_calendar_client_secret = "test-secret"
        mock_oauth.get_valid_token = AsyncMock(return_value=mock_token)

        tools = await _calendar_factory(ctx)

    assert len(tools) == 6


# ---------------------------------------------------------------------------
# Tool count and metadata
# ---------------------------------------------------------------------------


def test_calendar_tools_count(cal_tools: list[Tool]) -> None:
    """create_calendar_tools should return 6 tools."""
    assert len(cal_tools) == 6


def test_calendar_tools_have_params_model(cal_tools: list[Tool]) -> None:
    """All calendar tools must have a params_model set."""
    for tool in cal_tools:
        assert tool.params_model is not None, f"Tool {tool.name} missing params_model"


def test_calendar_tools_names(cal_tools: list[Tool]) -> None:
    """Verify all expected tool names are present."""
    names = {t.name for t in cal_tools}
    assert names == {
        ToolName.CALENDAR_LIST_CALENDARS,
        ToolName.CALENDAR_LIST_EVENTS,
        ToolName.CALENDAR_CREATE_EVENT,
        ToolName.CALENDAR_UPDATE_EVENT,
        ToolName.CALENDAR_DELETE_EVENT,
        ToolName.CALENDAR_CHECK_AVAILABILITY,
    }


# ---------------------------------------------------------------------------
# Approval policies
# ---------------------------------------------------------------------------


def test_list_calendars_is_always(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_CALENDARS)
    assert tool.approval_policy is not None
    assert tool.approval_policy.default_level == PermissionLevel.ALWAYS


def test_list_events_is_ask(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_EVENTS)
    assert tool.approval_policy is not None
    assert tool.approval_policy.default_level == PermissionLevel.ASK


def test_create_event_is_ask(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CREATE_EVENT)
    assert tool.approval_policy is not None
    assert tool.approval_policy.default_level == PermissionLevel.ASK


def test_update_event_is_ask(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_UPDATE_EVENT)
    assert tool.approval_policy is not None
    assert tool.approval_policy.default_level == PermissionLevel.ASK


def test_delete_event_is_ask(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_DELETE_EVENT)
    assert tool.approval_policy is not None
    assert tool.approval_policy.default_level == PermissionLevel.ASK


def test_check_availability_is_ask(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    assert tool.approval_policy is not None
    assert tool.approval_policy.default_level == PermissionLevel.ASK


@pytest.mark.parametrize(
    ("tool_name", "expected_resource"),
    [
        (ToolName.CALENDAR_CREATE_EVENT, "create event"),
        (ToolName.CALENDAR_UPDATE_EVENT, "update event"),
        (ToolName.CALENDAR_DELETE_EVENT, "delete event"),
    ],
)
def test_mutation_resource_is_coarse_not_per_event(
    cal_tools: list[Tool], tool_name: str, expected_resource: str
) -> None:
    """Create/update/delete scope approvals per action type, not per event_id.

    Regression for issue #1449: a per-event_id resource made "always allow"
    (and the within-turn approval cache) miss for every other event in a
    batch, so rescheduling a multi-day job re-prompted on each day even
    after the user chose "always allow". A coarse resource fixes that and
    keeps all three mutation tools consistent.
    """
    tool = _get_tool(cal_tools, tool_name)
    assert tool.approval_policy is not None
    assert tool.approval_policy.resource_extractor is not None
    # Two different events resolve to the same resource, so one approval covers both.
    assert tool.approval_policy.resource_extractor({"event_id": "evt-A"}) == expected_resource
    assert tool.approval_policy.resource_extractor({"event_id": "evt-B"}) == expected_resource


# ---------------------------------------------------------------------------
# Description builders
# ---------------------------------------------------------------------------


def test_list_events_description_builder(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_EVENTS)
    assert tool.approval_policy is not None
    assert tool.approval_policy.description_builder is not None
    desc = tool.approval_policy.description_builder(
        {"start_date": "2026-03-31T00:00:00", "end_date": "2026-03-31T23:59:59"}
    )
    assert "Read calendar events" in desc
    assert "2026-03-31" in desc


def test_check_availability_description_builder(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    assert tool.approval_policy is not None
    assert tool.approval_policy.description_builder is not None
    desc = tool.approval_policy.description_builder(
        {"start_date": "2026-03-31T00:00:00", "end_date": "2026-04-01T00:00:00"}
    )
    assert "Check calendar availability" in desc


def test_create_event_description_builder(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CREATE_EVENT)
    assert tool.approval_policy is not None
    assert tool.approval_policy.description_builder is not None
    desc = tool.approval_policy.description_builder({"title": "Job: Smith Remodel"})
    assert "Smith Remodel" in desc


def test_update_event_description_builder(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_UPDATE_EVENT)
    assert tool.approval_policy is not None
    assert tool.approval_policy.description_builder is not None
    # With title: shows the title
    desc = tool.approval_policy.description_builder({"event_id": "evt-001", "title": "Job: Smith"})
    assert "Job: Smith" in desc
    # Without title: generic description, no raw event ID
    desc_no_title = tool.approval_policy.description_builder({"event_id": "evt-001"})
    assert "evt-001" not in desc_no_title
    assert desc_no_title == "Update a calendar event"


def test_delete_event_description_builder(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_DELETE_EVENT)
    assert tool.approval_policy is not None
    assert tool.approval_policy.description_builder is not None
    desc = tool.approval_policy.description_builder({"event_id": "evt-002"})
    # Should not expose raw event ID
    assert "evt-002" not in desc
    assert desc == "Delete a calendar event"


# ---------------------------------------------------------------------------
# list_calendars
# ---------------------------------------------------------------------------


async def test_list_calendars_shows_enabled(cal_tools: list[Tool]) -> None:
    """Should return enabled calendars (not all Google calendars)."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_CALENDARS)
    result = await tool.function()
    assert result.is_error is False
    assert "2 enabled calendar(s)" in result.content
    assert "Personal" in result.content
    assert "Jobs" in result.content


async def test_list_calendars_single() -> None:
    """With a single enabled calendar, should show just that one."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service, enabled_calendars=[("jobs@example.com", "Jobs", [], "writer")]
    )
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_CALENDARS)
    result = await tool.function()
    assert result.is_error is False
    assert "1 enabled calendar(s)" in result.content
    assert "Jobs" in result.content


async def test_list_calendars_default_primary() -> None:
    """With no enabled_calendars, should default to primary."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service)
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_CALENDARS)
    result = await tool.function()
    assert result.is_error is False
    assert "Primary" in result.content


# ---------------------------------------------------------------------------
# list_events -- multi-calendar merge
# ---------------------------------------------------------------------------


async def test_list_events_multi_calendar_merge(cal_tools: list[Tool]) -> None:
    """Should merge events from all enabled calendars with labels."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-27T23:59:59",
    )
    assert result.is_error is False
    assert "2 event(s)" in result.content
    assert "Smith Kitchen Remodel" in result.content
    assert "Jones Roof Repair" in result.content
    # Multi-cal labels should be present
    assert "[Personal]" in result.content
    assert "[Jobs]" in result.content


async def test_list_events_single_calendar_no_label() -> None:
    """With a single enabled calendar, events should not have labels."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=[("primary", "Personal", [], "owner")])
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-27T23:59:59",
    )
    assert result.is_error is False
    assert "1 event(s)" in result.content
    assert "Smith Kitchen Remodel" in result.content
    assert "[Personal]" not in result.content


async def test_list_events_specific_calendar(cal_tools: list[Tool]) -> None:
    """Specifying a calendar_id should only query that calendar."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-27T23:59:59",
        calendar_id="primary",
    )
    assert result.is_error is False
    assert "1 event(s)" in result.content
    assert "Smith Kitchen Remodel" in result.content
    assert "Jones Roof Repair" not in result.content


async def test_list_events_invalid_calendar(cal_tools: list[Tool]) -> None:
    """Should reject a calendar_id not in the enabled set."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-27T23:59:59",
        calendar_id="not-enabled@example.com",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION
    assert "not in the enabled set" in result.content


async def test_list_events_no_results(cal_tools: list[Tool]) -> None:
    """Should handle empty result set."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-01-01T00:00:00",
        end_date="2026-01-02T23:59:59",
    )
    assert result.is_error is False
    assert "No events found" in result.content


async def test_list_events_invalid_date(cal_tools: list[Tool]) -> None:
    """Should reject invalid date format."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="not-a-date",
        end_date="2026-03-27T23:59:59",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION


async def test_list_events_api_error(cal_service: MockGoogleCalendarService) -> None:
    """Should handle API errors gracefully."""

    async def failing(*args: object, **kwargs: object) -> list:
        raise RuntimeError("API connection failed")

    cal_service.list_events = failing  # type: ignore[assignment]
    tools = create_calendar_tools(cal_service)
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-27T23:59:59",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.SERVICE


# ---------------------------------------------------------------------------
# create_event -- validation
# ---------------------------------------------------------------------------


async def test_create_event_auto_select_single_calendar() -> None:
    """With one enabled calendar, should auto-select it."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=[("primary", "Personal", [], "owner")])
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Job: Test - Plumbing",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is False
    assert result.content.startswith("ok")


async def test_create_event_requires_calendar_id_multi(cal_tools: list[Tool]) -> None:
    """With multiple enabled calendars and no primary marked, must specify calendar_id."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Job: Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION
    assert "Multiple calendars available" in result.content


async def test_create_event_uses_primary_when_id_omitted(
    cal_service: MockGoogleCalendarService,
) -> None:
    """When primary_calendar_id is set and the LLM omits calendar_id, the
    tool resolves to the primary instead of erroring on ambiguity. This
    is the case the contractor with crew sub-calendars hits constantly:
    "add to my calendar" used to fail because a personal calendar and
    several per-crew-member calendars were all enabled and the LLM
    didn't pick one.
    """
    tools = create_calendar_tools(
        cal_service,
        enabled_calendars=_DEFAULT_ENABLED,
        primary_calendar_id="primary",
    )
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Job: Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is False, result.content
    # Mock service records the calendar_id each event was created on.
    new_event = cal_service.events[-1]
    assert cal_service._event_calendar_map[new_event.id] == "primary"


async def test_create_event_falls_through_when_primary_not_in_allowed_set(
    cal_service: MockGoogleCalendarService,
) -> None:
    """If the configured primary is not in the enabled-and-allowed set
    (e.g. user revoked write access on it), the tiebreaker silently
    declines and the historical "Multiple calendars" error returns. This
    keeps the change strictly additive: no new failure modes.
    """
    tools = create_calendar_tools(
        cal_service,
        enabled_calendars=_DEFAULT_ENABLED,
        primary_calendar_id="not-in-enabled-set@example.com",
    )
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Job: Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is True
    assert "Multiple calendars available" in result.content


async def test_validate_calendar_id_prefers_primary() -> None:
    """Direct test of the helper: when calendar_id is empty and primary
    is in the allowed set, return it instead of the ambiguity error.
    """
    from backend.app.integrations.calendar.factory import _validate_calendar_id

    enabled: list[tuple[str, str, list[str], str]] = [
        ("primary", "Personal", [], "owner"),
        ("jobs@example.com", "Jobs", [], "writer"),
    ]
    resolved, err = _validate_calendar_id(
        "",
        enabled,
        tool_name=ToolName.CALENDAR_CREATE_EVENT,
        primary_calendar_id="primary",
    )
    assert err is None
    assert resolved == "primary"


async def test_validate_calendar_id_single_calendar_unaffected() -> None:
    """Single-calendar users still auto-resolve regardless of primary
    flag (preserves the historical fast path).
    """
    from backend.app.integrations.calendar.factory import _validate_calendar_id

    enabled: list[tuple[str, str, list[str], str]] = [("primary", "Personal", [], "owner")]
    resolved, err = _validate_calendar_id(
        "",
        enabled,
        tool_name=ToolName.CALENDAR_CREATE_EVENT,
        primary_calendar_id="",  # No primary flag, single cal still resolves.
    )
    assert err is None
    assert resolved == "primary"


async def test_create_event_validates_calendar_id(cal_tools: list[Tool]) -> None:
    """Should reject a calendar_id not in the enabled set."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Job: Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
        calendar_id="not-enabled@example.com",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION
    assert "not in the enabled set" in result.content


async def test_create_event_happy_path(cal_tools: list[Tool]) -> None:
    """Should create an event when calendar_id is specified."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Job: Test - Plumbing",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
        location="789 Main St",
        calendar_id="primary",
    )
    assert result.is_error is False
    assert result.content.startswith("ok")
    # Title and times are intentionally NOT in content -- they live in the
    # ToolReceipt below to stop the LLM from pattern-matching the structured
    # content into a fabricated bullet that doubles the receipt block.
    assert "Test - Plumbing" not in result.content
    assert result.receipt is not None
    assert result.receipt.target.startswith("Job: Test - Plumbing on 2026-03-28")


async def test_create_event_content_is_minimal(cal_tools: list[Tool]) -> None:
    """The LLM-visible content for create_event must omit title and dates.

    Regression for the prod calendar bug observed 2026-04-29: when create_event
    returned content like "Event created: Lunch with Tam | 2026-04-30 12:00 -
    13:00 | id: abc", the LLM pattern-matched the formatted layout into a
    fabricated "- Created Google Calendar event: Lunch with Tam\\n  Thu Apr
    30, 12:00 PM" bullet that doubled the receipt block. Match the CompanyCam
    convention: title and dates only in the ToolReceipt, never in content.
    """
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Lunch with Tam",
        start="2026-04-30T12:00:00",
        end="2026-04-30T13:00:00",
        calendar_id="primary",
    )
    assert result.is_error is False
    # Content must not contain anything the LLM could mirror as a fake receipt.
    assert "Lunch with Tam" not in result.content
    assert "2026-04-30" not in result.content
    assert "12:00" not in result.content
    assert "|" not in result.content
    # But the receipt has the full record so the user sees it.
    assert result.receipt is not None
    assert "Lunch with Tam" in result.receipt.target
    assert "2026-04-30" in result.receipt.target


async def test_update_event_content_is_minimal() -> None:
    """update_event content must also exclude title/dates (same reason as create)."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=[("primary", "Personal", [], "owner")])
    tool = _get_tool(tools, ToolName.CALENDAR_UPDATE_EVENT)
    result = await tool.function(
        event_id="evt-001",
        title="Lunch with Tam (revised)",
    )
    assert result.is_error is False
    assert "Lunch with Tam" not in result.content
    assert "|" not in result.content
    assert result.receipt is not None
    assert "Lunch with Tam (revised)" in result.receipt.target


async def test_create_event_passes_reminder_minutes_through(
    cal_service: MockGoogleCalendarService,
) -> None:
    """The tool must forward reminder_minutes_before to the service (#1067).

    This is the path the agent takes for 'remind me at 2pm': start=2pm,
    reminder_minutes_before=0. The popup must be wired all the way through
    to the CalendarEventCreate DTO, not silently dropped.
    """
    captured: dict[str, object] = {}
    original = cal_service.create_event

    async def spy(calendar_id: str, event: CalendarEventCreate) -> CalendarEventData:
        captured["reminder_minutes_before"] = event.reminder_minutes_before
        return await original(calendar_id, event)

    cal_service.create_event = spy  # type: ignore[assignment]

    tools = create_calendar_tools(cal_service)
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Call client",
        start="2026-03-28T14:00:00",
        end="2026-03-28T14:05:00",
        reminder_minutes_before=0,
    )
    assert result.is_error is False
    assert captured["reminder_minutes_before"] == 0


async def test_create_event_default_reminder_is_none(
    cal_service: MockGoogleCalendarService,
) -> None:
    """When reminder_minutes_before is not passed it must default to None.

    None tells the service not to set a `reminders` override on the API
    body, which preserves the user's Google Calendar default reminders.
    """
    captured: dict[str, object] = {}
    original = cal_service.create_event

    async def spy(calendar_id: str, event: CalendarEventCreate) -> CalendarEventData:
        captured["reminder_minutes_before"] = event.reminder_minutes_before
        return await original(calendar_id, event)

    cal_service.create_event = spy  # type: ignore[assignment]

    tools = create_calendar_tools(cal_service)
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    await tool.function(
        title="T",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert captured["reminder_minutes_before"] is None


async def test_create_event_invalid_date(cal_tools: list[Tool]) -> None:
    """Should reject invalid date format."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Test Event",
        start="bad-date",
        end="2026-03-28T17:00:00",
        calendar_id="primary",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION


async def test_create_event_end_before_start(cal_tools: list[Tool]) -> None:
    """Should reject end time before start time."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Test Event",
        start="2026-03-28T17:00:00",
        end="2026-03-28T09:00:00",
        calendar_id="primary",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION
    assert "after start" in result.content.lower()


async def test_create_event_api_error(
    cal_service: MockGoogleCalendarService,
) -> None:
    """Should handle API errors gracefully."""

    async def failing(*args: object, **kwargs: object) -> object:
        raise RuntimeError("API error")

    cal_service.create_event = failing  # type: ignore[assignment]
    tools = create_calendar_tools(cal_service)
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.SERVICE


# ---------------------------------------------------------------------------
# update_event
# ---------------------------------------------------------------------------


async def test_update_event_happy_path() -> None:
    """Should update an existing event with single calendar (auto-select)."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=[("primary", "Personal", [], "owner")])
    tool = _get_tool(tools, ToolName.CALENDAR_UPDATE_EVENT)
    result = await tool.function(
        event_id="evt-001",
        title="Job: Smith Kitchen Remodel (Revised)",
    )
    assert result.is_error is False
    assert result.content.startswith("ok")
    # Title is intentionally NOT in content; it lives in the receipt only.
    assert "Revised" not in result.content
    assert result.receipt is not None
    assert "Revised" in result.receipt.target


async def test_update_event_not_found(
    cal_service: MockGoogleCalendarService,
) -> None:
    """Should handle event not found."""
    tools = create_calendar_tools(cal_service)
    tool = _get_tool(tools, ToolName.CALENDAR_UPDATE_EVENT)
    result = await tool.function(
        event_id="nonexistent",
        title="Updated",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.SERVICE


async def test_update_event_invalid_date(cal_tools: list[Tool]) -> None:
    """Should reject invalid date in update."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_UPDATE_EVENT)
    result = await tool.function(
        event_id="evt-001",
        start="bad-date",
        calendar_id="primary",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION


# ---------------------------------------------------------------------------
# delete_event
# ---------------------------------------------------------------------------


async def test_delete_event_happy_path(
    cal_service: MockGoogleCalendarService,
) -> None:
    """Should delete an event."""
    tools = create_calendar_tools(cal_service)
    tool = _get_tool(tools, ToolName.CALENDAR_DELETE_EVENT)
    result = await tool.function(event_id="evt-001")
    assert result.is_error is False
    assert "deleted" in result.content

    # Verify event is gone
    assert len([e for e in cal_service.events if e.id == "evt-001"]) == 0


async def test_delete_event_not_found(
    cal_service: MockGoogleCalendarService,
) -> None:
    """Should handle deleting non-existent event."""
    tools = create_calendar_tools(cal_service)
    tool = _get_tool(tools, ToolName.CALENDAR_DELETE_EVENT)
    result = await tool.function(event_id="nonexistent")
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.SERVICE


# ---------------------------------------------------------------------------
# check_availability -- multi-calendar merge
# ---------------------------------------------------------------------------


async def test_check_availability_busy(cal_tools: list[Tool]) -> None:
    """Should return busy slots from all enabled calendars."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-26T00:00:00",
    )
    assert result.is_error is False
    assert "busy slot(s)" in result.content


async def test_check_availability_free(cal_tools: list[Tool]) -> None:
    """Should report free when no busy slots."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    result = await tool.function(
        start_date="2026-01-01T00:00:00",
        end_date="2026-01-02T00:00:00",
    )
    assert result.is_error is False
    assert "free" in result.content.lower()


async def test_check_availability_invalid_date(cal_tools: list[Tool]) -> None:
    """Should reject invalid date format."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    result = await tool.function(
        start_date="not-valid",
        end_date="2026-03-26T00:00:00",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION


async def test_check_availability_invalid_calendar(cal_tools: list[Tool]) -> None:
    """Should reject a calendar_id not in the enabled set."""
    tool = _get_tool(cal_tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-26T00:00:00",
        calendar_id="not-enabled@example.com",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION


async def test_check_availability_api_error(
    cal_service: MockGoogleCalendarService,
) -> None:
    """Should handle API errors gracefully."""

    async def failing(*args: object, **kwargs: object) -> list:
        raise RuntimeError("API error")

    cal_service.check_availability = failing  # type: ignore[assignment]
    tools = create_calendar_tools(cal_service)
    tool = _get_tool(tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-26T00:00:00",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.SERVICE


# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------


def test_resolve_tz_valid() -> None:
    """_resolve_tz should return a ZoneInfo for valid IANA names."""
    tz = _resolve_tz("America/New_York")
    assert tz.key == "America/New_York"  # type: ignore[union-attr]


def test_resolve_tz_empty_returns_utc() -> None:
    """_resolve_tz should return UTC for empty string."""
    from datetime import UTC

    assert _resolve_tz("") is UTC


def test_resolve_tz_invalid_returns_utc() -> None:
    """_resolve_tz should return UTC for invalid timezone names."""
    from datetime import UTC

    assert _resolve_tz("Not/A/Timezone") is UTC


def test_parse_dt_uses_default_tz() -> None:
    """_parse_dt should use default_tz for naive datetime strings."""
    import zoneinfo

    eastern = zoneinfo.ZoneInfo("America/New_York")
    dt = _parse_dt("2026-03-25T09:00:00", default_tz=eastern)
    assert dt.tzinfo is eastern
    # 9 AM Eastern = 1 PM UTC (EDT is UTC-4)
    assert dt.utctimetuple().tm_hour == 13


def test_parse_dt_preserves_explicit_offset() -> None:
    """_parse_dt should not override an explicit timezone offset."""
    import zoneinfo

    eastern = zoneinfo.ZoneInfo("America/New_York")
    # Pass a string with explicit UTC offset; default_tz should be ignored
    dt = _parse_dt("2026-03-25T09:00:00+00:00", default_tz=eastern)
    assert dt.utctimetuple().tm_hour == 9


def test_parse_dt_defaults_to_utc_when_no_tz() -> None:
    """_parse_dt with no default_tz should fall back to UTC."""
    from datetime import UTC

    dt = _parse_dt("2026-03-25T09:00:00")
    assert dt.tzinfo is UTC


async def test_list_events_respects_user_timezone() -> None:
    """Calendar tools with a user timezone interpret naive dates locally."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service,
        user_timezone="America/New_York",
        enabled_calendars=[("primary", "Personal", [], "owner")],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)

    # "March 25" in Eastern time: midnight to midnight Eastern
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-25T23:59:59",
    )
    assert result.is_error is False
    # The 09:00 UTC event (5 AM ET) is within the Eastern day
    assert "Smith Kitchen Remodel" in result.content


async def test_list_events_utc_default_without_timezone() -> None:
    """Without user timezone, naive dates are interpreted as UTC."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=[("primary", "Personal", [], "owner")])
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)

    # March 25 midnight-to-midnight UTC includes events at 09:00 UTC
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-25T23:59:59",
    )
    assert result.is_error is False
    assert "Smith Kitchen Remodel" in result.content


async def test_factory_passes_enabled_calendars() -> None:
    """_calendar_factory should pass enabled_calendars to create_calendar_tools."""
    ctx = MagicMock(spec=ToolContext)
    user = MagicMock(spec=User)
    user.id = "1"
    user.timezone = "America/New_York"
    ctx.user = user

    mock_token = MagicMock()
    mock_token.access_token = "test-access"
    mock_token.refresh_token = "test-refresh"
    mock_token.expires_at = 9999999999.0

    with (
        patch("backend.app.integrations.calendar.factory.settings") as mock_settings,
        patch("backend.app.integrations.calendar.factory.oauth_service") as mock_oauth,
        patch(
            "backend.app.integrations.calendar.factory.create_calendar_tools",
            wraps=create_calendar_tools,
        ) as mock_create,
        patch(
            "backend.app.integrations.calendar.factory._get_enabled_calendars",
            new_callable=AsyncMock,
            return_value=[
                ("primary", "Personal", [], "owner"),
                ("jobs@example.com", "Jobs", [ToolName.CALENDAR_CREATE_EVENT], "writer"),
            ],
        ),
        patch(
            "backend.app.integrations.calendar.factory._get_primary_calendar_id",
            new_callable=AsyncMock,
            return_value="",
        ),
    ):
        mock_settings.google_calendar_client_id = "test-id"
        mock_settings.google_calendar_client_secret = "test-secret"
        mock_oauth.get_valid_token = AsyncMock(return_value=mock_token)

        await _calendar_factory(ctx)

    mock_create.assert_called_once()
    assert mock_create.call_args.kwargs["user_timezone"] == "America/New_York"
    assert mock_create.call_args.kwargs["enabled_calendars"] == [
        ("primary", "Personal", [], "owner"),
        ("jobs@example.com", "Jobs", [ToolName.CALENDAR_CREATE_EVENT], "writer"),
    ]


# ---------------------------------------------------------------------------
# Per-calendar permissions
# ---------------------------------------------------------------------------

_MIXED_PERMS: list[tuple[str, str, list[str], str]] = [
    ("primary", "Personal", [], "owner"),
    ("jobs@example.com", "Jobs", list(_WRITE_TOOLS), "writer"),
]


async def test_list_calendars_shows_per_tool_access() -> None:
    """calendar_list_calendars should show per-calendar tool access."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=_MIXED_PERMS)
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_CALENDARS)
    result = await tool.function()
    assert result.is_error is False
    assert "allowed: list events, create event, update event, delete event" in result.content
    assert "blocked: create event, update event, delete event" in result.content


async def test_list_calendars_shows_read_only() -> None:
    """Calendars with all per-calendar tools disabled should show READ-ONLY."""
    service = MockGoogleCalendarService()
    all_disabled = [*_WRITE_TOOLS, ToolName.CALENDAR_LIST_EVENTS]
    tools = create_calendar_tools(
        service,
        enabled_calendars=[("primary", "Personal", all_disabled, "owner")],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_CALENDARS)
    result = await tool.function()
    assert result.is_error is False
    assert "allowed: none" in result.content
    assert "blocked: list events, create event, update event, delete event" in result.content


async def test_list_events_reads_from_restricted_calendar() -> None:
    """list_events should query calendars where list_events is not disabled."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=_MIXED_PERMS)
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-27T23:59:59",
    )
    assert result.is_error is False
    # Both calendars queried (list_events is not in the disabled set)
    assert "Smith Kitchen Remodel" in result.content
    assert "Jones Roof Repair" in result.content


async def test_create_event_blocked_on_disabled_calendar() -> None:
    """create_event should reject a calendar where create_event is disabled."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=_MIXED_PERMS)
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
        calendar_id="jobs@example.com",
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION
    assert "does not allow create event" in result.content


async def test_create_event_auto_selects_allowed_calendar() -> None:
    """With mixed perms, auto-select should pick the calendar that allows creation."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=_MIXED_PERMS)
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    # No calendar_id: should auto-select "primary" (the only one allowing create)
    result = await tool.function(
        title="Job: Auto-Select Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is False
    assert result.content.startswith("ok")


async def test_update_event_blocked_on_disabled() -> None:
    """update_event should reject a calendar where update_event is disabled."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=_MIXED_PERMS)
    tool = _get_tool(tools, ToolName.CALENDAR_UPDATE_EVENT)
    result = await tool.function(
        event_id="evt-002",
        title="Updated",
        calendar_id="jobs@example.com",
    )
    assert result.is_error is True
    assert "does not allow update event" in result.content


async def test_delete_event_blocked_on_disabled() -> None:
    """delete_event should reject a calendar where delete_event is disabled."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(service, enabled_calendars=_MIXED_PERMS)
    tool = _get_tool(tools, ToolName.CALENDAR_DELETE_EVENT)
    result = await tool.function(
        event_id="evt-002",
        calendar_id="jobs@example.com",
    )
    assert result.is_error is True
    assert "does not allow delete event" in result.content


async def test_no_calendars_allow_tool_error() -> None:
    """Write tools should error when all calendars have that tool disabled."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service,
        enabled_calendars=[
            ("primary", "Personal", list(_WRITE_TOOLS), "owner"),
            ("jobs@example.com", "Jobs", list(_WRITE_TOOLS), "writer"),
        ],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is True
    assert "No calendars allow create event" in result.content


async def test_check_availability_works_on_restricted_calendar() -> None:
    """check_availability should work even when write tools are disabled."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service,
        enabled_calendars=[("primary", "Personal", list(_WRITE_TOOLS), "owner")],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-26T00:00:00",
    )
    assert result.is_error is False
    assert "busy slot(s)" in result.content


async def test_list_events_skips_disabled_calendar() -> None:
    """list_events should skip calendars where list_events is disabled."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service,
        enabled_calendars=[
            ("primary", "Personal", [], "owner"),
            ("jobs@example.com", "Jobs", [ToolName.CALENDAR_LIST_EVENTS], "writer"),
        ],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-27T23:59:59",
    )
    assert result.is_error is False
    assert "Smith Kitchen Remodel" in result.content
    # Jobs calendar has list_events disabled, so its events should not appear
    assert "Jones Roof Repair" not in result.content


# ---------------------------------------------------------------------------
# Issue #881: agent should be aware of granular calendar permissions
# ---------------------------------------------------------------------------


async def test_list_calendars_shows_access_role() -> None:
    """calendar_list_calendars should display the Google access role."""
    service = MockGoogleCalendarService()
    service.calendars.append(
        CalendarInfo(id="shared@example.com", summary="Shared", access_role="reader")
    )
    tools = create_calendar_tools(
        service,
        enabled_calendars=[
            ("primary", "Personal", [], "owner"),
            ("shared@example.com", "Shared", [], "reader"),
        ],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_CALENDARS)
    result = await tool.function()
    assert result.is_error is False
    assert "access: owner" in result.content
    assert "access: reader" in result.content


async def test_read_only_calendar_blocks_write_tools() -> None:
    """Write tools should be auto-blocked on a read-only (reader role) calendar."""
    service = MockGoogleCalendarService()
    # Simulate a reader calendar with no explicit disabled_tools -- the system
    # should auto-block writes based on access_role.
    tools = create_calendar_tools(
        service,
        enabled_calendars=[
            ("primary", "Personal", [], "owner"),
            ("readonly@example.com", "Read Only", [], "reader"),
        ],
    )
    # Create event should auto-select the writable calendar
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Test Event",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is False
    assert result.content.startswith("ok")


async def test_read_only_calendar_explicit_write_blocked() -> None:
    """Explicitly targeting a read-only calendar for a write should fail."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service,
        enabled_calendars=[
            ("primary", "Personal", [], "owner"),
            ("readonly@example.com", "Read Only", [], "reader"),
        ],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Test Event",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
        calendar_id="readonly@example.com",
    )
    assert result.is_error is True
    assert "read-only calendar" in result.content


async def test_read_only_calendar_allows_list_events() -> None:
    """Read-only calendars should still allow listing events."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service,
        enabled_calendars=[
            ("readonly@example.com", "Read Only", [], "reader"),
        ],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(
        start_date="2026-03-25T00:00:00",
        end_date="2026-03-27T23:59:59",
    )
    assert result.is_error is False


async def test_free_busy_reader_blocks_writes() -> None:
    """freeBusyReader calendars should also block write tools."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service,
        enabled_calendars=[
            ("freebusy@example.com", "Free Busy", [], "freeBusyReader"),
        ],
    )
    tool = _get_tool(tools, ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Test",
        start="2026-03-28T09:00:00",
        end="2026-03-28T17:00:00",
    )
    assert result.is_error is True
    assert "No calendars allow create event" in result.content


def _make_http_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://www.googleapis.com/calendar/v3/calendars/x/events")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def test_401_is_auth_kind_not_service() -> None:
    """401 should classify as AUTH so the LLM hint tells the user to reconnect,
    not "try a different approach" (SERVICE hint)."""
    result = _handle_http_error(_make_http_error(401), "create event")
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.AUTH
    assert "disconnected" in result.content.lower()


def test_403_surfaces_google_message_not_read_only_guess() -> None:
    """403 with a non-read-only cause (Calendar API disabled in GCP) must
    surface Google's actual message, not the canned "read-only" guess.

    Sibling regression to the Gmail accessNotConfigured bug: the calendar
    factory was the original template the Gmail factory was copied from,
    and carried the same VALIDATION-misclassification + canned-guess shape.
    """
    google_message = (
        "Google Calendar API has not been used in project 1033137896781 before "
        "or it is disabled. Enable it by visiting https://console.developers."
        "google.com/apis/api/calendar-json.googleapis.com/overview?project="
        "1033137896781 then retry."
    )
    body = json.dumps(
        {
            "error": {
                "code": 403,
                "message": google_message,
                "errors": [{"reason": "accessNotConfigured", "message": google_message}],
            }
        }
    )
    result = _handle_http_error(_make_http_error(403, body), "create event")

    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.PERMISSION
    # Google's real message round-trips so the user sees the actual cause.
    assert google_message in result.content
    # The canned "read-only calendar" guess MUST NOT appear for this 403.
    assert "read-only" not in result.content.lower()


def test_403_with_unparseable_body_falls_back() -> None:
    result = _handle_http_error(
        _make_http_error(403, "<html>upstream proxy noise</html>"), "create event"
    )
    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.PERMISSION
    assert "HTTP 403" in result.content
    assert "read-only" not in result.content.lower()


# ---------------------------------------------------------------------------
# Saved calendars the connected account cannot see (reconnect incident)
# ---------------------------------------------------------------------------

_CONNECTED = "connected@example.com"
_STALE_ID = "stale-owner@example.com"


class _UnreachableCalendarService(MockGoogleCalendarService):
    """Mock where saved calendars in *unreachable* 404, like ids from another account."""

    def __init__(self, unreachable: set[str]) -> None:
        super().__init__()
        self.unreachable = unreachable

    def _guard(self, calendar_id: str) -> None:
        if calendar_id in self.unreachable:
            raise _make_http_error(404, '{"error": {"code": 404, "message": "Not Found"}}')

    async def list_events(
        self, calendar_id: str, time_min: datetime, time_max: datetime
    ) -> list[CalendarEventData]:
        self._guard(calendar_id)
        return await super().list_events(calendar_id, time_min, time_max)

    async def create_event(self, calendar_id: str, event: CalendarEventCreate) -> CalendarEventData:
        self._guard(calendar_id)
        return await super().create_event(calendar_id, event)

    async def update_event(
        self, calendar_id: str, event_id: str, updates: CalendarEventUpdate
    ) -> CalendarEventData:
        self._guard(calendar_id)
        if not any(e.id == event_id for e in self.events):
            raise _make_http_error(404)
        return await super().update_event(calendar_id, event_id, updates)

    async def check_availability(
        self, calendar_id: str, time_min: datetime, time_max: datetime
    ) -> list[BusySlot]:
        self._guard(calendar_id)
        return await super().check_availability(calendar_id, time_min, time_max)


def _stale_tools(
    service: MockGoogleCalendarService, *, connected_account: str = _CONNECTED
) -> list[Tool]:
    return create_calendar_tools(
        service,
        enabled_calendars=[(_STALE_ID, "Old Personal", [], "owner")],
        connected_account=connected_account,
    )


async def test_create_event_404_names_connected_account_and_overrides_hint() -> None:
    """The incident: a create against a saved calendar from another account.

    The result must say the calendar is not visible to the connected account,
    name that account, and carry a hint that replaces the generic NOT_FOUND
    "verify the identifier and try again" guidance.
    """
    service = _UnreachableCalendarService({_STALE_ID})
    tool = _get_tool(_stale_tools(service), ToolName.CALENDAR_CREATE_EVENT)
    result = await tool.function(
        title="Job: Test", start="2026-03-25T09:00:00", end="2026-03-25T10:00:00"
    )

    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.NOT_FOUND
    assert "not visible to the Google account" in result.content
    assert _CONNECTED in result.content
    assert "Refresh calendar config" not in result.content
    hint = build_error_hint(result)
    assert "Do not guess" in hint
    assert "reconnect Google Calendar with the account that owns it" in hint
    assert "Verify the identifier" not in hint


async def test_404_with_unknown_account_says_so() -> None:
    """Tokens stored before accounts were recorded: say it is unknown, never blank."""
    service = _UnreachableCalendarService({_STALE_ID})
    tools = _stale_tools(service, connected_account="")
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(start_date="2026-03-25T00:00:00", end_date="2026-03-26T00:00:00")
    assert result.is_error is True
    assert "unrecorded account" in result.content
    assert "()" not in result.content


async def test_update_event_404_on_visible_calendar_is_event_not_found() -> None:
    """A 404 for a missing event must not be blamed on the account."""
    service = _UnreachableCalendarService(set())
    service.calendars.append(CalendarInfo(id=_STALE_ID, summary="Old Personal"))
    tool = _get_tool(_stale_tools(service), ToolName.CALENDAR_UPDATE_EVENT)
    result = await tool.function(event_id="does-not-exist", title="x")
    assert result.is_error is True
    assert "Event does-not-exist was not found" in result.content
    assert "not visible" not in result.content


async def test_update_event_404_on_unreachable_calendar_is_not_visible() -> None:
    service = _UnreachableCalendarService({_STALE_ID})
    tool = _get_tool(_stale_tools(service), ToolName.CALENDAR_UPDATE_EVENT)
    result = await tool.function(event_id="evt-001", title="x")
    assert result.is_error is True
    assert "not visible to the Google account" in result.content
    assert "Do not guess" in build_error_hint(result)


async def test_list_events_skips_unreachable_with_guidance() -> None:
    """Multi-calendar read: the reachable calendars still answer, and the note
    about the skipped one names the connected account instead of 'refresh'."""
    service = _UnreachableCalendarService({_STALE_ID})
    tools = create_calendar_tools(
        service,
        enabled_calendars=[(_STALE_ID, "Old Personal", [], "owner"), *_DEFAULT_ENABLED],
        connected_account=_CONNECTED,
    )
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS)
    result = await tool.function(start_date="2026-03-24T00:00:00", end_date="2026-03-27T00:00:00")
    assert result.is_error is False
    assert "Found 2 event(s)" in result.content
    assert "Old Personal" in result.content
    assert _CONNECTED in result.content
    assert "Do not guess" in result.content


async def test_check_availability_all_unreachable_is_error_not_free() -> None:
    """Nothing could be checked, so it must not report the user as free."""
    service = _UnreachableCalendarService({_STALE_ID, "other-stale@example.com"})
    tools = create_calendar_tools(
        service,
        enabled_calendars=[
            (_STALE_ID, "Old Personal", [], "owner"),
            ("other-stale@example.com", "Old Crew", [], "owner"),
        ],
        connected_account=_CONNECTED,
    )
    tool = _get_tool(tools, ToolName.CALENDAR_CHECK_AVAILABILITY)
    result = await tool.function(start_date="2026-01-01T00:00:00", end_date="2026-01-02T00:00:00")
    assert result.is_error is True
    assert "free" not in result.content.lower()
    assert "Do not guess" in build_error_hint(result)


async def test_list_calendars_flags_saved_calendars_not_visible() -> None:
    """The listing is labelled as the saved selection, names the account, flags
    unreachable calendars, and does not present their stale role as live."""
    service = MockGoogleCalendarService()
    tools = create_calendar_tools(
        service,
        enabled_calendars=[(_STALE_ID, "Old Personal", [], "owner"), *_DEFAULT_ENABLED],
        connected_account=_CONNECTED,
    )
    tool = _get_tool(tools, ToolName.CALENDAR_LIST_CALENDARS)
    result = await tool.function()
    assert result.is_error is False
    assert "saved in Settings" in result.content
    assert f"Connected Google account: {_CONNECTED}" in result.content
    stale_line = next(line for line in result.content.splitlines() if "Old Personal" in line)
    assert "NOT VISIBLE" in stale_line
    assert "access: owner" not in stale_line
    jobs_line = next(line for line in result.content.splitlines() if "Jobs |" in line)
    assert "NOT VISIBLE" not in jobs_line


async def test_list_calendars_says_when_it_could_not_verify() -> None:
    service = MockGoogleCalendarService()

    async def failing(*, show_hidden: bool = False) -> list[CalendarInfo]:
        raise httpx.ConnectError("down")

    service.list_calendars = failing  # type: ignore[method-assign]
    tools = create_calendar_tools(service, enabled_calendars=_DEFAULT_ENABLED)
    result = await _get_tool(tools, ToolName.CALENDAR_LIST_CALENDARS).function()
    assert result.is_error is False
    assert "Could not check these against Google" in result.content


def test_not_visible_message_is_redacted_in_admin_views() -> None:
    """Admin shared-data views run tool results through ``redact_pii``; the
    connected account email in these errors must not survive it."""
    result = _calendar_not_visible_result(_STALE_ID, "Old Personal", "create event", _CONNECTED)
    redacted = redact_pii(result.content + " " + result.hint)
    assert _CONNECTED not in redacted
    assert _STALE_ID not in redacted


# ---------------------------------------------------------------------------
# list_events output: grouping, cap, determinism
# ---------------------------------------------------------------------------


def _event(
    event_id: str,
    start: datetime,
    *,
    hours: int = 2,
    title: str = "Job",
    location: str = "",
    description: str = "",
    all_day: bool = False,
) -> CalendarEventData:
    return CalendarEventData(
        id=event_id,
        title=title,
        start=start,
        end=start + timedelta(hours=hours) if not all_day else start + timedelta(days=1),
        location=location,
        description=description,
        all_day=all_day,
    )


def _week_of_events() -> list[tuple[str, CalendarEventData]]:
    """Two calendars, two events on each of two days, interleaved in time."""
    return [
        ("Personal", _event("p1", datetime(2026, 3, 25, 9, tzinfo=UTC), title="Kitchen A")),
        ("Jobs", _event("j1", datetime(2026, 3, 25, 10, tzinfo=UTC), title="Roof B")),
        ("Personal", _event("p2", datetime(2026, 3, 25, 13, tzinfo=UTC), title="Kitchen C")),
        ("Jobs", _event("j2", datetime(2026, 3, 26, 8, tzinfo=UTC), title="Roof D")),
    ]


def test_event_list_groups_by_calendar_then_day() -> None:
    out = _format_event_list(_week_of_events(), ["Personal", "Jobs"], show_label=True)
    lines = out.splitlines()
    assert lines == [
        "Found 4 event(s):",
        "[Personal]",
        "2026-03-25 Wed",
        "- 09:00-11:00 Kitchen A | id: p1",
        "- 13:00-15:00 Kitchen C | id: p2",
        "[Jobs]",
        "2026-03-25 Wed",
        "- 10:00-12:00 Roof B | id: j1",
        "2026-03-26 Thu",
        "- 08:00-10:00 Roof D | id: j2",
    ]


def test_event_list_single_calendar_has_no_label() -> None:
    events = [pair for pair in _week_of_events() if pair[0] == "Personal"]
    out = _format_event_list(events, ["Personal"], show_label=False)
    assert "[Personal]" not in out
    assert out.count("2026-03-25 Wed") == 1


def test_event_line_keeps_location_notes_and_id() -> None:
    long_notes = "n" * 150
    event = _event(
        "evt-x", datetime(2026, 3, 25, 9, tzinfo=UTC), location="1 Oak St", description=long_notes
    )
    out = _format_event_list([("Personal", event)], ["Personal"], show_label=False)
    assert f"- 09:00-11:00 Job @ 1 Oak St | {'n' * 100}... | id: evt-x" in out


def test_event_line_all_day_and_overnight() -> None:
    all_day = _event("ad", datetime(2026, 3, 25, tzinfo=UTC), title="Holiday", all_day=True)
    overnight = _event("on", datetime(2026, 3, 25, 22, tzinfo=UTC), hours=4, title="Pour")
    out = _format_event_list(
        [("Personal", overnight), ("Personal", all_day)], ["Personal"], show_label=False
    )
    assert "- all day Holiday | id: ad" in out
    assert "- 22:00-2026-03-26 02:00 Pour | id: on" in out
    # All-day sorts first on its day (it starts at midnight).
    assert out.index("id: ad") < out.index("id: on")


def test_all_day_event_stays_on_its_own_day_west_of_utc() -> None:
    """An all-day event starts at UTC midnight, which in Denver is 18:00 the
    evening before. It must not split the previous day's heading."""
    mdt = timezone(timedelta(hours=-6))
    events = [
        ("Jobs", _event("a", datetime(2026, 9, 23, 17, tzinfo=mdt), title="Site visit")),
        ("Jobs", _event("b", datetime(2026, 9, 24, tzinfo=UTC), title="Off", all_day=True)),
        ("Jobs", _event("c", datetime(2026, 9, 23, 19, tzinfo=mdt), title="Estimate")),
        ("Jobs", _event("d", datetime(2026, 9, 24, 8, tzinfo=mdt), title="Roof")),
    ]
    out = _format_event_list(events, ["Jobs"], show_label=False)
    assert out.splitlines() == [
        "Found 4 event(s):",
        "2026-09-23 Wed",
        "- 17:00-19:00 Site visit | id: a",
        "- 19:00-21:00 Estimate | id: c",
        "2026-09-24 Thu",
        "- all day Off | id: b",
        "- 08:00-10:00 Roof | id: d",
    ]
    capped = _format_event_list(events, ["Jobs"], show_label=False, limit=2)
    assert "2026-09-24" not in capped.split("\n(")[0]
    assert "(2 more event(s) from 2026-09-24 on not shown" in capped


def _many_events(n_personal: int, n_jobs: int) -> list[tuple[str, CalendarEventData]]:
    base = datetime(2026, 4, 1, 8, tzinfo=UTC)
    events = [
        ("Personal", _event(f"p{i:03d}", base + timedelta(days=i), title=f"P{i}"))
        for i in range(n_personal)
    ]
    events += [
        ("Jobs", _event(f"j{i:03d}", base + timedelta(days=i, hours=1), title=f"J{i}"))
        for i in range(n_jobs)
    ]
    return events


def test_event_list_caps_and_counts_the_rest_per_calendar() -> None:
    out = _format_event_list(_many_events(50, 30), ["Personal", "Jobs"], show_label=True)
    assert out.startswith("Found 80 event(s), showing the first 60 by start time:")
    # The earliest 60 are shown: 30 of each calendar through day 29.
    assert out.count("| id: ") == 60
    assert "id: p029" in out
    assert "id: j029" in out
    assert "id: p030" not in out
    assert out.rstrip().endswith(
        "(20 more event(s) from 2026-05-01 on not shown: Personal 20. "
        "Narrow the date range or pass calendar_id to see them.)"
    )


def test_event_list_under_the_cap_has_no_footer() -> None:
    out = _format_event_list(_many_events(30, 30), ["Personal", "Jobs"], show_label=True)
    assert out.startswith("Found 60 event(s):")
    assert "not shown" not in out
    assert out.count("| id: ") == 60


def test_event_list_is_deterministic_regardless_of_input_order() -> None:
    """Ties on start time break on calendar order, then title, then id."""
    same_time = datetime(2026, 3, 25, 9, tzinfo=UTC)
    events = [
        ("Jobs", _event("j-b", same_time, title="B")),
        ("Personal", _event("p-z", same_time, title="Z")),
        ("Jobs", _event("j-a", same_time, title="A")),
        ("Personal", _event("p-a", same_time, title="A")),
    ]
    order = ["Personal", "Jobs"]
    first = _format_event_list(events, order, show_label=True, limit=3)
    second = _format_event_list(list(reversed(events)), order, show_label=True, limit=3)
    assert first == second
    # Cap keeps Personal's two and the first Jobs event by title.
    assert "id: p-a" in first
    assert "id: p-z" in first
    assert "id: j-a" in first
    assert "id: j-b" not in first
    assert "Jobs 1" in first


async def test_list_events_tool_caps_a_long_range() -> None:
    service = MockGoogleCalendarService()
    base = datetime(2026, 5, 1, 9, tzinfo=UTC)
    for i in range(75):
        event = _event(f"bulk-{i:03d}", base + timedelta(days=i // 2, hours=i % 2))
        service.events.append(event)
        service._event_calendar_map[event.id] = "primary" if i % 3 else "jobs@example.com"
    tools = create_calendar_tools(service, enabled_calendars=_DEFAULT_ENABLED)
    result = await _get_tool(tools, ToolName.CALENDAR_LIST_EVENTS).function(
        start_date="2026-05-01T00:00:00", end_date="2026-08-01T00:00:00"
    )
    assert result.is_error is False
    assert "Found 75 event(s), showing the first 60" in result.content
    assert result.content.count("| id: bulk-") == 60
    assert "15 more event(s)" in result.content
    assert "Personal 10, Jobs 5" in result.content
    assert "Narrow the date range" in result.content


def test_list_events_description_names_the_cap(cal_tools: list[Tool]) -> None:
    tool = _get_tool(cal_tools, ToolName.CALENDAR_LIST_EVENTS)
    assert "at most 60 events" in tool.description
    assert "narrow the range" in tool.description
