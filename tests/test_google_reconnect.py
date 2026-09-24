"""A dead Google connection reads as AUTH ("reconnect"), not SERVICE.

Runs the real wiring for Google Calendar, Gmail, and Google Drive: the tool
factory (or ``init_storage``) builds the client from a stored token, a
Google 401 routes the refresh through ``oauth_service.refresh_rejected_token``,
and each tool classifies what comes back. The Google APIs and the token
endpoint are fakes; Drive is faked one layer down, at ``httplib2``, so
google-auth's own 401 handling runs.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httplib2
import httpx
import pytest

from backend.app.agent.router import init_storage
from backend.app.agent.tool_errors import build_error_hint
from backend.app.agent.tools.base import Tool, ToolErrorKind, ToolResult
from backend.app.agent.tools.file_tools import create_file_tools
from backend.app.agent.tools.registry import ToolContext
from backend.app.config import settings
from backend.app.integrations.appfolio_vendor.media_resolver import resolve_staged_files
from backend.app.integrations.calendar.factory import _calendar_factory
from backend.app.integrations.calendar.sync import resync_after_connect
from backend.app.integrations.companycam.photos import build_photo_tools
from backend.app.integrations.companycam.service import CompanyCamService
from backend.app.integrations.gmail.factory import _gmail_factory
from backend.app.models import User
from backend.app.services import oauth as oauth_module
from backend.app.services.oauth import (
    OAuthConfig,
    OAuthTokenData,
    ReconnectRequired,
    oauth_service,
)

_TOKEN_URL = "https://oauth.example.invalid/token"

Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)
ToolCall = Callable[[dict[str, Tool]], Awaitable[ToolResult]]


def _token_endpoint(status: int, body: dict[str, Any]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return handler


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


_INVALID_GRANT = _token_endpoint(400, {"error": "invalid_grant"})
_REFRESH_OK = _token_endpoint(
    200, {"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 3600}
)
_TRANSIENT = pytest.mark.parametrize(
    "token_endpoint",
    [_token_endpoint(503, {"error": "temporarily_unavailable"}), _connect_error],
    ids=["token_endpoint_5xx", "network_error"],
)


@pytest.fixture()
def notify() -> Iterator[AsyncMock]:
    with patch.object(oauth_service, "_notify_reauth_needed", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture(autouse=True)
def _google_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "google_calendar_client_id",
        "google_calendar_client_secret",
        "gmail_client_id",
        "gmail_client_secret",
        "google_drive_client_id",
        "google_drive_client_secret",
    ):
        monkeypatch.setattr(settings, name, "configured")


async def _connect(user: User, integration: str) -> None:
    await oauth_service.save_token(
        user.id,
        integration,
        OAuthTokenData(
            access_token="at-old", refresh_token="rt-old", expires_at=time.time() + 3600
        ),
    )


async def _stored(user: User, integration: str) -> OAuthTokenData | None:
    return await oauth_service.load_token_uncached(user.id, integration)


def _wire(monkeypatch: pytest.MonkeyPatch, *, google: Handler, token_endpoint: Handler) -> None:
    """Point the Google REST APIs and the OAuth token endpoint at fakes."""
    real_async_client = httpx.AsyncClient
    google_transport = httpx.MockTransport(google)

    def google_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = google_transport
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    token_client = real_async_client(transport=httpx.MockTransport(token_endpoint))
    monkeypatch.setattr(oauth_service, "_get_http", lambda: token_client)
    monkeypatch.setattr(httpx, "AsyncClient", google_client)
    monkeypatch.setattr(
        "backend.app.services.oauth.get_oauth_config",
        lambda integration: OAuthConfig(
            integration=integration,
            client_id="cid",
            client_secret="csecret",
            authorize_url="https://oauth.example.invalid/authorize",
            token_url=_TOKEN_URL,
            scopes=[],
        ),
    )


def _google_always(status: int, body: dict[str, Any] | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body or {"error": {"code": status}})

    return handler


def _google_rejects_then(status: int, seen: list[str]) -> Handler:
    """401 to the first request, then *status* to the retry. Records the bearer of each."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(401 if len(seen) == 1 else status, json={})

    return handler


def _assert_reconnect(result: ToolResult, product: str) -> None:
    assert result.is_error is True
    assert result.error_kind is ToolErrorKind.AUTH
    assert f"reconnect {product} in Settings" in result.content
    hint = build_error_hint(result)
    assert "temporarily unavailable" not in hint
    # The model is pointed at a link it can offer, not only at Settings.
    assert "manage_integration(action='connect', target='" in hint


# ---------------------------------------------------------------------------
# Calendar and Gmail: one spec each, same scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    integration: str
    product: str
    factory: Callable[[ToolContext], Awaitable[list[Tool]]]
    calls: tuple[ToolCall, ...]
    call_ids: tuple[str, ...]


def _cal_list_events(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["calendar_list_events"].function(
        start_date="2026-03-25T00:00:00", end_date="2026-03-26T00:00:00"
    )


def _cal_create_event(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["calendar_create_event"].function(
        title="Site visit", start="2026-03-25T09:00:00", end="2026-03-25T10:00:00"
    )


def _cal_check_availability(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["calendar_check_availability"].function(
        start_date="2026-03-25T00:00:00", end_date="2026-03-26T00:00:00"
    )


def _cal_list_calendars(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["calendar_list_calendars"].function()


def _gmail_search(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["gmail_search"].function(query="from:jane.doe@example.com")


def _gmail_get_message(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["gmail_get_message"].function(message_id="m1")


def _gmail_send(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["gmail_send"].function(to=["jane.doe@example.com"], subject="Quote", body="Hi")


CALENDAR = _Spec(
    "google_calendar",
    "Google Calendar",
    _calendar_factory,
    (_cal_list_events, _cal_create_event, _cal_check_availability, _cal_list_calendars),
    ("list_events", "create_event", "check_availability", "list_calendars"),
)
GMAIL = _Spec(
    "gmail",
    "Gmail",
    _gmail_factory,
    (_gmail_search, _gmail_get_message, _gmail_send),
    ("search", "get_message", "send"),
)

_CASES = [(spec, call) for spec in (CALENDAR, GMAIL) for call in spec.calls]
_CASE_IDS = [f"{spec.integration}-{name}" for spec in (CALENDAR, GMAIL) for name in spec.call_ids]
EVERY_TOOL = pytest.mark.parametrize(
    ("spec", "call"),
    _CASES,
    ids=_CASE_IDS,
)
EVERY_INTEGRATION = pytest.mark.parametrize("spec", [CALENDAR, GMAIL], ids=["calendar", "gmail"])


async def _tools(user: User, spec: _Spec) -> dict[str, Tool]:
    tools = await spec.factory(ToolContext(user=user))
    assert tools, "factory returned no tools for a connected user"
    return {t.name: t for t in tools}


@EVERY_TOOL
async def test_invalid_grant_on_refresh_is_auth_and_retires_token(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    spec: _Spec,
    call: ToolCall,
) -> None:
    """Regression: an expired or revoked refresh token read as an outage."""
    await _connect(test_user, spec.integration)
    _wire(monkeypatch, google=_google_always(401), token_endpoint=_INVALID_GRANT)
    tools = await _tools(test_user, spec)

    result = await call(tools)

    _assert_reconnect(result, spec.product)
    assert await _stored(test_user, spec.integration) is None
    notify.assert_awaited_once_with(test_user.id, spec.integration)


@EVERY_INTEGRATION
async def test_later_calls_in_the_turn_stay_auth_after_the_token_is_retired(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, spec: _Spec
) -> None:
    await _connect(test_user, spec.integration)
    _wire(monkeypatch, google=_google_always(401), token_endpoint=_INVALID_GRANT)
    tools = await _tools(test_user, spec)

    for call in spec.calls:
        _assert_reconnect(await call(tools), spec.product)
    notify.assert_awaited_once()


@EVERY_TOOL
async def test_401_persisting_after_refresh_is_auth(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    spec: _Spec,
    call: ToolCall,
) -> None:
    await _connect(test_user, spec.integration)
    seen: list[str] = []
    _wire(monkeypatch, google=_google_rejects_then(401, seen), token_endpoint=_REFRESH_OK)
    tools = await _tools(test_user, spec)

    result = await call(tools)

    _assert_reconnect(result, spec.product)
    assert seen[:2] == ["Bearer at-old", "Bearer at-new"]
    # The refresh itself worked, so the rotated token is kept.
    stored = await _stored(test_user, spec.integration)
    assert stored is not None
    assert stored.refresh_token == "rt-new"
    notify.assert_not_awaited()


@EVERY_TOOL
@_TRANSIENT
async def test_transient_refresh_failure_stays_service(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    spec: _Spec,
    call: ToolCall,
    token_endpoint: Handler,
) -> None:
    await _connect(test_user, spec.integration)
    _wire(monkeypatch, google=_google_always(401), token_endpoint=token_endpoint)
    tools = await _tools(test_user, spec)

    result = await call(tools)

    if call is _cal_list_calendars:
        # Listing the saved selection checks Google best effort: an outage
        # there is a caveat on the listing, not a failed call.
        assert result.is_error is False
        assert "Could not check these against Google" in result.content
    else:
        assert result.error_kind is ToolErrorKind.SERVICE
    stored = await _stored(test_user, spec.integration)
    assert stored is not None
    assert stored.refresh_token == "rt-old"
    notify.assert_not_awaited()


@EVERY_INTEGRATION
@pytest.mark.parametrize(
    ("status", "kind"),
    [(403, ToolErrorKind.PERMISSION), (429, ToolErrorKind.SERVICE)],
    ids=["403", "429"],
)
async def test_403_and_429_are_not_refreshed_or_reclassified(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    spec: _Spec,
    status: int,
    kind: ToolErrorKind,
) -> None:
    await _connect(test_user, spec.integration)
    refreshes: list[httpx.Request] = []

    def token_endpoint(request: httpx.Request) -> httpx.Response:
        refreshes.append(request)
        return httpx.Response(500)

    body = {"error": {"code": status, "message": "Daily limit for this project exceeded."}}
    _wire(monkeypatch, google=_google_always(status, body), token_endpoint=token_endpoint)
    tools = await _tools(test_user, spec)

    result = await spec.calls[0](tools)

    assert result.error_kind is kind
    if status == 403:
        # Google's own reason, verbatim, rather than a canned guess.
        assert "Daily limit for this project exceeded." in result.content
    assert refreshes == []
    notify.assert_not_awaited()


@EVERY_INTEGRATION
async def test_a_401_arriving_after_a_sibling_refreshed_does_not_refresh_again(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, spec: _Spec
) -> None:
    """Regression: the service named its current token, not the one Google refused,
    so a 401 that came back after a sibling's refresh posted the new refresh
    token again and rotated the tokens for nothing."""
    await _connect(test_user, spec.integration)
    first_retry_done = asyncio.Event()
    old_token_calls = 0
    posts: list[httpx.Request] = []

    async def google(request: httpx.Request) -> httpx.Response:
        nonlocal old_token_calls
        if request.headers["Authorization"] == "Bearer at-new":
            first_retry_done.set()
            return httpx.Response(200, json={"items": [], "messages": []})
        old_token_calls += 1
        if old_token_calls == 2:
            # This call's 401 comes back only after the sibling refreshed.
            await asyncio.wait_for(first_retry_done.wait(), timeout=5)
        return httpx.Response(401, json={})

    def token_endpoint(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(
            200, json={"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 3600}
        )

    _wire(monkeypatch, google=google, token_endpoint=token_endpoint)
    tools = await _tools(test_user, spec)

    results = await asyncio.gather(spec.calls[0](tools), spec.calls[0](tools))

    assert [r.is_error for r in results] == [False, False], [r.content for r in results]
    assert old_token_calls == 2
    assert len(posts) == 1
    stored = await _stored(test_user, spec.integration)
    assert stored is not None
    assert (stored.access_token, stored.refresh_token) == ("at-new", "rt-new")
    notify.assert_not_awaited()


async def test_concurrent_calls_on_a_dead_grant_post_once_and_notify_once(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """Two callers with their own refresh hooks hit the same dead grant at once.

    Separate hooks, as two workers have, so the second caller queues on the
    advisory lock (calls sharing one hook await its in-flight refresh
    instead). Regression: the token was deleted only after the refresh lock
    was released, so the peer waiting on the lock reloaded the still-present
    token, posted the dead refresh token again, and notified a second time.
    """
    await _connect(test_user, "google_calendar")
    posts: list[httpx.Request] = []
    peer_waiting = asyncio.Event()

    async def token_endpoint(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        if len(posts) == 1:
            # Hold the first refresh inside the lock until the peer queues on it.
            await asyncio.wait_for(peer_waiting.wait(), timeout=5)
        return httpx.Response(400, json={"error": "invalid_grant"})

    _wire(monkeypatch, google=_google_always(401), token_endpoint=token_endpoint)

    real_acquire = oauth_module._try_acquire_advisory_lock_async
    acquires = 0

    async def acquire(conn: Any, lock_key: str) -> bool:
        nonlocal acquires
        acquires += 1
        if acquires == 2:
            peer_waiting.set()
        return await real_acquire(conn, lock_key)

    monkeypatch.setattr(oauth_module, "_try_acquire_advisory_lock_async", acquire)

    # Delay the post-lock retirement well past the lock poll interval, so a
    # token left in place after the lock is released would be reloaded.
    real_handle = oauth_service.handle_permanent_refresh_failure

    async def slow_handle(user_id: str, integration: str, error: Exception) -> bool:
        await asyncio.sleep(oauth_module._LOCK_RETRY_INTERVAL_S * 5)
        return await real_handle(user_id, integration, error)

    monkeypatch.setattr(oauth_service, "handle_permanent_refresh_failure", slow_handle)

    first = oauth_service.build_rejected_token_refresher(test_user.id, "google_calendar")
    second = oauth_service.build_rejected_token_refresher(test_user.id, "google_calendar")
    results = await asyncio.gather(first("at-old"), second("at-old"), return_exceptions=True)

    assert all(isinstance(r, ReconnectRequired) for r in results), results
    assert len(posts) == 1
    assert await _stored(test_user, "google_calendar") is None
    notify.assert_awaited_once_with(test_user.id, "google_calendar")


# ---------------------------------------------------------------------------
# Background and non-agent callers
# ---------------------------------------------------------------------------


async def test_post_connect_resync_with_a_dead_grant_retires_quietly(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """The resync hook runs outside a turn: it must not raise, and must notify once."""
    await _connect(test_user, "google_calendar")
    _wire(monkeypatch, google=_google_always(401), token_endpoint=_INVALID_GRANT)
    token = await _stored(test_user, "google_calendar")
    assert token is not None

    await resync_after_connect(test_user.id, token)

    assert await _stored(test_user, "google_calendar") is None
    notify.assert_awaited_once_with(test_user.id, "google_calendar")


@EVERY_INTEGRATION
async def test_next_background_turn_after_retirement_offers_no_tools(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, spec: _Spec
) -> None:
    """A heartbeat turn after the grant died builds no tools and sends nothing more.

    Heartbeats run the same factories as a user turn. Once a dead grant has
    retired the token, the factory returns nothing, so later ticks neither
    call Google nor repeat the reconnect notice.
    """
    await _connect(test_user, spec.integration)
    google_calls: list[httpx.Request] = []

    def google(request: httpx.Request) -> httpx.Response:
        google_calls.append(request)
        return httpx.Response(401, json={})

    _wire(monkeypatch, google=google, token_endpoint=_INVALID_GRANT)
    tools = await _tools(test_user, spec)
    _assert_reconnect(await spec.calls[0](tools), spec.product)
    calls_before = len(google_calls)

    assert await spec.factory(ToolContext(user=test_user)) == []
    assert len(google_calls) == calls_before
    notify.assert_awaited_once()


# ---------------------------------------------------------------------------
# Google Drive (google-api-python-client over httplib2)
# ---------------------------------------------------------------------------


def _drive_http(monkeypatch: pytest.MonkeyPatch, statuses: list[int]) -> list[str]:
    """Answer Drive requests with *statuses* in order (the last one repeats).

    Returns the list the bearer of each request is appended to.
    """
    seen: list[str] = []

    def request(
        self: httplib2.Http, uri: str, method: str = "GET", **kwargs: Any
    ) -> tuple[httplib2.Response, bytes]:
        headers = kwargs.get("headers") or {}
        seen.append(headers.get("authorization") or headers.get("Authorization", ""))
        status = statuses[min(len(seen) - 1, len(statuses) - 1)]
        body = {"files": []} if status == 200 else {"error": {"code": status}}
        return httplib2.Response({"status": status}), json.dumps(body).encode()

    monkeypatch.setattr(httplib2.Http, "request", request)
    return seen


async def _drive_tools(user: User) -> dict[str, Tool]:
    storage = await init_storage(user)
    assert storage is not None
    return {t.name: t for t in create_file_tools(user, storage)}


def _drive_search(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["find_saved_files"].function(query="invoice")


def _drive_read(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["read_from_storage"].function(file_path="/Inbox/notes.txt")


DRIVE_TOOLS = pytest.mark.parametrize(
    "call", [_drive_search, _drive_read], ids=["find_saved_files", "read_from_storage"]
)


@DRIVE_TOOLS
async def test_drive_invalid_grant_is_auth_and_retires_token(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, call: ToolCall
) -> None:
    """Regression: google-auth refreshed on its own, unlocked, and a dead grant
    surfaced as a tool crash while the stale token stayed connected."""
    await _connect(test_user, "google_drive")
    _wire(monkeypatch, google=_google_always(401), token_endpoint=_INVALID_GRANT)
    seen = _drive_http(monkeypatch, [401])
    tools = await _drive_tools(test_user)

    result = await call(tools)

    _assert_reconnect(result, "Google Drive")
    assert seen == ["Bearer at-old"]
    assert await _stored(test_user, "google_drive") is None
    notify.assert_awaited_once_with(test_user.id, "google_drive")


async def test_drive_refreshes_once_and_retries_with_the_new_token(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    await _connect(test_user, "google_drive")
    _wire(monkeypatch, google=_google_always(401), token_endpoint=_REFRESH_OK)
    seen = _drive_http(monkeypatch, [401, 200])
    tools = await _drive_tools(test_user)

    result = await _drive_search(tools)

    assert result.error_kind is not ToolErrorKind.AUTH
    assert seen[:2] == ["Bearer at-old", "Bearer at-new"]
    assert all(bearer == "Bearer at-new" for bearer in seen[1:])
    stored = await _stored(test_user, "google_drive")
    assert stored is not None
    assert (stored.access_token, stored.refresh_token) == ("at-new", "rt-new")
    notify.assert_not_awaited()


@DRIVE_TOOLS
async def test_drive_401_persisting_after_refresh_is_auth(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, call: ToolCall
) -> None:
    await _connect(test_user, "google_drive")
    _wire(monkeypatch, google=_google_always(401), token_endpoint=_REFRESH_OK)
    seen = _drive_http(monkeypatch, [401])
    tools = await _drive_tools(test_user)

    result = await call(tools)

    _assert_reconnect(result, "Google Drive")
    assert seen == ["Bearer at-old", "Bearer at-new"]
    stored = await _stored(test_user, "google_drive")
    assert stored is not None
    assert stored.refresh_token == "rt-new"
    # The token is still stored, so ``connect`` alone would refuse.
    assert "manage_integration(action='disconnect', target='google_drive')" in (
        build_error_hint(result)
    )
    notify.assert_not_awaited()


@_TRANSIENT
async def test_drive_transient_refresh_failure_stays_service(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, token_endpoint: Handler
) -> None:
    await _connect(test_user, "google_drive")
    _wire(monkeypatch, google=_google_always(401), token_endpoint=token_endpoint)
    _drive_http(monkeypatch, [401])
    tools = await _drive_tools(test_user)

    result = await _drive_read(tools)

    assert result.error_kind is ToolErrorKind.SERVICE
    stored = await _stored(test_user, "google_drive")
    assert stored is not None
    assert stored.refresh_token == "rt-old"
    notify.assert_not_awaited()


async def test_gmail_attachment_from_a_dead_drive_names_drive(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """gmail_send reads attachments from Drive; a dead Drive grant must not blame Gmail."""
    await _connect(test_user, "gmail")
    await _connect(test_user, "google_drive")
    _wire(monkeypatch, google=_google_always(500), token_endpoint=_INVALID_GRANT)
    _drive_http(monkeypatch, [401])
    storage = await init_storage(test_user)
    tools = {t.name: t for t in await _gmail_factory(ToolContext(user=test_user, storage=storage))}

    result = await tools["gmail_send"].function(
        to=["jane.doe@example.com"], subject="Quote", body="Hi", attachments=["/Inbox/quote.pdf"]
    )

    _assert_reconnect(result, "Google Drive")
    assert await _stored(test_user, "google_drive") is None
    assert await _stored(test_user, "gmail") is not None
    notify.assert_awaited_once_with(test_user.id, "google_drive")


async def test_drive_read_by_companycam_and_appfolio_names_drive(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """CompanyCam and AppFolio read saved photos from Drive; a dead Drive grant is AUTH.

    Regression: both let ``ReconnectRequired`` escape, so the tool runner
    reported INTERNAL with a traceback instead of asking for a Drive reconnect.
    """
    await _connect(test_user, "google_drive")
    _wire(monkeypatch, google=_google_always(500), token_endpoint=_INVALID_GRANT)
    _drive_http(monkeypatch, [401])
    storage = await init_storage(test_user)
    ctx = ToolContext(user=test_user, storage=storage)

    companycam = MagicMock(spec=CompanyCamService)
    upload = next(
        t for t in build_photo_tools(companycam, ctx) if t.name == "companycam_upload_photo"
    )
    companycam_result = await upload.function(
        project_id="p1", original_url="/Inbox/photos/roof.jpg"
    )
    appfolio_result = await resolve_staged_files(ctx, ["/Inbox/photos/roof.jpg"])

    _assert_reconnect(companycam_result, "Google Drive")
    assert isinstance(appfolio_result, ToolResult)
    _assert_reconnect(appfolio_result, "Google Drive")
    companycam.upload_photo.assert_not_awaited()
    assert await _stored(test_user, "google_drive") is None
    notify.assert_awaited_once_with(test_user.id, "google_drive")


# ---------------------------------------------------------------------------
# Refresh that could not run, and a notice that must not be lost
# ---------------------------------------------------------------------------


@EVERY_TOOL
async def test_contended_refresh_lock_is_service_not_a_dead_connection(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    spec: _Spec,
    call: ToolCall,
) -> None:
    """Regression: the 401 left standing by a contended lock read as "expired or revoked".

    QuickBooks and Drive already treat this as SERVICE; Calendar and Gmail
    told the model not to retry.
    """
    await _connect(test_user, spec.integration)
    posts: list[httpx.Request] = []

    def token_endpoint(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(400, json={"error": "invalid_grant"})

    _wire(monkeypatch, google=_google_always(401), token_endpoint=token_endpoint)

    async def contended(conn: Any, lock_key: str) -> bool:
        return False

    monkeypatch.setattr(oauth_module, "_try_acquire_advisory_lock_async", contended)
    tools = await _tools(test_user, spec)

    result = await call(tools)

    if call is _cal_list_calendars:
        # Checking the saved selection against Google is best effort there.
        assert result.is_error is False
        assert "Could not check these against Google" in result.content
    else:
        assert result.error_kind is ToolErrorKind.SERVICE
        assert "expired or revoked" not in build_error_hint(result)
    assert posts == []
    stored = await _stored(test_user, spec.integration)
    assert stored is not None
    assert stored.refresh_token == "rt-old"
    notify.assert_not_awaited()


@EVERY_INTEGRATION
async def test_401_with_nothing_to_refresh_with_stays_auth(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, spec: _Spec
) -> None:
    """Only a contended lock is SERVICE. A 401 on a token with no refresh token
    has nothing to retry with, so it keeps its reconnect classification."""
    await oauth_service.save_token(
        test_user.id,
        spec.integration,
        OAuthTokenData(access_token="at-old", refresh_token="", expires_at=time.time() + 3600),
    )
    _wire(monkeypatch, google=_google_always(401), token_endpoint=_REFRESH_OK)
    tools = await _tools(test_user, spec)

    result = await spec.calls[0](tools)

    assert result.error_kind is ToolErrorKind.AUTH
    assert "Retry this call shortly" not in build_error_hint(result)


async def test_user_is_notified_even_when_the_second_delete_fails(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """Regression: the token is deleted under the refresh lock, so nothing refreshes it
    again. If ``handle_permanent_refresh_failure``'s own delete then raised, the
    notice after it never ran and the user was never told."""
    await _connect(test_user, "google_calendar")
    _wire(monkeypatch, google=_google_always(401), token_endpoint=_INVALID_GRANT)
    real_delete = oauth_service.delete_token
    deletes = 0

    async def delete_then_fail(user_id: str, integration: str) -> bool:
        nonlocal deletes
        deletes += 1
        if deletes == 1:
            return await real_delete(user_id, integration)
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(oauth_service, "delete_token", delete_then_fail)
    refresh = oauth_service.build_rejected_token_refresher(test_user.id, "google_calendar")

    with pytest.raises(ReconnectRequired):
        await refresh("at-old")

    assert deletes == 2
    assert await _stored(test_user, "google_calendar") is None
    notify.assert_awaited_once_with(test_user.id, "google_calendar")
