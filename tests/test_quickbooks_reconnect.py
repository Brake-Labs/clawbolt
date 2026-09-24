"""A dead QuickBooks connection reads as AUTH ("reconnect"), not SERVICE.

Runs the real wiring: ``_get_quickbooks_service_for_user`` builds the service
from a stored token, a QBO 401 routes the refresh through
``oauth_service.refresh_rejected_token``, and each tool classifies what comes
back. QBO and the Intuit token endpoint are ``httpx.MockTransport`` fakes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.agent.tool_errors import build_error_hint
from backend.app.agent.tools.base import Tool, ToolErrorKind, ToolResult
from backend.app.integrations.quickbooks import service as service_module
from backend.app.integrations.quickbooks.factory import (
    _get_quickbooks_service_for_user,
    create_quickbooks_tools,
)
from backend.app.models import User
from backend.app.services import oauth as oauth_module
from backend.app.services.oauth import OAuthConfig, OAuthTokenData, oauth_service

_TOKEN_URL = "https://oauth.example.invalid/oauth2/v1/tokens/bearer"

TokenHandler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)
QboHandler = TokenHandler


def _call_query(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["qb_query"].function(query="SELECT * FROM Customer")


def _call_create(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["qb_create"].function(entity_type="Customer", data={"DisplayName": "Acme"})


def _call_update(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["qb_update"].function(
        entity_type="Customer",
        data={"Id": "1", "SyncToken": "0", "DisplayName": "Acme Plumbing"},
        delete_line_ids=[],
    )


def _call_send(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["qb_send"].function(
        entity_type="Invoice", entity_id="42", email="jane.doe@example.com"
    )


ALL_TOOLS = pytest.mark.parametrize(
    "call",
    [_call_query, _call_create, _call_update, _call_send],
    ids=["qb_query", "qb_create", "qb_update", "qb_send"],
)


def _token_endpoint(status: int, body: dict[str, Any]) -> TokenHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return handler


def _qbo_always(status: int, body: dict[str, Any] | None = None) -> QboHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body or {"fault": "rejected"})

    return handler


@pytest.fixture()
def notify() -> Iterator[AsyncMock]:
    with patch.object(oauth_service, "_notify_reauth_needed", new_callable=AsyncMock) as mock:
        yield mock


async def _connect(user: User) -> None:
    await oauth_service.save_token(
        user.id,
        "quickbooks",
        OAuthTokenData(
            access_token="at-old",
            refresh_token="rt-old",
            expires_at=time.time() + 3600,
            realm_id="9999",
        ),
    )


async def _tools(
    user: User,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qbo: QboHandler,
    token_endpoint: TokenHandler,
) -> dict[str, Tool]:
    """Wire QBO and the token endpoint to fakes, then build the real tools."""
    real_async_client = httpx.AsyncClient
    qbo_transport = httpx.MockTransport(qbo)

    def qbo_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = qbo_transport
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service_module.httpx, "AsyncClient", qbo_client)
    token_client = real_async_client(transport=httpx.MockTransport(token_endpoint))
    monkeypatch.setattr(oauth_service, "_get_http", lambda: token_client)
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

    svc = await _get_quickbooks_service_for_user(user.id)
    assert svc is not None
    return {t.name: t for t in create_quickbooks_tools(svc)}


def _assert_reconnect(result: ToolResult) -> None:
    assert result.is_error is True
    assert result.error_kind is ToolErrorKind.AUTH
    assert "reconnect" in result.content.lower()
    assert "temporarily unavailable" not in build_error_hint(result)


@ALL_TOOLS
async def test_invalid_grant_on_refresh_is_auth_and_retires_token(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    call: Callable[[dict[str, Tool]], Awaitable[ToolResult]],
) -> None:
    """Regression: an expired or revoked refresh token read as an outage."""
    await _connect(test_user)
    tools = await _tools(
        test_user,
        monkeypatch,
        qbo=_qbo_always(401),
        token_endpoint=_token_endpoint(400, {"error": "invalid_grant"}),
    )

    result = await call(tools)

    _assert_reconnect(result)
    # The shared dead-grant path ran: token gone (status reads disconnected)
    # and the user was told, as for every other OAuth integration.
    assert await oauth_service.load_token_uncached(test_user.id, "quickbooks") is None
    notify.assert_awaited_once_with(test_user.id, "quickbooks")


async def test_later_calls_in_the_turn_stay_auth_after_the_token_is_retired(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    await _connect(test_user)
    tools = await _tools(
        test_user,
        monkeypatch,
        qbo=_qbo_always(401),
        token_endpoint=_token_endpoint(400, {"error": "invalid_grant"}),
    )

    _assert_reconnect(await _call_query(tools))
    _assert_reconnect(await _call_send(tools))
    notify.assert_awaited_once()


@ALL_TOOLS
@pytest.mark.parametrize("status", [401, 403])
async def test_rejection_persisting_after_refresh_is_auth(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    call: Callable[[dict[str, Tool]], Awaitable[ToolResult]],
    status: int,
) -> None:
    await _connect(test_user)
    seen_tokens: list[str] = []

    def qbo(request: httpx.Request) -> httpx.Response:
        seen_tokens.append(request.headers["Authorization"])
        return httpx.Response(401 if len(seen_tokens) == 1 else status, json={})

    tools = await _tools(
        test_user,
        monkeypatch,
        qbo=qbo,
        token_endpoint=_token_endpoint(
            200, {"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 3600}
        ),
    )

    result = await call(tools)

    _assert_reconnect(result)
    assert seen_tokens == ["Bearer at-old", "Bearer at-new"]
    # The refresh itself worked, so the rotated token is kept.
    stored = await oauth_service.load_token_uncached(test_user.id, "quickbooks")
    assert stored is not None
    assert stored.refresh_token == "rt-new"
    notify.assert_not_awaited()


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


@ALL_TOOLS
@pytest.mark.parametrize(
    "token_endpoint",
    [_token_endpoint(503, {"error": "temporarily_unavailable"}), _connect_error],
    ids=["token_endpoint_5xx", "network_error"],
)
async def test_transient_refresh_failure_stays_service(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    call: Callable[[dict[str, Tool]], Awaitable[ToolResult]],
    token_endpoint: TokenHandler,
) -> None:
    await _connect(test_user)
    tools = await _tools(
        test_user, monkeypatch, qbo=_qbo_always(401), token_endpoint=token_endpoint
    )

    result = await call(tools)

    assert result.error_kind is ToolErrorKind.SERVICE
    assert await oauth_service.load_token_uncached(test_user.id, "quickbooks") is not None
    notify.assert_not_awaited()


@ALL_TOOLS
async def test_throttle_stays_service(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    call: Callable[[dict[str, Tool]], Awaitable[ToolResult]],
) -> None:
    await _connect(test_user)
    refreshes: list[httpx.Request] = []

    def token_endpoint(request: httpx.Request) -> httpx.Response:
        refreshes.append(request)
        return httpx.Response(500)

    fault = {"Fault": {"Error": [{"Message": "ThrottleExceeded", "code": "3001"}]}}
    tools = await _tools(
        test_user, monkeypatch, qbo=_qbo_always(429, fault), token_endpoint=token_endpoint
    )

    result = await call(tools)

    assert result.error_kind is ToolErrorKind.SERVICE
    assert refreshes == []
    notify.assert_not_awaited()


@ALL_TOOLS
async def test_contended_refresh_lock_is_a_retryable_service_error(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    caplog: pytest.LogCaptureFixture,
    call: Callable[[dict[str, Tool]], Awaitable[ToolResult]],
) -> None:
    """Regression: a contended refresh lock escaped the service, so the model read
    the internal "OAuth refresh lock ... held by a peer" text and the failure was
    logged with a traceback."""
    await _connect(test_user)
    posts: list[httpx.Request] = []

    def token_endpoint(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(400, json={"error": "invalid_grant"})

    tools = await _tools(
        test_user, monkeypatch, qbo=_qbo_always(401), token_endpoint=token_endpoint
    )

    async def contended(conn: Any, lock_key: str) -> bool:
        return False

    monkeypatch.setattr(oauth_module, "_try_acquire_advisory_lock_async", contended)

    with caplog.at_level(logging.WARNING):
        result = await call(tools)

    assert result.is_error is True
    assert result.error_kind is ToolErrorKind.SERVICE
    assert "Retry this call shortly" in result.hint
    assert "lock" not in result.content.lower()
    assert not [r for r in caplog.records if r.exc_info], "logged a traceback"
    assert posts == []
    assert await oauth_service.load_token_uncached(test_user.id, "quickbooks") is not None
    notify.assert_not_awaited()


async def test_parallel_401s_with_a_slow_refresh_post_once_and_all_succeed(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """Regression: a sibling call in the same turn waited on the DB lock, gave up
    after ``_LOCK_MAX_WAIT_S`` while its own sibling was still refreshing, and
    reported "retry shortly"."""
    await _connect(test_user)
    monkeypatch.setattr(oauth_module, "_LOCK_MAX_WAIT_S", 0.2)
    rejected = 0
    all_rejected = asyncio.Event()
    posts: list[httpx.Request] = []

    async def qbo(request: httpx.Request) -> httpx.Response:
        nonlocal rejected
        if request.headers["Authorization"] == "Bearer at-new":
            return httpx.Response(200, json={"QueryResponse": {"Customer": []}})
        rejected += 1
        if rejected == 3:
            all_rejected.set()
        return httpx.Response(401, json={})

    async def token_endpoint(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        await asyncio.wait_for(all_rejected.wait(), timeout=5)
        # Outlast the lock wait, as a slow token endpoint does.
        await asyncio.sleep(0.6)
        return httpx.Response(
            200, json={"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 3600}
        )

    tools = await _tools(test_user, monkeypatch, qbo=qbo, token_endpoint=token_endpoint)

    results = await asyncio.gather(*(_call_query(tools) for _ in range(3)))

    assert [r.is_error for r in results] == [False, False, False], [r.content for r in results]
    assert len(posts) == 1
    stored = await oauth_service.load_token_uncached(test_user.id, "quickbooks")
    assert stored is not None
    assert stored.refresh_token == "rt-new"
    notify.assert_not_awaited()


async def test_a_401_arriving_after_a_sibling_refreshed_does_not_refresh_again(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """Regression: the service named its current token, not the one QBO refused,
    so a 401 that came back after a sibling's refresh posted the new refresh
    token again and rotated the tokens for nothing."""
    await _connect(test_user)
    first_retry_done = asyncio.Event()
    old_token_calls = 0
    posts: list[httpx.Request] = []

    async def qbo(request: httpx.Request) -> httpx.Response:
        nonlocal old_token_calls
        if request.headers["Authorization"] == "Bearer at-new":
            first_retry_done.set()
            return httpx.Response(200, json={"QueryResponse": {"Customer": []}})
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

    tools = await _tools(test_user, monkeypatch, qbo=qbo, token_endpoint=token_endpoint)

    results = await asyncio.gather(_call_query(tools), _call_query(tools))

    assert [r.is_error for r in results] == [False, False], [r.content for r in results]
    assert len(posts) == 1


async def test_disconnect_during_the_refresh_post_stays_disconnected(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """Regression: the user disconnected while the refresh POST was in flight, and
    the refresh then saved its result, upserting the credential back."""
    await _connect(test_user)

    async def token_endpoint(request: httpx.Request) -> httpx.Response:
        # The user disconnects QuickBooks while the token endpoint answers.
        await oauth_service.delete_token(test_user.id, "quickbooks")
        return httpx.Response(
            200, json={"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 3600}
        )

    tools = await _tools(
        test_user, monkeypatch, qbo=_qbo_always(401), token_endpoint=token_endpoint
    )

    result = await _call_query(tools)

    _assert_reconnect(result)
    assert await oauth_service.load_token_uncached(test_user.id, "quickbooks") is None
    # The user chose to disconnect, so there is nothing to tell them.
    notify.assert_not_awaited()


async def _reconnect(user: User, realm_id: str) -> None:
    """The user disconnects QuickBooks and connects again, as the OAuth callback does."""
    await oauth_service.delete_token(user.id, "quickbooks")
    await oauth_service.save_token(
        user.id,
        "quickbooks",
        OAuthTokenData(
            access_token="at-new-grant",
            refresh_token="rt-new-grant",
            expires_at=time.time() + 3600,
            realm_id=realm_id,
        ),
    )


@pytest.mark.parametrize(
    ("new_realm", "succeeds"),
    [("9999", True), ("8888", False)],
    ids=["same_company", "different_company"],
)
async def test_reconnect_during_the_refresh_post_keeps_the_new_connection(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    new_realm: str,
    succeeds: bool,
) -> None:
    """Regression: the user disconnected and reconnected while a refresh POST for the
    old grant was in flight, and the refresh then saved the old grant's tokens and
    old ``realm_id`` over the new connection."""
    await _connect(test_user)
    sent: list[tuple[str, str]] = []

    def qbo(request: httpx.Request) -> httpx.Response:
        # An Intuit token reaches one company: the new grant works only on its own.
        auth = request.headers["Authorization"]
        sent.append((auth, request.url.path))
        if auth == "Bearer at-new-grant" and f"/company/{new_realm}/" in request.url.path:
            return httpx.Response(200, json={"QueryResponse": {"Customer": []}})
        return httpx.Response(401, json={})

    async def token_endpoint(request: httpx.Request) -> httpx.Response:
        await _reconnect(test_user, new_realm)
        return httpx.Response(
            200,
            json={
                "access_token": "at-old-grant",
                "refresh_token": "rt-old-grant",
                "expires_in": 3600,
            },
        )

    tools = await _tools(test_user, monkeypatch, qbo=qbo, token_endpoint=token_endpoint)

    result = await _call_query(tools)

    stored = await oauth_service.load_token_uncached(test_user.id, "quickbooks")
    assert stored is not None
    assert (stored.access_token, stored.refresh_token, stored.realm_id) == (
        "at-new-grant",
        "rt-new-grant",
        new_realm,
    )
    # The call retries once with the new connection's token, still against the
    # company this turn was built for. The old grant's token is never sent again.
    assert [auth for auth, _ in sent] == ["Bearer at-old", "Bearer at-new-grant"]
    assert all("/company/9999/" in path for _, path in sent)
    if succeeds:
        assert result.is_error is False, result.content
    else:
        # A token for another company cannot reach this one, so nothing is
        # written there. The next turn is built for the new company.
        _assert_reconnect(result)
    notify.assert_not_awaited()
