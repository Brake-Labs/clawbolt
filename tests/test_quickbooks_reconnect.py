"""A dead QuickBooks connection reads as AUTH ("reconnect"), not SERVICE.

Runs the real wiring: ``_get_quickbooks_service_for_user`` builds the service
from a stored token, a QBO 401 routes the refresh through
``oauth_service.refresh_rejected_token``, and each tool classifies what comes
back. QBO and the Intuit token endpoint are ``httpx.MockTransport`` fakes.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterator
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
from backend.app.services.oauth import OAuthConfig, OAuthTokenData, oauth_service

_TOKEN_URL = "https://oauth.example.invalid/oauth2/v1/tokens/bearer"

TokenHandler = Callable[[httpx.Request], httpx.Response]
QboHandler = Callable[[httpx.Request], httpx.Response]


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
