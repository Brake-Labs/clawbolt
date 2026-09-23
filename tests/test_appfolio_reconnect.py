"""A dead AppFolio credential reads as AUTH and is retired; an outage stays SERVICE.

Runs the real wiring: the tool factory builds the service from a credential
stored in ``oauth_tokens``, an AppFolio 401 refreshes through
``oauth_service.refresh_rejected_token`` (the shared advisory lock and the
AppFolio refresh grant), and the tool classifies what comes back. The
AppFolio API and its OAuth token endpoint are ``httpx`` fakes.

Token endpoint classification (``refresh_access_token``):

==============================================  ==========  =================
Token endpoint answer                           Tool error  Credential
==============================================  ==========  =================
400/401 with an RFC 6749 permanent ``error``    AUTH        deleted, notified
any other 4xx except 408/429 (HTML, 404, ...)   AUTH        kept, no notice
5xx, 408, 429                                   SERVICE     kept
network failure                                 SERVICE     kept
2xx without an access_token                     SERVICE     kept
refresh lock held by a peer                     SERVICE     kept
==============================================  ==========  =================
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.agent.tool_errors import build_error_hint
from backend.app.agent.tools.base import Tool, ToolErrorKind, ToolResult
from backend.app.agent.tools.registry import ToolContext
from backend.app.integrations.appfolio_vendor.auth import (
    INTEGRATION_NAME,
    AppFolioCredential,
    load_credential,
    save_credential,
    save_customer_ids,
)
from backend.app.integrations.appfolio_vendor.factory import (
    _appfolio_vendor_auth_check,
    _appfolio_vendor_factory,
)
from backend.app.models import User
from backend.app.services import oauth as oauth_module
from backend.app.services.oauth import oauth_service

Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)

_TOKEN_HOST = "oauth.appf.io"
_LOGIN_401 = {"login_url": "https://passport.example.invalid/authorize"}
_WORK_ORDERS = {"work_orders": [{"id": 7, "customer_id": "c1", "status": "open"}]}


@pytest.fixture()
def notify() -> Iterator[AsyncMock]:
    with patch.object(oauth_service, "_notify_reauth_needed", new_callable=AsyncMock) as mock:
        yield mock


async def _connect(user: User) -> None:
    await save_credential(
        user_id=user.id,
        jwt="jwt-old",
        fingerprint="fp-1",
        customer_ids=["c1"],
        refresh_token="rt-old",
    )


def _wire(monkeypatch: pytest.MonkeyPatch, *, api: Handler, token: Handler) -> None:
    """Send AppFolio API requests to *api* and token endpoint requests to *token*."""
    real_async_client = httpx.AsyncClient

    async def route(request: httpx.Request) -> httpx.Response:
        handler = token if request.url.host == _TOKEN_HOST else api
        response = handler(request)
        if not isinstance(response, httpx.Response):
            response = await response
        return response

    transport = httpx.MockTransport(route)

    def client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", client)


def _token(status: int, body: dict[str, Any] | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body if body is not None else {})

    return handler


def _token_network_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


_REFRESH_OK = _token(
    200, {"access_token": "jwt-new", "refresh_token": "rt-new", "expires_in": 7200}
)


def _api_accepts_only(jwt: str, seen: list[str] | None = None) -> Handler:
    """200 for requests bearing *jwt*, a ``login_url`` 401 for anything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        bearer = request.headers["Authorization"]
        if seen is not None:
            seen.append(bearer)
        if bearer == f"Bearer {jwt}":
            return httpx.Response(200, json=_WORK_ORDERS)
        return httpx.Response(401, json=_LOGIN_401)

    return handler


async def _tools(user: User) -> dict[str, Tool]:
    tools = await _appfolio_vendor_factory(ToolContext(user=user))
    assert tools, "factory returned no tools for a connected user"
    return {t.name: t for t in tools}


def _list(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["appfolio_list_work_orders"].function()


def _search(tools: dict[str, Tool]) -> Awaitable[ToolResult]:
    return tools["appfolio_search_work_orders"].function(search_term="leak")


async def _stored(user: User) -> AppFolioCredential | None:
    return await load_credential(user.id)


def _assert_web_app_reconnect(result: ToolResult) -> None:
    assert result.is_error is True
    assert result.error_kind is ToolErrorKind.AUTH
    hint = build_error_hint(result)
    assert "Integrations page of the Clawbolt web app" in hint
    assert "magic link" in hint
    # AppFolio has no chat connect flow, so no chat link is offered.
    assert "manage_integration(action='connect'" not in hint


def _assert_kept(stored: AppFolioCredential | None) -> None:
    assert stored is not None
    assert (stored.jwt, stored.refresh_token) == ("jwt-old", "rt-old")
    assert stored.fingerprint == "fp-1"


# ---------------------------------------------------------------------------
# Dead grant: AUTH, retired, notified once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, {"error": "invalid_grant", "error_description": "refresh token revoked"}),
        (401, {"error": "invalid_client"}),
    ],
    ids=["400_invalid_grant", "401_invalid_client"],
)
async def test_dead_refresh_grant_is_auth_retires_and_notifies_once(
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
    notify: AsyncMock,
    status: int,
    body: dict[str, Any],
) -> None:
    """Regression: a dead grant was reported as AUTH but never deleted or announced,
    so every later turn re-offered the tools and re-posted the dead refresh token."""
    await _connect(test_user)
    posts: list[httpx.Request] = []

    def token(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(status, json=body)

    _wire(monkeypatch, api=_api_accepts_only("jwt-new"), token=token)
    tools = await _tools(test_user)

    first = await _list(tools)
    # A later call in the same turn finds the credential gone: still AUTH,
    # no second POST of the dead refresh token, no second notice.
    second = await _search(tools)

    _assert_web_app_reconnect(first)
    _assert_web_app_reconnect(second)
    assert len(posts) == 1
    assert await _stored(test_user) is None
    notify.assert_awaited_once_with(test_user.id, INTEGRATION_NAME)


async def test_next_turn_after_retirement_offers_no_tools(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """Once retired, the auth check reports not connected and no tools are built."""
    await _connect(test_user)
    _wire(
        monkeypatch,
        api=_api_accepts_only("jwt-new"),
        token=_token(400, {"error": "invalid_grant"}),
    )
    _assert_web_app_reconnect(await _list(await _tools(test_user)))

    ctx = ToolContext(user=test_user)
    assert await _appfolio_vendor_auth_check(ctx) is not None
    assert await _appfolio_vendor_factory(ctx) == []
    notify.assert_awaited_once()


def _token_raw(status: int, content: bytes, content_type: str) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content, headers={"content-type": content_type})

    return handler


@pytest.mark.parametrize(
    "token",
    [
        _token_raw(403, b"<html><body>Request blocked</body></html>", "text/html"),
        _token(404, {"message": "Not Found"}),
        _token(400, {"message": "unexpected parameter", "code": 12}),
        _token(415, {"error": "unsupported_media_type"}),
        _token(403, {"error": "invalid_grant"}),
        _token(400),
    ],
    ids=["403_html", "404", "400_not_oauth", "415", "403_oauth_code", "400_empty"],
)
async def test_unconfirmed_refusal_is_auth_but_keeps_the_credential(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, token: Handler
) -> None:
    """Regression: any token endpoint 4xx retired the credential and notified the user.

    The endpoint is reverse-engineered and has moved before, so a WAF page, a
    404 or a body that is not OAuth may be our fault or transient. The model
    still hears AUTH, as before, but reconnecting stays the user's choice.
    """
    await _connect(test_user)
    _wire(monkeypatch, api=_api_accepts_only("jwt-new"), token=token)
    tools = await _tools(test_user)

    result = await _list(tools)

    assert result.is_error is True
    assert result.error_kind is ToolErrorKind.AUTH
    _assert_kept(await _stored(test_user))
    notify.assert_not_awaited()


# ---------------------------------------------------------------------------
# Transient failure: SERVICE, credential kept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        _token(500, {"error": "server_error"}),
        _token(503),
        _token(429, {"error": "rate_limited"}),
        _token(408),
        _token_network_error,
        _token(200, {"token_type": "Bearer"}),
    ],
    ids=["500", "503", "429", "408", "network_error", "200_without_token"],
)
async def test_transient_refresh_failure_is_service_and_keeps_the_credential(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, token: Handler
) -> None:
    """Regression: any token endpoint status >= 400 was reported as an expired session."""
    await _connect(test_user)
    _wire(monkeypatch, api=_api_accepts_only("jwt-new"), token=token)
    tools = await _tools(test_user)

    result = await _list(tools)

    assert result.is_error is True
    assert result.error_kind is ToolErrorKind.SERVICE
    assert "session expired" not in result.content
    _assert_kept(await _stored(test_user))
    notify.assert_not_awaited()


async def test_contended_refresh_lock_is_service_not_session_expired(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """A refresh that could not take the lock says nothing about the credential."""
    await _connect(test_user)
    posts: list[httpx.Request] = []

    def token(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(500)

    _wire(monkeypatch, api=_api_accepts_only("jwt-new"), token=token)

    async def contended(conn: Any, lock_key: str) -> bool:
        return False

    monkeypatch.setattr(oauth_module, "_try_acquire_advisory_lock_async", contended)
    tools = await _tools(test_user)

    result = await _list(tools)

    assert result.error_kind is ToolErrorKind.SERVICE
    assert "session expired" not in result.content
    assert posts == []
    _assert_kept(await _stored(test_user))
    notify.assert_not_awaited()


# ---------------------------------------------------------------------------
# Successful refresh
# ---------------------------------------------------------------------------


async def test_customer_ids_saved_during_the_refresh_post_survive_it(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    """Regression: the refresh saved ``extra`` from the copy it loaded before the POST,
    so customer IDs recorded while the POST was in flight were overwritten."""
    await _connect(test_user)

    async def token(request: httpx.Request) -> httpx.Response:
        await save_customer_ids(test_user.id, ["c1", "c2"])
        return httpx.Response(200, json={"access_token": "jwt-new", "refresh_token": "rt-new"})

    _wire(monkeypatch, api=_api_accepts_only("jwt-new"), token=token)
    tools = await _tools(test_user)

    assert (await _list(tools)).is_error is False

    stored = await _stored(test_user)
    assert stored is not None
    assert (stored.jwt, stored.refresh_token) == ("jwt-new", "rt-new")
    assert stored.customer_ids == ["c1", "c2"]
    assert stored.fingerprint == "fp-1"


async def test_refresh_persists_rotated_tokens_and_keeps_appfolio_metadata(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock
) -> None:
    await _connect(test_user)
    seen: list[str] = []
    _wire(monkeypatch, api=_api_accepts_only("jwt-new", seen), token=_REFRESH_OK)
    tools = await _tools(test_user)

    result = await _list(tools)

    assert result.is_error is False
    assert seen == ["Bearer jwt-old", "Bearer jwt-new"]
    stored = await _stored(test_user)
    assert stored is not None
    assert (stored.jwt, stored.refresh_token) == ("jwt-new", "rt-new")
    assert stored.fingerprint == "fp-1"
    assert stored.customer_ids == ["c1"]
    # No expiry is stored, so the background sweep leaves AppFolio alone and
    # it keeps refreshing only on a 401, as before.
    token = await oauth_service.load_token_uncached(test_user.id, INTEGRATION_NAME)
    assert token is not None
    assert token.expires_at == 0.0
    notify.assert_not_awaited()


@pytest.mark.parametrize("shared_service", [True, False], ids=["one_turn", "two_workers"])
async def test_concurrent_401s_refresh_once(
    test_user: User, monkeypatch: pytest.MonkeyPatch, notify: AsyncMock, shared_service: bool
) -> None:
    """Two calls get a 401 while the first refresh is still in flight.

    ``one_turn``: parallel tool calls sharing one service. Regression: the
    second 401 skipped its refresh and raised a false "session expired".
    ``two_workers``: two services built from the same credential, as a
    heartbeat and a user turn are. Regression: both posted the same refresh
    token and the second save overwrote the first.
    """
    await _connect(test_user)
    rejected = 0
    both_rejected = asyncio.Event()
    posted: list[str] = []

    def api(request: httpx.Request) -> httpx.Response:
        nonlocal rejected
        if request.headers["Authorization"] == "Bearer jwt-new":
            return httpx.Response(200, json=_WORK_ORDERS)
        rejected += 1
        if rejected == 2:
            both_rejected.set()
        return httpx.Response(401, json=_LOGIN_401)

    async def token(request: httpx.Request) -> httpx.Response:
        posted.append(request.read().decode())
        # Hold the refresh open until the other call has its 401 too.
        await asyncio.wait_for(both_rejected.wait(), timeout=5)
        return httpx.Response(
            200, json={"access_token": "jwt-new", "refresh_token": "rt-new", "expires_in": 7200}
        )

    _wire(monkeypatch, api=api, token=token)
    first_tools = await _tools(test_user)
    second_tools = first_tools if shared_service else await _tools(test_user)

    results = await asyncio.gather(_list(first_tools), _search(second_tools))

    assert [r.is_error for r in results] == [False, False], [r.content for r in results]
    assert len(posted) == 1
    assert "rt-old" in posted[0]
    stored = await _stored(test_user)
    assert stored is not None
    assert (stored.jwt, stored.refresh_token) == ("jwt-new", "rt-new")
    notify.assert_not_awaited()
