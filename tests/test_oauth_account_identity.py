"""Recording which Google account an OAuth connection belongs to."""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.parse
from collections.abc import Callable, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from backend.app.agent.tools.integration_tools import _handle_status
from backend.app.agent.tools.registry import default_registry, ensure_tool_modules_imported
from backend.app.auth.dependencies import get_current_user
from backend.app.config import settings
from backend.app.database import db_session_async
from backend.app.main import app
from backend.app.models import User
from backend.app.services.oauth import (
    ACCOUNT_EMAIL_KEY,
    OAuthConfig,
    OAuthService,
    OAuthTokenData,
    get_gmail_oauth_config,
    get_google_calendar_oauth_config,
    get_google_drive_oauth_config,
    oauth_service,
)

ACCOUNT = "owner@example.com"


@pytest_asyncio.fixture()
async def user() -> User:
    async with db_session_async() as db:
        u = User(user_id="oauth-identity-user", onboarding_complete=True)
        db.add(u)
        await db.commit()
        await db.refresh(u)
        db.expunge(u)
    return u


def _config(integration: str) -> OAuthConfig:
    return OAuthConfig(
        integration=integration,
        client_id="cid",
        client_secret="csec",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["scope"],
        use_pkce=False,
    )


def _json(body: dict[str, object], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", "https://x.test"))


async def _connect(
    svc: OAuthService, user_id: str, integration: str, identity: httpx.Response | Exception
) -> tuple[OAuthTokenData, MagicMock]:
    config = _config(integration)
    url = svc.get_authorization_url(config, user_id)
    state = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["state"][0]
    client = MagicMock()
    client.post = AsyncMock(
        return_value=_json({"access_token": "at", "refresh_token": "rt", "expires_in": 3600})
    )
    client.get = (
        AsyncMock(side_effect=identity)
        if isinstance(identity, Exception)
        else AsyncMock(return_value=identity)
    )
    with (
        patch.object(svc, "_get_http", return_value=client),
        patch("backend.app.services.oauth.get_oauth_config", return_value=config),
    ):
        token = await svc.handle_callback(state, "code")
    return token, client


@pytest.mark.parametrize(
    ("integration", "body", "url_part"),
    [
        ("gmail", {"emailAddress": "Owner@Example.com"}, "gmail/v1/users/me/profile"),
        ("google_calendar", {"id": ACCOUNT}, "calendarList/primary"),
        ("google_drive", {"user": {"emailAddress": ACCOUNT}}, "drive/v3/about"),
    ],
)
async def test_connect_records_google_account(
    user: User, integration: str, body: dict[str, object], url_part: str
) -> None:
    svc = OAuthService()
    token, client = await _connect(svc, user.id, integration, _json(body))

    assert token.account_email == ACCOUNT
    stored = await svc.load_token_uncached(user.id, integration)
    assert stored is not None
    assert stored.extra[ACCOUNT_EMAIL_KEY] == ACCOUNT
    called_url = client.get.call_args.args[0]
    assert url_part in called_url
    assert client.get.call_args.kwargs["headers"]["Authorization"] == "Bearer at"


async def test_identity_lookup_failure_still_connects(
    user: User, caplog: pytest.LogCaptureFixture
) -> None:
    svc = OAuthService()
    with caplog.at_level(logging.DEBUG):
        token, _ = await _connect(
            svc, user.id, "gmail", _json({"error": {"code": 403, "message": ACCOUNT}}, 403)
        )
    assert token.account_email == ""
    stored = await svc.load_token_uncached(user.id, "gmail")
    assert stored is not None
    assert stored.access_token == "at"
    # PII: neither the email nor the response body reaches the logs.
    assert ACCOUNT not in caplog.text


async def test_identity_lookup_skipped_for_non_google(user: User) -> None:
    svc = OAuthService()
    token, client = await _connect(svc, user.id, "quickbooks", _json({}))
    client.get.assert_not_called()
    assert token.account_email == ""


async def test_reconnect_replaces_recorded_account(user: User) -> None:
    svc = OAuthService()
    await _connect(svc, user.id, "google_calendar", _json({"id": "first@example.com"}))
    await _connect(svc, user.id, "google_calendar", _json({"id": "second@example.com"}))
    assert await svc.get_account_email(user.id, "google_calendar") == "second@example.com"


async def test_refresh_keeps_recorded_account(user: User) -> None:
    svc = OAuthService()
    await svc.save_token(
        user.id,
        "google_calendar",
        OAuthTokenData(access_token="old", refresh_token="rt", extra={ACCOUNT_EMAIL_KEY: ACCOUNT}),
    )
    await svc.build_on_refresh_callback(user.id, "google_calendar")(
        "new", "rt2", time.time() + 3600
    )
    assert await svc.get_account_email(user.id, "google_calendar") == ACCOUNT


async def test_post_connect_hook_failure_does_not_fail_connect(user: User) -> None:
    svc = OAuthService()
    hook = AsyncMock(side_effect=RuntimeError("boom"))
    svc.register_post_connect_hook("google_calendar", hook)
    svc.register_post_connect_hook("google_calendar", hook)  # idempotent
    token, _ = await _connect(svc, user.id, "google_calendar", _json({"id": ACCOUNT}))
    assert token.access_token == "at"
    hook.assert_awaited_once()
    assert hook.call_args.args[0] == user.id


async def test_slow_post_connect_hook_is_bounded(user: User) -> None:
    svc = OAuthService()

    async def _hang(user_id: str, token: OAuthTokenData) -> None:
        await asyncio.sleep(60)

    svc.register_post_connect_hook("google_calendar", _hang)
    with patch("backend.app.services.oauth._POST_CONNECT_HOOK_TIMEOUT_S", 0.01):
        token, _ = await _connect(svc, user.id, "google_calendar", _json({"id": ACCOUNT}))
    assert token.access_token == "at"
    assert await svc.get_account_email(user.id, "google_calendar") == ACCOUNT


async def test_unknown_account_for_legacy_token(user: User) -> None:
    svc = OAuthService()
    await svc.save_token(user.id, "gmail", OAuthTokenData(access_token="at"))
    assert await svc.get_account_email(user.id, "gmail") == ""


# ---------------------------------------------------------------------------
# Surfaces: /oauth/status (Settings) and manage_integration status (agent)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(user: User) -> Generator[TestClient]:
    app.dependency_overrides[get_current_user] = lambda: user
    with (
        patch("backend.app.main._verify_llm_settings", new_callable=AsyncMock),
        patch("backend.app.agent.heartbeat.heartbeat_scheduler.start"),
        TestClient(app) as c,
    ):
        yield c
    app.dependency_overrides.clear()


async def test_status_endpoint_reports_account(client: TestClient, user: User) -> None:
    await oauth_service.save_token(
        user.id,
        "google_calendar",
        OAuthTokenData(access_token="at", extra={ACCOUNT_EMAIL_KEY: ACCOUNT}),
    )
    await oauth_service.save_token(user.id, "gmail", OAuthTokenData(access_token="at"))
    with (
        patch.object(settings, "google_calendar_client_id", "cid"),
        patch.object(settings, "google_calendar_client_secret", "cs"),
    ):
        resp = client.get("/api/oauth/status")
    assert resp.status_code == 200
    by_name = {e["integration"]: e for e in resp.json()["integrations"]}
    assert by_name["google_calendar"]["account_email"] == ACCOUNT
    assert by_name["gmail"]["connected"] is True
    assert by_name["gmail"]["account_email"] is None
    assert by_name["quickbooks"]["account_email"] is None


async def test_manage_integration_status_reports_account(user: User) -> None:
    ensure_tool_modules_imported()
    await oauth_service.save_token(
        user.id,
        "google_calendar",
        OAuthTokenData(access_token="at", extra={ACCOUNT_EMAIL_KEY: ACCOUNT}),
    )
    await oauth_service.save_token(user.id, "gmail", OAuthTokenData(access_token="at"))
    with (
        patch.object(settings, "google_calendar_client_id", "cid"),
        patch.object(settings, "google_calendar_client_secret", "cs"),
        patch.object(settings, "gmail_client_id", "cid"),
        patch.object(settings, "gmail_client_secret", "cs"),
    ):
        result = await _handle_status(user.id, default_registry)

    calendar_line = next(ln for ln in result.content.splitlines() if "calendar:" in ln)
    assert f"Google account: {ACCOUNT}" in calendar_line
    gmail_line = next(ln for ln in result.content.splitlines() if ln.startswith("- gmail"))
    assert "Google account: unknown" in gmail_line


@pytest.mark.parametrize(
    "config_factory",
    [
        get_gmail_oauth_config,
        get_google_calendar_oauth_config,
        get_google_drive_oauth_config,
    ],
)
def test_google_connect_links_ask_which_account(
    config_factory: Callable[[], OAuthConfig | None],
) -> None:
    """Without select_account Google silently uses the browser's current
    account, so a user with two accounts connects the wrong one and their
    saved calendars and files read as missing."""
    with patch.multiple(
        settings,
        gmail_client_id="cid",
        gmail_client_secret="secret",
        google_calendar_client_id="cid",
        google_calendar_client_secret="secret",
        google_drive_client_id="cid",
        google_drive_client_secret="secret",
    ):
        config = config_factory()
    assert config is not None
    url = OAuthService().get_authorization_url(config, "user-1")
    prompt = httpx.URL(url).params["prompt"]
    assert "select_account" in prompt
    # consent still has to be there: Google only returns a refresh token with it.
    assert "consent" in prompt
