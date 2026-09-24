"""Generic OAuth 2.0 service with PKCE support.

Handles authorization URL generation, callback processing, token storage,
and automatic token refresh. Tokens are persisted in PostgreSQL (oauth_tokens
table) with encrypted access/refresh token columns.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import json
import logging
import random
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import sqlalchemy as sa
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.config import settings
from backend.app.database import db_session_async, get_async_engine

logger = logging.getLogger(__name__)


def _refresh_lock_key(user_id: str, integration: str) -> str:
    """Advisory-lock key for serializing OAuth token refresh per (user, integration)."""
    return f"oauth_refresh:{user_id}:{integration}"


# Bound on how long we'll wait for an OAuth refresh advisory lock.
# Session-scoped advisory locks survive only as long as the holder's PG
# session does, but a connection that drops without proper teardown can
# leave a lock held until Postgres notices the peer is gone, which on a
# proxied/NAT'd setup can take an hour or more. ``pg_advisory_lock`` would
# block the calling thread (and a sync DB call inside an async route would
# block the entire event loop) for that whole window. Polling
# ``pg_try_advisory_lock`` with a bounded wait fails fast instead: a
# legitimate peer normally releases within milliseconds, and a stale lock
# means we skip this refresh attempt and let the next one (sweep tick or
# next get_valid_token call) try again.
_LOCK_RETRY_INTERVAL_S = 0.1
_LOCK_MAX_WAIT_S = 5.0


async def _try_acquire_advisory_lock_async(conn: Any, lock_key: str) -> bool:
    """Bounded acquire of a session-scoped advisory lock.

    Returns True on acquire, False if the wait expires. Uses
    ``await asyncio.sleep`` between polls so the event loop stays
    responsive during contention.

    ``conn`` MUST be a SQLAlchemy ``AsyncConnection`` (or other handle
    whose ``commit()`` does NOT return the underlying DBAPI connection
    to the pool). Passing an ``AsyncSession`` would break the lock
    semantics: in SQLAlchemy 2.0 ``AsyncSession.commit()`` releases the
    underlying connection back to the pool, where a peer task can pick
    it up and call ``pg_try_advisory_lock`` on it (locks are reentrant
    per PG session), causing both callers to enter the critical
    section. The advisory lock must stay pinned to the same physical
    connection from acquire through unlock; only ``AsyncConnection``
    provides that pinning.
    """
    deadline = time.monotonic() + _LOCK_MAX_WAIT_S
    while True:
        result = await conn.execute(
            text("SELECT pg_try_advisory_lock(hashtext(:k))"),
            {"k": lock_key},
        )
        acquired = result.scalar()
        # Commit ends the implicit transaction so we don't sit
        # idle-in-transaction between polls. On an AsyncConnection this
        # does NOT return the connection to the pool, so the
        # session-scoped advisory lock stays attached to this same
        # connection.
        await conn.commit()
        if acquired:
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_LOCK_RETRY_INTERVAL_S)


# Per-process TTL on cached tokens. A single agent turn loads OAuth
# credentials repeatedly (auth_check during registry build, factory
# create() during tool instantiation, then again per actual tool call).
# In production logs we saw 6+ load_token DB hits per inbound. This
# cache deduplicates them. Kept short so a refresh in another worker
# becomes visible quickly; even a stale read here is safe because the
# returned access_token has its own expires_at that callers honor.
_TOKEN_CACHE_TTL_SECONDS = 30.0

# Shorter TTL for negative results (no token row exists). Cross-worker
# OAuth completion race: worker B handles the OAuth callback and
# save_token invalidates B's local cache, but worker A's cache still
# has a "not connected" entry from a recent is_connected() check. Until
# A's entry expires, A reports the user as unconnected even though
# they just finished connecting. Keep negative TTL short enough that
# the post-OAuth blackout is barely noticeable, but long enough to
# still dedupe the multiple auth_check reads within a single agent turn.
_NEGATIVE_TOKEN_CACHE_TTL_SECONDS = 5.0


# Token expiry buffer: refresh 5 minutes before actual expiry.
_EXPIRY_BUFFER_SECONDS = 300

# Intuit discovery document URL (OpenID Connect configuration).
_INTUIT_DISCOVERY_URL = "https://developer.api.intuit.com/.well-known/openid_configuration"

# Cache TTL for the discovery document (24 hours).
_DISCOVERY_CACHE_TTL_SECONDS = 86400

# OAuth state entries expire after 10 minutes.
_STATE_TTL_SECONDS = 600

# RFC 6749 Section 5.2 error codes indicating a permanently invalid token.
# These mean the user must re-authenticate; retrying will not help.
_PERMANENT_OAUTH_ERROR_CODES = frozenset(
    {
        "invalid_grant",
        "invalid_client",
        "unauthorized_client",
    }
)


# Integrations whose credential is a pasted secret entered in the web app,
# never over chat (issue #1337), so ``manage_integration(action='connect')``
# has no link to offer for them. Keyed to the reconnect wording the agent
# relays instead.
_WEB_CONNECT_RECONNECT_INSTRUCTIONS: dict[str, str] = {
    "appfolio_vendor": (
        "AppFolio has no chat connect flow, so do not offer a connection link. Have the "
        "user reconnect AppFolio on the Integrations page of the Clawbolt web app with a "
        "fresh magic link (requested from vendor.appfolio.com). Do not ask them to paste "
        "the link into chat."
    ),
}

# Names for integrations whose key does not title-case into their product name.
_DISPLAY_NAMES: dict[str, str] = {"appfolio_vendor": "AppFolio Vendor Portal"}


def _display_name(integration: str) -> str:
    return _DISPLAY_NAMES.get(integration) or integration.replace("_", " ").title()


def reconnect_instruction(integration: str) -> str:
    """How the agent gets a dead connection back: offer the user a fresh link.

    A refused token is not always retired (a 401 that survives a refresh, or
    a refresh that could not run, leaves it stored), and ``connect`` refuses
    while a token is stored, so the instruction covers disconnecting first.
    Integrations connected only in the web app point the user there instead.
    """
    web_connect = _WEB_CONNECT_RECONNECT_INSTRUCTIONS.get(integration)
    if web_connect is not None:
        return web_connect
    return (
        f"Use manage_integration(action='connect', target='{integration}') to generate "
        "a connection link for the user. If it reports the integration is still "
        f"connected, run manage_integration(action='disconnect', target='{integration}') "
        "first. The user can also reconnect in Settings > Integrations."
    )


class ReconnectRequired(Exception):
    """The provider has rejected the connection itself; only reconnecting fixes it.

    Raised mid-call when a token refresh fails permanently (the stored token
    has already been retired and the user notified) or when the provider
    still rejects a freshly refreshed token. Tools classify it as
    ``ToolErrorKind.AUTH`` so the agent tells the user to reconnect instead
    of calling the service temporarily unavailable.
    """

    def __init__(self, integration: str, message: str) -> None:
        super().__init__(message)
        self.integration = integration


class PermanentRefreshError(Exception):
    """A registered refresh grant's verdict that the grant itself is dead.

    Raised by a grant registered with ``OAuthService.register_refresh_grant``
    when its token endpoint refuses the refresh token in a way retrying cannot
    fix. The shared refresh path treats it like an RFC 6749 ``invalid_grant``:
    it retires the token and notifies the user.
    """


class RefreshLockContended(Exception):
    """A peer held the refresh lock past the bounded wait, so no refresh ran.

    Transient and says nothing about the grant. Raised by
    ``refresh_rejected_token`` (and so by the mid-call refresh hook) so a
    provider service can report it as retryable instead of letting the
    original 401 read as a dead connection.
    """


class TokenRefreshUnavailable(httpx.HTTPStatusError):
    """The provider refused the access token (401) while the refresh lock was busy.

    Raised by a provider service when its mid-call refresh hook raises
    ``RefreshLockContended``. That says nothing about the grant, so callers
    classify it as a transient ``ToolErrorKind.SERVICE`` rather than as a
    dead connection. It is an ``HTTPStatusError`` carrying the 401, so
    handlers that branch on a 401 must check for this type first.
    """


def is_dead_grant_response(resp: httpx.Response) -> bool:
    """True when a token endpoint answer says the grant itself is dead.

    Only a 400 or 401 whose JSON ``error`` is an RFC 6749 permanent code
    counts. Anything else (an HTML error page, a moved endpoint, a body that
    is not OAuth) is unrecognized, and retiring a credential on it would cost
    the user a manual reconnect for what may be a transient or local fault.
    """
    if resp.status_code not in (400, 401):
        return False
    try:
        body = resp.json()
    except Exception:
        return False
    return isinstance(body, dict) and body.get("error") in _PERMANENT_OAUTH_ERROR_CODES


# A refresh grant an integration registers when its token endpoint does not
# take the standard form-encoded, client-secret refresh. Takes the stored
# refresh token and returns the token endpoint's payload (``access_token``,
# optionally ``refresh_token`` and ``expires_in``). Raises
# ``PermanentRefreshError`` for a dead grant and anything else for a
# transient failure.
RefreshGrant = Callable[[str], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Intuit discovery document cache
# ---------------------------------------------------------------------------

_intuit_discovery_cache: dict[str, Any] = {}
_intuit_discovery_fetched_at: float = 0.0


async def warm_intuit_discovery() -> None:
    """Fetch and cache the Intuit OpenID Connect discovery document.

    Called at app startup so that ``get_quickbooks_oauth_config()`` can
    resolve endpoints from the discovery document instead of relying on
    hardcoded URLs. Failures are logged and swallowed; the hardcoded
    fallback URLs will be used until the next successful fetch.
    """
    global _intuit_discovery_cache, _intuit_discovery_fetched_at
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_INTUIT_DISCOVERY_URL)
            resp.raise_for_status()
            _intuit_discovery_cache = resp.json()
            _intuit_discovery_fetched_at = time.time()
            logger.info(
                "Intuit discovery document cached: authorization_endpoint=%s token_endpoint=%s",
                _intuit_discovery_cache.get("authorization_endpoint"),
                _intuit_discovery_cache.get("token_endpoint"),
            )
    except Exception:
        logger.warning(
            "Failed to fetch Intuit discovery document from %s, "
            "falling back to hardcoded endpoints",
            _INTUIT_DISCOVERY_URL,
            exc_info=True,
        )


def _get_intuit_endpoints() -> tuple[str, str]:
    """Return (authorize_url, token_url) from the discovery cache or fallbacks.

    If the cache is stale (older than ``_DISCOVERY_CACHE_TTL_SECONDS``) or
    missing, falls back to the hardcoded endpoint constants.
    """
    if (
        _intuit_discovery_cache
        and (time.time() - _intuit_discovery_fetched_at) < _DISCOVERY_CACHE_TTL_SECONDS
    ):
        authorize = _intuit_discovery_cache.get(
            "authorization_endpoint", _QBO_AUTHORIZE_URL_FALLBACK
        )
        token = _intuit_discovery_cache.get("token_endpoint", _QBO_TOKEN_URL_FALLBACK)
        return authorize, token
    return _QBO_AUTHORIZE_URL_FALLBACK, _QBO_TOKEN_URL_FALLBACK


@dataclass
class OAuthConfig:
    """Configuration for an OAuth 2.0 integration."""

    integration: str
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    scopes: list[str]
    callback_path: str = "/api/oauth/callback"
    use_pkce: bool = True
    extra_auth_params: dict[str, str] = field(default_factory=dict)

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


# Key in ``OAuthTokenData.extra`` (persisted as ``oauth_tokens.extra_json``)
# holding the email of the Google account a connection was granted by.
# Captured once at connect time; absent on tokens stored before it existed,
# which every reader must render as "unknown", not as an error.
ACCOUNT_EMAIL_KEY = "account_email"

# Google endpoints that return the account email under scopes each Google
# integration already requests, so identifying the account needs no extra
# scope and no consent-screen change:
# - Gmail ``users.getProfile`` (gmail.readonly) returns ``emailAddress``.
# - Calendar's primary calendar (calendar.readonly) has the account email as
#   its id.
# - Drive ``about.get`` with ``fields=user`` is allowed under drive.file.
# ``openid email`` would also work but adds a scope to every consent screen.
_GOOGLE_ACCOUNT_IDENTITY_ENDPOINTS: dict[str, tuple[str, dict[str, str], tuple[str, ...]]] = {
    "gmail": (
        "https://gmail.googleapis.com/gmail/v1/users/me/profile",
        {},
        ("emailAddress",),
    ),
    "google_calendar": (
        "https://www.googleapis.com/calendar/v3/users/me/calendarList/primary",
        {"fields": "id"},
        ("id",),
    ),
    "google_drive": (
        "https://www.googleapis.com/drive/v3/about",
        {"fields": "user(emailAddress)"},
        ("user", "emailAddress"),
    ),
}

# Integrations whose connections record the granting account's email.
ACCOUNT_TRACKED_INTEGRATIONS = frozenset(_GOOGLE_ACCOUNT_IDENTITY_ENDPOINTS)

# Called after a successful connect or reconnect with the user id and the
# freshly stored token. Registered per integration by the integration itself
# (``register_post_connect_hook``) so this module never imports integrations.
PostConnectHook = Callable[[str, "OAuthTokenData"], Awaitable[None]]

# Both run inside the OAuth callback request, before the user is redirected,
# so each is bounded well under typical proxy timeouts.
_ACCOUNT_LOOKUP_TIMEOUT_S = 10.0
_POST_CONNECT_HOOK_TIMEOUT_S = 20.0


def parse_extra_json(raw: str) -> dict[str, Any]:
    """Decode an ``oauth_tokens.extra_json`` value, tolerating empty or corrupt rows."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def account_email_from_extra(extra: dict[str, Any]) -> str:
    """The recorded account email in a token's extra data, or "" when unknown."""
    value = extra.get(ACCOUNT_EMAIL_KEY, "")
    return value if isinstance(value, str) else ""


@dataclass
class _PendingState:
    """In-memory record for a pending OAuth authorization."""

    user_id: str
    integration: str
    code_verifier: str
    redirect_uri: str
    expires_at: float
    source: str = "web"


@dataclass
class OAuthTokenData:
    """Stored OAuth token data."""

    access_token: str
    refresh_token: str = ""
    token_type: str = "Bearer"
    expires_at: float = 0.0
    scopes: list[str] = field(default_factory=list)
    realm_id: str = ""  # QuickBooks company ID
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def account_email(self) -> str:
        """Email of the account that granted this token, or "" when unknown."""
        return account_email_from_extra(self.extra)

    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() >= (self.expires_at - _EXPIRY_BUFFER_SECONDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at,
            "scopes": self.scopes,
            "realm_id": self.realm_id,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthTokenData:
        return cls(
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_at=data.get("expires_at", 0.0),
            scopes=data.get("scopes", []),
            realm_id=data.get("realm_id", ""),
            extra=data.get("extra", {}),
        )


def _generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class OAuthService:
    """Manages OAuth flows and token lifecycle.

    State is held in memory (pending authorization flows, a small TTL
    cache of recent token reads) and in PostgreSQL (persisted tokens via
    the oauth_tokens table).

    Token cache semantics:

    - Per-process: each uvicorn worker has its own cache, no shared memory.
    - Positive entries live ``_TOKEN_CACHE_TTL_SECONDS``; negative entries
      live ``_NEGATIVE_TOKEN_CACHE_TTL_SECONDS`` (shorter, to keep the
      post-OAuth-completion blackout small).
    - ``save_token`` and ``delete_token`` invalidate the local cache.
    - Inside an advisory-lock critical section, callers MUST use
      ``load_token_uncached`` to defeat the cache: the whole point of
      the post-lock reload is to observe a peer worker's just-persisted
      refresh, and a stale cache would mask it.
    - Cross-worker staleness is bounded by the TTL. Within that window a
      worker may return a token whose access_token was rotated by a peer,
      but the cached access_token's own expires_at is honored by callers,
      so behavior is correct.
    """

    def __init__(self) -> None:
        self._pending_states: dict[str, _PendingState] = {}
        self._http: httpx.AsyncClient | None = None
        # In-memory TTL cache for load_token. Keyed by (user_id, integration);
        # value is (token_data_or_none, monotonic_expires_at).
        self._token_cache: dict[tuple[str, str], tuple[OAuthTokenData | None, float]] = {}
        self._post_connect_hooks: dict[str, list[PostConnectHook]] = {}
        self._refresh_grants: dict[str, RefreshGrant] = {}

    def register_refresh_grant(self, integration: str, grant: RefreshGrant) -> None:
        """Refresh *integration*'s tokens through *grant* instead of an ``OAuthConfig``.

        For integrations stored in ``oauth_tokens`` whose token endpoint needs
        its own request shape (AppFolio posts JSON with a public client id).
        Everything around the POST stays shared: the advisory lock, the
        peer-refresh check, persistence, and retiring a dead grant.
        """
        self._refresh_grants[integration] = grant

    def register_post_connect_hook(self, integration: str, hook: PostConnectHook) -> None:
        """Run *hook* after every successful connect or reconnect of *integration*.

        Hooks are best-effort: a failure is logged and never fails the
        connect, because the token is already stored by the time they run.
        Registering the same function twice is a no-op, so a module that
        registers at import time stays idempotent across reloads.
        """
        hooks = self._post_connect_hooks.setdefault(integration, [])
        if hook not in hooks:
            hooks.append(hook)

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    # -- Authorization URL generation ------------------------------------------

    def get_authorization_url(
        self,
        config: OAuthConfig,
        user_id: str,
        source: str = "web",
    ) -> str:
        """Build an authorization URL with PKCE and state parameter.

        *source* tracks where the flow was initiated from ("web" for the
        frontend UI, "chat" for the manage_integration tool). The callback
        uses this to decide whether to redirect to the SPA or render a
        standalone confirmation page.
        """
        self._cleanup_expired_states()

        state = secrets.token_urlsafe(32)
        verifier, challenge = _generate_pkce_pair()

        base_url = settings.app_base_url.rstrip("/")
        redirect_uri = f"{base_url}{config.callback_path}"

        self._pending_states[state] = _PendingState(
            user_id=user_id,
            integration=config.integration,
            code_verifier=verifier,
            redirect_uri=redirect_uri,
            expires_at=time.time() + _STATE_TTL_SECONDS,
            source=source,
        )

        params: dict[str, str] = {
            "client_id": config.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(config.scopes),
            "state": state,
        }
        if config.use_pkce:
            params["code_challenge"] = challenge
            params["code_challenge_method"] = "S256"
        if config.extra_auth_params:
            params.update(config.extra_auth_params)

        return str(httpx.URL(config.authorize_url, params=params))

    # -- Callback handling -----------------------------------------------------

    async def handle_callback(
        self,
        state: str,
        code: str,
        *,
        realm_id: str = "",
    ) -> OAuthTokenData:
        """Exchange an authorization code for tokens and store them."""
        pending = self._pending_states.pop(state, None)
        if pending is None:
            raise ValueError("Invalid or expired OAuth state")

        if time.time() > pending.expires_at:
            raise ValueError("OAuth state has expired")

        config = get_oauth_config(pending.integration)
        if config is None:
            raise ValueError(f"No OAuth config for integration: {pending.integration}")

        token_data = await self._exchange_code(
            config=config,
            code=code,
            redirect_uri=pending.redirect_uri,
            code_verifier=pending.code_verifier,
        )

        if realm_id:
            token_data.realm_id = realm_id

        account_email = await self._fetch_account_email(
            pending.integration, token_data.access_token
        )
        if account_email:
            token_data.extra[ACCOUNT_EMAIL_KEY] = account_email

        await self.save_token(pending.user_id, pending.integration, token_data)
        await self._run_post_connect_hooks(pending.user_id, pending.integration, token_data)
        return token_data

    async def _fetch_account_email(self, integration: str, access_token: str) -> str:
        """Best-effort lookup of the account email a Google token belongs to.

        Returns "" for non-Google integrations and on any failure: a missing
        account label must never fail the connect. The email itself is PII,
        so neither it nor the response body is logged.
        """
        endpoint = _GOOGLE_ACCOUNT_IDENTITY_ENDPOINTS.get(integration)
        if endpoint is None or not access_token:
            return ""
        url, params, path = endpoint
        try:
            resp = await self._get_http().get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=_ACCOUNT_LOOKUP_TIMEOUT_S,
            )
            resp.raise_for_status()
            value: Any = resp.json()
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
        except Exception as exc:
            logger.warning(
                "Could not identify the connected account: integration=%s error=%s",
                integration,
                type(exc).__name__,
            )
            return ""
        if not isinstance(value, str) or "@" not in value:
            logger.warning(
                "Connected-account lookup returned no email: integration=%s", integration
            )
            return ""
        return value.strip().lower()

    async def _run_post_connect_hooks(
        self, user_id: str, integration: str, token: OAuthTokenData
    ) -> None:
        for hook in self._post_connect_hooks.get(integration, []):
            try:
                await asyncio.wait_for(hook(user_id, token), _POST_CONNECT_HOOK_TIMEOUT_S)
            except Exception:
                logger.exception(
                    "Post-connect hook failed, connect still succeeded: user=%s integration=%s",
                    user_id,
                    integration,
                )

    async def get_account_email(self, user_id: str, integration: str) -> str:
        """Return the account email recorded for a connection, or "" if unknown."""
        token = await self.load_token(user_id, integration)
        return token.account_email if token is not None else ""

    async def _exchange_code(
        self,
        config: OAuthConfig,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> OAuthTokenData:
        """Exchange authorization code for access and refresh tokens."""
        http = self._get_http()
        token_data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
        if config.use_pkce:
            token_data["code_verifier"] = code_verifier
        resp = await http.post(
            config.token_url,
            data=token_data,
            auth=(config.client_id, config.client_secret),
        )
        resp.raise_for_status()
        body = resp.json()

        expires_at = 0.0
        if "expires_in" in body:
            expires_at = time.time() + body["expires_in"]

        scope_raw = body.get("scope", "")
        scopes = scope_raw.split() if isinstance(scope_raw, str) else []

        return OAuthTokenData(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token", ""),
            token_type=body.get("token_type", "Bearer"),
            expires_at=expires_at,
            scopes=scopes,
        )

    # -- Token persistence (database-backed) ------------------------------------

    async def save_token(
        self,
        user_id: str,
        integration: str,
        token: OAuthTokenData,
    ) -> None:
        """Persist token data to the oauth_tokens table (atomic upsert)."""
        from backend.app.models import OAuthToken

        values = {
            "user_id": user_id,
            "integration": integration,
            "access_token": token.access_token,
            "refresh_token": token.refresh_token,
            "token_type": token.token_type,
            "expires_at": token.expires_at,
            "scopes_json": json.dumps(token.scopes),
            "realm_id": token.realm_id,
            "extra_json": json.dumps(token.extra),
        }
        update_cols = {k: v for k, v in values.items() if k not in ("user_id", "integration")}
        update_cols["updated_at"] = sa.func.now()

        stmt = (
            pg_insert(OAuthToken)
            .values(**values)
            .on_conflict_do_update(
                constraint="uq_oauth_token_user_integration",
                set_=update_cols,
            )
        )

        async with db_session_async() as db:
            await db.execute(stmt)
            await db.commit()
        # Drop any cached entry so the next load_token re-reads the
        # freshly persisted row (e.g. after a refresh writes new tokens).
        self._token_cache.pop((user_id, integration), None)

    async def load_token(
        self,
        user_id: str,
        integration: str,
    ) -> OAuthTokenData | None:
        """Load token data from the oauth_tokens table.

        Results are cached in-memory so a single agent turn (auth_check,
        factory create, tool invocation) does not produce N duplicate DB
        roundtrips. Positive entries cache for ``_TOKEN_CACHE_TTL_SECONDS``;
        negative entries (no row) cache for the shorter
        ``_NEGATIVE_TOKEN_CACHE_TTL_SECONDS`` so a freshly-completed
        OAuth flow on a peer worker becomes visible quickly. The cache
        is invalidated on ``save_token`` and ``delete_token`` within this
        process. Callers inside an advisory-lock critical section must
        use ``load_token_uncached`` instead.
        """
        cache_key = (user_id, integration)
        now = time.monotonic()
        cached = self._token_cache.get(cache_key)
        if cached is not None and cached[1] > now:
            return cached[0]

        from backend.app.models import OAuthToken

        logger.info(
            "credential.read action=load user=%s integration=%s",
            user_id,
            integration,
        )
        async with db_session_async() as db:
            row = (
                await db.execute(
                    select(OAuthToken).where(
                        OAuthToken.user_id == user_id,
                        OAuthToken.integration == integration,
                    )
                )
            ).scalar_one_or_none()

            if row is None:
                self._token_cache[cache_key] = (
                    None,
                    now + _NEGATIVE_TOKEN_CACHE_TTL_SECONDS,
                )
                return None

            try:
                scopes = json.loads(row.scopes_json) if row.scopes_json else []
            except json.JSONDecodeError:
                scopes = []

            extra = parse_extra_json(row.extra_json)

            token = OAuthTokenData(
                access_token=row.access_token,
                refresh_token=row.refresh_token,
                token_type=row.token_type,
                expires_at=row.expires_at,
                scopes=scopes,
                realm_id=row.realm_id,
                extra=extra,
            )
            self._token_cache[cache_key] = (token, now + _TOKEN_CACHE_TTL_SECONDS)
            return token

    async def load_token_uncached(
        self,
        user_id: str,
        integration: str,
    ) -> OAuthTokenData | None:
        """Force a DB read by dropping any cached entry first.

        Use inside an advisory-lock critical section where the whole
        point of the read is to observe a peer worker's just-persisted
        token. A stale cache would mask the peer's write and cause this
        worker to repeat work (e.g. duplicate HTTP refresh that
        overwrites the rotated refresh_token, or a redundant
        client-credentials mint for integrations that bypass
        ``refresh_token`` entirely). All other callers should use
        ``load_token``.
        """
        self._token_cache.pop((user_id, integration), None)
        return await self.load_token(user_id, integration)

    async def delete_token(
        self,
        user_id: str,
        integration: str,
    ) -> bool:
        """Remove a stored token row."""
        from backend.app.models import OAuthToken

        logger.info(
            "credential.read action=revoke user=%s integration=%s",
            user_id,
            integration,
        )
        async with db_session_async() as db:
            row = (
                await db.execute(
                    select(OAuthToken).where(
                        OAuthToken.user_id == user_id,
                        OAuthToken.integration == integration,
                    )
                )
            ).scalar_one_or_none()

            if row is None:
                self._token_cache.pop((user_id, integration), None)
                return False

            await db.delete(row)
            await db.commit()
            self._token_cache.pop((user_id, integration), None)
            return True

    async def is_connected(self, user_id: str, integration: str) -> bool:
        """Check if a valid (non-expired) token exists for this user/integration.

        Returns True when a token row exists and is either not expired or
        has a refresh token that could renew it. Returns False when no
        token exists or the token is expired without a refresh token.
        """
        token = await self.load_token(user_id, integration)
        if token is None:
            return False
        if not token.is_expired():
            return True
        # Expired but has a refresh token: still considered "connected"
        # because get_valid_token() will refresh it on next use.
        return bool(token.refresh_token)

    @asynccontextmanager
    async def refresh_lock(
        self,
        user_id: str,
        integration: str,
    ) -> AsyncIterator[bool]:
        """Hold the OAuth refresh advisory lock for the duration of the block.

        Yields True when the lock is held; False when acquisition timed out
        (the caller should treat that as a transient refresh failure and let
        the next attempt try again). The lock releases automatically on exit.

        Integrations that mint their own bearers outside the standard
        ``refresh_token`` flow (notably client-credentials grants with no
        OAuth refresh token) wrap their mint-and-save sequence in this
        lock so concurrent mints serialize through the same primitive the
        standard refresh path uses. The lock is keyed on
        ``(user_id, integration)``, identical to ``refresh_token`` above.
        """
        # Same connection-pinning constraint as ``refresh_token``: the lock
        # must live on a dedicated ``AsyncConnection`` so the release runs
        # on the connection that acquired it.
        lock_conn = await get_async_engine().connect()
        lock_key = _refresh_lock_key(user_id, integration)
        try:
            acquired = await _try_acquire_advisory_lock_async(lock_conn, lock_key)
            try:
                yield acquired
            finally:
                if acquired:
                    try:
                        await lock_conn.execute(
                            text("SELECT pg_advisory_unlock(hashtext(:k))"),
                            {"k": lock_key},
                        )
                        await lock_conn.commit()
                    except Exception:
                        logger.exception(
                            "Failed to release OAuth refresh lock: user=%s integration=%s",
                            user_id,
                            integration,
                        )
        finally:
            await lock_conn.close()

    def build_rejected_token_refresher(
        self,
        user_id: str,
        integration: str,
    ) -> Callable[[str], Awaitable[str | None]]:
        """Return the mid-call refresh hook a provider service calls after a 401.

        The hook takes the access token the provider just rejected and
        returns a fresh one, or None when there is nothing to refresh with. It
        goes through ``refresh_rejected_token``, so the refresh is locked
        against peers and persisted, a dead grant retires the token, notifies
        the user once, and raises ``ReconnectRequired``, and a contended lock
        raises ``RefreshLockContended``.

        A service builds one hook per turn and its parallel tool calls share
        it. Calls rejected with the same token while a refresh for it is in
        flight await that refresh and get its outcome, rather than queueing on
        the advisory lock: a slow token endpoint can hold the lock past
        ``_LOCK_MAX_WAIT_S``, and a sibling that gave up would report "retry
        shortly" while its own turn was the one refreshing. Other processes
        still meet at the advisory lock.
        """
        in_flight: dict[str, asyncio.Future[OAuthTokenData | None]] = {}

        def _forget(rejected: str, task: asyncio.Future[OAuthTokenData | None]) -> None:
            if in_flight.get(rejected) is task:
                del in_flight[rejected]
            # Every waiter may have been cancelled; retrieve the outcome so an
            # exception is never reported as unretrieved.
            if not task.cancelled():
                task.exception()

        async def _refresh(rejected_access_token: str) -> str | None:
            task = in_flight.get(rejected_access_token)
            if task is None:
                task = asyncio.ensure_future(
                    self.refresh_rejected_token(user_id, integration, rejected_access_token)
                )
                in_flight[rejected_access_token] = task
                task.add_done_callback(functools.partial(_forget, rejected_access_token))
            # Shielded so one cancelled caller does not cancel the refresh its
            # siblings are waiting on.
            refreshed = await asyncio.shield(task)
            return refreshed.access_token if refreshed else None

        return _refresh

    # -- Token refresh with error classification --------------------------------

    @staticmethod
    def _is_permanent_refresh_failure(error: Exception) -> bool:
        """Return True when the refresh error is permanent (user must re-auth).

        Permanent errors (e.g. ``invalid_grant`` from a revoked token) mean
        re-authentication is required. Transient errors (network timeouts,
        provider 5xx) leave the token intact for a later retry. A token
        endpoint refusal counts only under ``is_dead_grant_response``, the one
        rule every integration shares.
        """
        if isinstance(error, PermanentRefreshError):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            is_permanent = is_dead_grant_response(error.response)
            logger.debug(
                "OAuth error classification: status=%s permanent=%s",
                error.response.status_code,
                is_permanent,
            )
            return is_permanent
        return False

    async def refresh_token(
        self,
        user_id: str,
        integration: str,
        *,
        rejected_access_token: str = "",
        raise_on_contention: bool = False,
    ) -> OAuthTokenData | None:
        """Refresh an expired OAuth token via the provider's token endpoint.

        ``raise_on_contention`` raises ``RefreshLockContended`` instead of
        returning None when a peer holds the lock past the bounded wait, so a
        caller can tell that transient case from "nothing to refresh".

        Returns the updated token data on success, or None if no token or
        refresh token exists, including when the user disconnected while the
        refresh POST was in flight. When the user reconnected during the POST,
        the refreshed tokens belong to the replaced grant: they are dropped and
        the new connection is returned unchanged. Raises on HTTP errors so the caller can
        classify them via ``_is_permanent_refresh_failure``. A permanent
        failure has already deleted the token, under the lock, by then.

        ``rejected_access_token`` is the access token a provider API just
        answered 401 to. When given, the refresh is skipped only if the stored
        access token has already moved past it (a peer refreshed), rather than
        whenever the stored token is unexpired: a token the provider rejects
        before its expiry still needs refreshing.

        A session-scoped Postgres advisory lock serializes concurrent
        refreshes for the same (user, integration). Without it, two workers
        racing a 401 both POST ``refresh_token`` to the provider. Providers
        that rotate the refresh token (Google, QuickBooks) may invalidate
        the other worker's newly-issued token, or the losing worker may
        overwrite the winner's rotated refresh_token in the DB. After
        acquiring the lock we re-load the token and skip the HTTP call
        entirely if another worker already refreshed it.
        """
        logger.info(
            "credential.read action=refresh user=%s integration=%s",
            user_id,
            integration,
        )
        # The advisory lock is session-scoped (lives on the underlying
        # PG connection until released or the connection closes). We
        # must hold it on a dedicated ``AsyncConnection`` rather than
        # an ``AsyncSession``: ``AsyncSession.commit()`` returns the
        # connection to the pool, where a peer coroutine can check it
        # out and call ``pg_try_advisory_lock`` on it (locks are
        # reentrant per PG session), letting both callers enter the
        # critical section. The advisory lock must stay pinned to the
        # same physical connection from acquire through unlock; only
        # ``AsyncConnection`` provides that pinning.
        lock_conn = await get_async_engine().connect()
        lock_key = _refresh_lock_key(user_id, integration)
        try:
            if not await _try_acquire_advisory_lock_async(lock_conn, lock_key):
                logger.warning(
                    "OAuth refresh lock contended for >%.1fs, skipping refresh: "
                    "user=%s integration=%s",
                    _LOCK_MAX_WAIT_S,
                    user_id,
                    integration,
                )
                if raise_on_contention:
                    raise RefreshLockContended(
                        f"OAuth refresh lock for {integration} held by a peer past the wait"
                    )
                return None
            try:
                # Bypass the cache: the post-lock reload exists to detect
                # a peer worker's just-persisted refresh, which an in-memory
                # cache from before the lock would mask.
                token = await self.load_token_uncached(user_id, integration)
                if not token or not token.refresh_token:
                    logger.debug(
                        "Cannot refresh token (missing): user=%s integration=%s "
                        "has_token=%s has_refresh=%s",
                        user_id,
                        integration,
                        token is not None,
                        bool(token and token.refresh_token),
                    )
                    return None

                # If a peer worker already refreshed under the lock, the
                # reloaded token will carry a future expires_at. Skip the
                # redundant HTTP call. ``expires_at > 0`` guards against
                # providers that don't return an expiry (expires_at == 0,
                # treated as non-expiring), where an explicit
                # ``refresh_token`` call still needs to hit the provider.
                # After a mid-call 401 the stored token may be unexpired yet
                # rejected, so there a peer refresh shows as a changed
                # access token instead.
                if rejected_access_token:
                    peer_refreshed = token.access_token != rejected_access_token
                else:
                    peer_refreshed = token.expires_at > 0 and not token.is_expired()
                if peer_refreshed:
                    logger.info(
                        "Token already refreshed by another worker: user=%s integration=%s",
                        user_id,
                        integration,
                    )
                    return token

                grant = self._refresh_grants.get(integration)
                if grant is None:
                    config = get_oauth_config(integration)
                    if config is None:
                        logger.debug(
                            "Cannot refresh token (no config): user=%s integration=%s",
                            user_id,
                            integration,
                        )
                        return None
                    grant = functools.partial(self._post_refresh_grant, config)

                # ``token`` is updated in place with the response, so keep the
                # refresh token this POST sends for the check after it.
                posted_refresh_token = token.refresh_token
                try:
                    data = await grant(posted_refresh_token)
                except Exception as exc:
                    if self._is_permanent_refresh_failure(exc):
                        # Retire the dead grant before releasing the lock. A
                        # peer waiting on it then reloads no token and stops,
                        # instead of posting the dead refresh token again and
                        # sending the user a second reconnect notice. The
                        # caller's ``handle_permanent_refresh_failure`` still
                        # notifies, once, and retries the delete if this one
                        # failed.
                        try:
                            await self.delete_token(user_id, integration)
                        except Exception:
                            logger.exception(
                                "Could not retire dead OAuth token under the lock: "
                                "user=%s integration=%s",
                                user_id,
                                integration,
                            )
                    raise

                token.access_token = data["access_token"]
                if "refresh_token" in data:
                    token.refresh_token = data["refresh_token"]

                # Compute absolute expiry from the provider's response.
                # RFC 6749 Section 5.1: expires_in is RECOMMENDED, not REQUIRED.
                # Some providers return an absolute expires_at timestamp instead.
                # When neither is present the token is treated as non-expiring
                # (expires_at stays 0, and is_expired() returns False).
                if data.get("expires_in"):
                    token.expires_at = time.time() + int(data["expires_in"])
                elif data.get("expires_at"):
                    token.expires_at = float(data["expires_at"])
                else:
                    token.expires_at = 0.0

                # ``extra`` is integration metadata, not token state, and the
                # lock does not cover its writers (AppFolio records customer
                # IDs mid-turn). Re-read it after the POST so a write that
                # landed during the request is not overwritten by the copy
                # loaded before it.
                current = await self.load_token_uncached(user_id, integration)
                if current is None:
                    # The user disconnected while the POST was in flight.
                    # Saving would upsert the credential they just removed, so
                    # drop the refreshed tokens and report nothing to refresh:
                    # callers then read the connection as gone. No notice, as
                    # the user chose this.
                    logger.info(
                        "Token disconnected during refresh, discarding result: "
                        "user=%s integration=%s",
                        user_id,
                        integration,
                    )
                    return None
                if current.refresh_token != posted_refresh_token:
                    # The row holds a different grant: the user reconnected
                    # during the POST (peers cannot rotate the refresh token
                    # while this lock is held, and every connect stores a new
                    # one). Saving would put the old grant, and with it the
                    # old QuickBooks company or AppFolio fingerprint, over the
                    # new connection. Keep the new one and hand it back: it is
                    # live, so callers use it as they would a peer's refresh.
                    logger.info(
                        "Token reconnected during refresh, discarding result: "
                        "user=%s integration=%s",
                        user_id,
                        integration,
                    )
                    return current
                token.extra = current.extra
                await self.save_token(user_id, integration, token)
                logger.info(
                    "Refreshed OAuth token: user=%s integration=%s",
                    user_id,
                    integration,
                )
                return token
            finally:
                try:
                    await lock_conn.execute(
                        text("SELECT pg_advisory_unlock(hashtext(:k))"),
                        {"k": lock_key},
                    )
                    await lock_conn.commit()
                except Exception:
                    logger.exception(
                        "Failed to release OAuth refresh lock: user=%s integration=%s",
                        user_id,
                        integration,
                    )
        finally:
            await lock_conn.close()

    async def _post_refresh_grant(self, config: OAuthConfig, refresh_token: str) -> dict[str, Any]:
        """POST the standard refresh grant; raises ``HTTPStatusError`` on a refusal."""
        logger.debug(
            "Attempting token refresh: integration=%s token_url=%s",
            config.integration,
            config.token_url,
        )
        resp = await self._get_http().post(
            config.token_url,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            auth=(config.client_id, config.client_secret),
        )
        resp.raise_for_status()
        return resp.json()

    async def get_valid_token(
        self,
        user_id: str,
        integration: str,
    ) -> OAuthTokenData | None:
        """Return a valid token, refreshing automatically if expired.

        On permanent refresh failure (e.g. revoked grant), deletes the
        stale token and sends a re-auth notification to the user.
        On transient failure (e.g. network error), keeps the token for
        a later retry and returns None.
        """
        token = await self.load_token(user_id, integration)
        if not token:
            logger.debug("No token found: user=%s integration=%s", user_id, integration)
            return None

        if not token.is_expired():
            logger.debug(
                "Token valid (not expired): user=%s integration=%s expires_at=%s",
                user_id,
                integration,
                token.expires_at,
            )
            return token

        logger.info(
            "Token expired: user=%s integration=%s expires_at=%s",
            user_id,
            integration,
            token.expires_at,
        )

        if not token.refresh_token:
            logger.warning(
                "Token expired with no refresh token: user=%s integration=%s",
                user_id,
                integration,
            )
            return None

        try:
            return await self.refresh_token(user_id, integration)
        except Exception as exc:
            logger.warning(
                "Token refresh failed: user=%s integration=%s error=%s",
                user_id,
                integration,
                exc,
            )
            if not await self.handle_permanent_refresh_failure(user_id, integration, exc):
                logger.info(
                    "Transient OAuth failure, keeping token for retry: user=%s integration=%s",
                    user_id,
                    integration,
                )
            return None

    async def refresh_rejected_token(
        self,
        user_id: str,
        integration: str,
        rejected_access_token: str,
    ) -> OAuthTokenData | None:
        """Refresh after a provider API rejected *rejected_access_token* mid-call.

        For provider services that retry a request on 401. Runs the refresh
        under the same lock as every other path and, on a permanent failure
        (``invalid_grant`` and friends), retires the token and notifies the
        user through ``handle_permanent_refresh_failure``, then raises
        ``ReconnectRequired``. A transient failure (provider 5xx, network
        error) propagates unchanged and leaves the token for a later retry.
        Also raises ``ReconnectRequired`` when the token row is gone, as it is
        for every later call in a turn whose first call retired it. Raises
        ``RefreshLockContended`` when a peer held the lock past the wait,
        which is transient. Returns None when there is nothing to refresh
        with (no refresh token, no OAuth config), leaving the caller's
        original 401 to stand. When the user reconnected while the refresh ran,
        returns the new connection, so the caller retries with its token.
        """
        friendly = _display_name(integration)
        reconnect_message = (
            f"The {friendly} connection has expired or was revoked. "
            f"{reconnect_instruction(integration)}"
        )
        try:
            refreshed = await self.refresh_token(
                user_id,
                integration,
                rejected_access_token=rejected_access_token,
                raise_on_contention=True,
            )
        except Exception as exc:
            if await self.handle_permanent_refresh_failure(user_id, integration, exc):
                raise ReconnectRequired(integration, reconnect_message) from exc
            raise
        if refreshed is None and await self.load_token_uncached(user_id, integration) is None:
            raise ReconnectRequired(integration, reconnect_message)
        return refreshed

    async def handle_permanent_refresh_failure(
        self,
        user_id: str,
        integration: str,
        error: Exception,
    ) -> bool:
        """Retire the token when *error* means the user has to reconnect.

        Deletes the stored token (``refresh_token`` normally has already, under
        the refresh lock) and tells the user, then returns True. Returns
        False for a transient error, leaving the token for a later retry. Shared
        by the inline path and the background sweep: a sweep that skipped this
        kept a dead token due for refresh and retried it on every tick, forever.

        The notice never depends on this delete. By the time it runs the token
        is usually gone already, and once it is gone nothing refreshes it again,
        so a delete that raised here would leave the user never told.
        """
        if not self._is_permanent_refresh_failure(error):
            return False
        logger.warning(
            "Permanent OAuth failure, deleting token: user=%s integration=%s",
            user_id,
            integration,
        )
        try:
            await self.delete_token(user_id, integration)
        except Exception:
            logger.exception(
                "Could not delete dead OAuth token, notifying anyway: user=%s integration=%s",
                user_id,
                integration,
            )
        await self._notify_reauth_needed(user_id, integration)
        return True

    async def _notify_reauth_needed(
        self,
        user_id: str,
        integration: str,
    ) -> None:
        """Best-effort notification that an OAuth integration has disconnected.

        Looks up the user's active channel route and sends a message via
        the bus. Failures are logged and swallowed so token cleanup is
        never blocked by a notification error.
        """
        try:
            from backend.app.bus import OutboundMessage, message_bus
            from backend.app.models import ChannelRoute

            async with db_session_async() as db:
                route = (
                    (
                        await db.execute(
                            select(ChannelRoute).where(
                                ChannelRoute.user_id == user_id,
                                ChannelRoute.enabled.is_(True),
                            )
                        )
                    )
                    .scalars()
                    .first()
                )

            if route is None:
                logger.debug(
                    "No active channel route for reauth notification: user=%s",
                    user_id,
                )
                return

            friendly = _display_name(integration)
            text = (
                f"Your {friendly} connection has expired. "
                "Please reconnect it in Settings > Integrations."
            )

            await message_bus.publish_outbound(
                OutboundMessage(
                    channel=route.channel,
                    chat_id=route.channel_identifier,
                    content=text,
                )
            )
        except Exception:
            logger.warning(
                "Failed to notify user about disconnected integration: user=%s integration=%s",
                user_id,
                integration,
            )

    # -- State management helpers ----------------------------------------------

    def get_pending_state_integration(self, state: str) -> str | None:
        """Return the integration name for a pending state, or None."""
        pending = self._pending_states.get(state)
        if pending is None or time.time() > pending.expires_at:
            return None
        return pending.integration

    def get_pending_state_source(self, state: str) -> str:
        """Return the source ("web" or "chat") for a pending state."""
        pending = self._pending_states.get(state)
        if pending is None or time.time() > pending.expires_at:
            return "web"
        return pending.source

    def _cleanup_expired_states(self) -> None:
        """Remove expired pending states."""
        now = time.time()
        expired = [k for k, v in self._pending_states.items() if now > v.expires_at]
        for k in expired:
            del self._pending_states[k]


# Module-level singleton.
oauth_service = OAuthService()


# ---------------------------------------------------------------------------
# Background refresh scheduler
# ---------------------------------------------------------------------------

# How often the background sweep runs.
_REFRESH_SWEEP_INTERVAL_SECONDS = 120.0

# Random jitter applied to each inter-sweep sleep so that schedulers
# running in N uvicorn workers do not synchronize and stampede the DB
# / advisory locks at the same instant. Bounded so the effective
# interval stays in the [105s, 135s] range.
_REFRESH_SWEEP_JITTER_SECONDS = 15.0

# How far ahead the sweep looks for tokens to refresh. Slightly larger
# than ``_EXPIRY_BUFFER_SECONDS`` (300s) so the sweep catches tokens
# before ``OAuthTokenData.is_expired()`` flips True and the inline
# refresh in ``get_valid_token`` would fire on the user-facing path.
_REFRESH_LOOKAHEAD_SECONDS = 360.0

# Only refresh in the background for users who have been active in the
# last N days. Why: refresh tokens for some providers (notably
# QuickBooks, 100 day inactivity expiry) are designed to expire when
# a user stops using the integration, forcing reconnection and a fresh
# user-consent confirmation. Background refresh would silently bypass
# that signal for dormant accounts. Activity is "received an inbound
# message via any channel route" -- this captures iMessage, Telegram,
# webchat, etc. Inactive users still get the inline refresh path on
# their next interaction.
_REFRESH_ACTIVITY_WINDOW_DAYS = 14


class OAuthRefreshScheduler:
    """Periodically refresh OAuth tokens before they expire.

    Without this, every user message that arrives during the 5 minute
    pre-expiry window pays the cost of an inline HTTP refresh
    (~150ms). Sweeping in the background pulls that cost off the
    critical path: by the time the user texts, the token is already
    fresh and ``get_valid_token`` returns immediately.

    Failures never stop the sweep from processing the rest. A permanent
    failure (revoked grant, rejected client) retires the token and asks the
    user to reconnect, as the inline path does; a transient one is logged
    and retried next tick. Inline refresh in
    ``get_valid_token`` remains the safety net for tokens the sweep
    missed (e.g. process just started, sweep hasn't run yet).
    """

    def __init__(self, service: OAuthService) -> None:
        self._service = service
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the sweep loop (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop (sync test harness, etc.): skip silently.
            return
        self._task = loop.create_task(self._run())
        logger.info(
            "OAuth refresh sweep started (interval=%.0fs lookahead=%.0fs)",
            _REFRESH_SWEEP_INTERVAL_SECONDS,
            _REFRESH_LOOKAHEAD_SECONDS,
        )

    def stop(self) -> None:
        """Cancel the sweep loop."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
            logger.info("OAuth refresh sweep stopped")

    async def _run(self) -> None:
        while True:
            try:
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("OAuth refresh sweep failed")
            jitter = random.uniform(-_REFRESH_SWEEP_JITTER_SECONDS, _REFRESH_SWEEP_JITTER_SECONDS)
            await asyncio.sleep(_REFRESH_SWEEP_INTERVAL_SECONDS + jitter)

    async def sweep(self) -> int:
        """Run one sweep. Returns the number of tokens that were refreshed.

        Only refreshes tokens for users who have been active in the last
        ``_REFRESH_ACTIVITY_WINDOW_DAYS`` days, so dormant users still
        hit the natural provider-side inactivity expiry on their refresh
        tokens (e.g. QuickBooks 100 day rule). Inactive users fall back
        to the inline refresh path the next time they interact.

        Exposed publicly so tests can drive a single tick without
        spinning up the background task.
        """
        import datetime as _dt

        from backend.app.models import ChannelRoute, OAuthToken

        cutoff = time.time() + _REFRESH_LOOKAHEAD_SECONDS
        activity_cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(
            days=_REFRESH_ACTIVITY_WINDOW_DAYS
        )
        active_user_ids = (
            select(ChannelRoute.user_id)
            .where(ChannelRoute.last_inbound_at.is_not(None))
            .where(ChannelRoute.last_inbound_at > activity_cutoff)
        )
        async with db_session_async() as db:
            rows = (
                await db.execute(
                    select(OAuthToken.user_id, OAuthToken.integration)
                    .where(OAuthToken.expires_at > 0)
                    .where(OAuthToken.expires_at < cutoff)
                    .where(OAuthToken.refresh_token != "")
                    .where(OAuthToken.user_id.in_(active_user_ids))
                )
            ).all()
        if not rows:
            return 0
        logger.info("OAuth refresh sweep: %d token(s) due for refresh", len(rows))
        refreshed = 0
        for user_id, integration in rows:
            try:
                result = await self._service.refresh_token(user_id, integration)
            except Exception as exc:
                try:
                    retired = await self._service.handle_permanent_refresh_failure(
                        user_id, integration, exc
                    )
                except Exception:
                    logger.exception(
                        "Could not retire OAuth token after permanent failure: "
                        "user=%s integration=%s",
                        user_id,
                        integration,
                    )
                    continue
                if not retired:
                    logger.error(
                        "Background OAuth refresh failed: user=%s integration=%s",
                        user_id,
                        integration,
                        exc_info=exc,
                    )
                continue
            if result is not None:
                refreshed += 1
        return refreshed


oauth_refresh_scheduler = OAuthRefreshScheduler(oauth_service)


# ---------------------------------------------------------------------------
# Integration-specific config builders
# ---------------------------------------------------------------------------

# QuickBooks OAuth 2.0 endpoint fallbacks (used when the Intuit discovery
# document is unavailable or stale).
_QBO_AUTHORIZE_URL_FALLBACK = "https://appcenter.intuit.com/connect/oauth2"
_QBO_TOKEN_URL_FALLBACK = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
QBO_SCOPES = ["com.intuit.quickbooks.accounting"]

# Google Calendar OAuth 2.0 endpoints
GOOGLE_CALENDAR_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_CALENDAR_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# Google Drive OAuth 2.0 endpoints. ``drive.file`` is the narrow per-app
# scope: the integration only sees files it created itself, not the user's
# entire Drive. Drive shares Google's OAuth endpoints with Calendar.
GOOGLE_DRIVE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_DRIVE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
]

# Gmail OAuth 2.0 endpoints. Separate Google OAuth client from Calendar /
# Drive so the Gmail scope set can be approved independently and so users
# who only want one of the three never see the others on the consent
# screen. ``gmail.readonly`` covers search + fetch; ``gmail.send`` covers
# composing new messages and threaded replies. We deliberately avoid the
# broader ``gmail.modify`` so the integration cannot delete, archive, or
# label the user's mail.
GMAIL_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

# CompanyCam OAuth 2.0 endpoints
COMPANYCAM_AUTHORIZE_URL = "https://app.companycam.com/oauth/authorize"
COMPANYCAM_TOKEN_URL = "https://app.companycam.com/oauth/token"
COMPANYCAM_SCOPES = ["read", "write", "destroy"]

# Registry of all supported OAuth integrations.
_OAUTH_INTEGRATIONS = (
    "quickbooks",
    "google_calendar",
    "google_drive",
    "gmail",
    "companycam",
)


def get_quickbooks_oauth_config() -> OAuthConfig | None:
    """Build the QuickBooks OAuth config from settings.

    Endpoint URLs are resolved from the Intuit OpenID Connect discovery
    document when available (populated by ``warm_intuit_discovery()`` at
    startup). Falls back to hardcoded URLs if the discovery document has
    not been fetched or is stale.
    """
    authorize_url, token_url = _get_intuit_endpoints()
    config = OAuthConfig(
        integration="quickbooks",
        client_id=settings.quickbooks_client_id,
        client_secret=settings.quickbooks_client_secret,
        authorize_url=authorize_url,
        token_url=token_url,
        scopes=QBO_SCOPES,
    )
    return config if config.is_configured else None


# Google picks the browser's signed-in account unless asked to show the
# chooser, so a user with two accounts silently connects the wrong one and
# every saved calendar or file from the other account reads as missing.
# ``consent`` stays because a refresh token only comes back with it.
_GOOGLE_PROMPT = "consent select_account"


def get_google_calendar_oauth_config() -> OAuthConfig | None:
    """Build the Google Calendar OAuth config from settings."""
    config = OAuthConfig(
        integration="google_calendar",
        client_id=settings.google_calendar_client_id,
        client_secret=settings.google_calendar_client_secret,
        authorize_url=GOOGLE_CALENDAR_AUTHORIZE_URL,
        token_url=GOOGLE_CALENDAR_TOKEN_URL,
        scopes=GOOGLE_CALENDAR_SCOPES,
        use_pkce=False,
        extra_auth_params={"access_type": "offline", "prompt": _GOOGLE_PROMPT},
    )
    return config if config.is_configured else None


def get_google_drive_oauth_config() -> OAuthConfig | None:
    """Build the Google Drive OAuth config from settings."""
    config = OAuthConfig(
        integration="google_drive",
        client_id=settings.google_drive_client_id,
        client_secret=settings.google_drive_client_secret,
        authorize_url=GOOGLE_DRIVE_AUTHORIZE_URL,
        token_url=GOOGLE_DRIVE_TOKEN_URL,
        scopes=GOOGLE_DRIVE_SCOPES,
        use_pkce=False,
        extra_auth_params={"access_type": "offline", "prompt": _GOOGLE_PROMPT},
    )
    return config if config.is_configured else None


def get_gmail_oauth_config() -> OAuthConfig | None:
    """Build the Gmail OAuth config from settings."""
    config = OAuthConfig(
        integration="gmail",
        client_id=settings.gmail_client_id,
        client_secret=settings.gmail_client_secret,
        authorize_url=GMAIL_AUTHORIZE_URL,
        token_url=GMAIL_TOKEN_URL,
        scopes=GMAIL_SCOPES,
        use_pkce=False,
        extra_auth_params={"access_type": "offline", "prompt": _GOOGLE_PROMPT},
    )
    return config if config.is_configured else None


def get_companycam_oauth_config() -> OAuthConfig | None:
    """Build the CompanyCam OAuth config from settings."""
    config = OAuthConfig(
        integration="companycam",
        client_id=settings.companycam_client_id,
        client_secret=settings.companycam_client_secret,
        authorize_url=COMPANYCAM_AUTHORIZE_URL,
        token_url=COMPANYCAM_TOKEN_URL,
        scopes=COMPANYCAM_SCOPES,
        use_pkce=False,
    )
    return config if config.is_configured else None


def get_oauth_config(integration: str) -> OAuthConfig | None:
    """Return the OAuth config for the named integration, or None."""
    if integration == "quickbooks":
        return get_quickbooks_oauth_config()
    if integration == "google_calendar":
        return get_google_calendar_oauth_config()
    if integration == "google_drive":
        return get_google_drive_oauth_config()
    if integration == "gmail":
        return get_gmail_oauth_config()
    if integration == "companycam":
        return get_companycam_oauth_config()
    return None


def list_oauth_integrations() -> tuple[str, ...]:
    """Return names of all supported OAuth integrations."""
    return _OAUTH_INTEGRATIONS
