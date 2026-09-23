"""Keep saved calendar configuration in step with the connected Google account.

``calendar_configs`` holds the calendars a user enabled in Settings, by
Google calendar id. Those ids belong to whichever Google account granted the
token at the time. When the user reconnects, possibly with a different
account, or the operator swaps OAuth clients, saved ids the new connection
cannot see keep answering 404 on every read and write. Nothing corrected
that before, so the agent kept acting on calendars it could no longer reach.

``resync_calendar_configs`` runs after every successful connect (registered
as a post-connect hook in ``factory._register``) and reconciles the saved
rows against the live ``calendarList``:

- A saved id the connection can see is kept, with its access role and name
  refreshed and its ``disabled_tools`` untouched.
- A saved id the connection cannot see is deleted. A row's presence is what
  "enabled" means in this model and in the Settings picker, which only lists
  live calendars, so a flagged-but-kept row would be invisible in Settings
  yet still offered to the agent. If the user reconnects the owning account
  later, they re-tick those calendars.
- The account's primary calendar is enabled when the connection is new (no
  saved rows survive) or when stale rows were just dropped, so a reconnect
  never leaves the user with nothing usable. A reconnect that drops nothing
  keeps the user's choice to leave the primary calendar off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import select

from backend.app.database import db_session_async
from backend.app.integrations.calendar.provider import CalendarInfo
from backend.app.integrations.calendar.service import GoogleCalendarService
from backend.app.models import CalendarConfig
from backend.app.services.oauth import OAuthTokenData, ReconnectRequired, oauth_service

logger = logging.getLogger(__name__)

PROVIDER = "google_calendar"

# Google's alias for "this account's primary calendar". A saved row with this
# id follows whichever account is connected, so it is always reachable.
PRIMARY_ALIAS = "primary"


def build_calendar_service(user_id: str, token: OAuthTokenData) -> GoogleCalendarService:
    """Build a Calendar client for *token* that refreshes through the shared OAuth path."""
    return GoogleCalendarService(
        access_token=token.access_token,
        refresh_access_token=oauth_service.build_rejected_token_refresher(user_id, PROVIDER),
    )


def is_calendar_visible(calendar_id: str, live_ids: set[str]) -> bool:
    """Whether a saved calendar id is reachable by the connected account."""
    return calendar_id == PRIMARY_ALIAS or calendar_id in live_ids


@dataclass
class CalendarResyncPlan:
    """What a resync changes. Holds calendar ids, which are PII: never log them."""

    updated: dict[str, CalendarInfo] = field(default_factory=dict)
    removed: list[str] = field(default_factory=list)
    added_primary: CalendarInfo | None = None
    primary_id: str = ""


def plan_calendar_resync(
    saved_ids: list[str], live: list[CalendarInfo]
) -> CalendarResyncPlan | None:
    """Decide how saved calendar ids reconcile with the live calendar list.

    Returns ``None`` when *live* has no primary calendar. Every Google
    account has one, so its absence means the listing is not trustworthy,
    and deleting saved rows against it could wipe a working configuration.
    """
    primary = next((c for c in live if c.primary and c.id), None)
    if primary is None:
        return None
    live_by_id = {c.id: c for c in live if c.id}
    plan = CalendarResyncPlan(primary_id=primary.id)
    for cid in saved_ids:
        if cid == PRIMARY_ALIAS:
            continue
        info = live_by_id.get(cid)
        if info is None:
            plan.removed.append(cid)
        else:
            plan.updated[cid] = info

    kept = [cid for cid in saved_ids if cid not in plan.removed]
    primary_enabled = primary.id in kept or PRIMARY_ALIAS in kept
    if not primary_enabled and (plan.removed or not kept):
        plan.added_primary = primary
    return plan


async def resync_calendar_configs(
    user_id: str, service: GoogleCalendarService
) -> CalendarResyncPlan | None:
    """Reconcile the user's saved calendars with the account *service* is bound to.

    Returns the applied plan, or ``None`` when the live listing could not be
    trusted and nothing was changed. Raises on a Calendar API failure; the
    post-connect hook runner logs it and the connect still succeeds.
    """
    live = await service.list_calendars(show_hidden=True)
    async with db_session_async() as db:
        rows = list(
            (
                await db.execute(
                    select(CalendarConfig)
                    .filter_by(user_id=user_id, provider=PROVIDER)
                    .with_for_update()
                )
            )
            .scalars()
            .all()
        )
        plan = plan_calendar_resync([r.calendar_id for r in rows], live)
        if plan is None:
            logger.warning(
                "Calendar resync skipped, live calendar list has no primary: user=%s live_count=%d",
                user_id,
                len(live),
            )
            return None

        removed = set(plan.removed)
        for row in rows:
            if row.calendar_id in removed:
                await db.delete(row)
                continue
            info = plan.updated.get(row.calendar_id)
            if info is not None:
                row.access_role = info.access_role
                if info.summary:
                    row.display_name = info.summary
            # At most one row may carry the flag: ``_get_primary_calendar_id``
            # reads it with ``scalar_one_or_none``.
            row.is_primary = row.calendar_id == plan.primary_id

        added = plan.added_primary
        if added is not None:
            # No per-tool restrictions: write tools on a read-only role are
            # blocked at tool build time from ``access_role`` alone.
            db.add(
                CalendarConfig(
                    user_id=user_id,
                    provider=PROVIDER,
                    calendar_id=added.id,
                    display_name=added.summary or added.id,
                    disabled_tools="",
                    access_role=added.access_role,
                    is_primary=True,
                )
            )
        await db.commit()

    logger.info(
        "Calendar config resynced after connect: user=%s kept=%d removed=%d added_primary=%s",
        user_id,
        len(rows) - len(plan.removed),
        len(plan.removed),
        plan.added_primary is not None,
    )
    return plan


async def resync_after_connect(user_id: str, token: OAuthTokenData) -> None:
    """Post-connect hook: resync saved calendars against the new connection."""
    if not token.access_token:
        return
    try:
        await resync_calendar_configs(user_id, build_calendar_service(user_id, token))
    except ReconnectRequired:
        # The grant died between connect and resync. The shared refresh has
        # already retired the token and told the user, so there is nothing to
        # resync against and nothing worth a traceback.
        logger.warning("Calendar resync skipped, connection no longer valid: user=%s", user_id)
