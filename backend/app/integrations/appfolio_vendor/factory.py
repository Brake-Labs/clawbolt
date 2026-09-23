"""AppFolio Vendor Portal tool registration and factories.

The specialist includes work-order reads and status updates, notes with photos,
and invoices. Authentication status gates the tools. Users enter the single-use
magic link on the Integrations page, never in chat history.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.app.agent.tools.base import Tool
from backend.app.agent.tools.names import ToolName
from backend.app.config import settings
from backend.app.integrations.appfolio_vendor.auth import (
    INTEGRATION_NAME,
    load_credential,
    save_customer_ids,
)
from backend.app.integrations.appfolio_vendor.invoices import build_invoice_tools
from backend.app.integrations.appfolio_vendor.notes import build_note_tools
from backend.app.integrations.appfolio_vendor.service import build_service
from backend.app.integrations.appfolio_vendor.work_order_writes import (
    build_work_order_write_tools,
)
from backend.app.integrations.appfolio_vendor.work_orders import build_work_order_tools
from backend.app.services.oauth import oauth_service

if TYPE_CHECKING:
    from backend.app.agent.tools.registry import ToolContext

logger = logging.getLogger(__name__)


_DATA_FACTORY = "appfolio_vendor"


async def _appfolio_vendor_factory(ctx: ToolContext) -> list[Tool]:
    """Assemble the AppFolio data tools for an authenticated user.

    Callers should not invoke this when the user has no credential; the
    registry guards via ``_appfolio_vendor_auth_check`` and skips factory
    creation in that case. The defensive ``return []`` covers the rare
    race where the credential disappears between auth check and create
    (e.g. the user disconnected mid-turn). We log a warning so the race
    is observable rather than silent.
    """
    cred = await load_credential(ctx.user.id)
    if cred is None or not cred.jwt:
        logger.warning(
            "AppFolio credential missing during factory creation for user %s "
            "despite passing auth_check; user may have disconnected mid-turn",
            ctx.user.id,
        )
        return []

    user_id = ctx.user.id

    async def _persist_customer_ids(customer_ids: list[str]) -> None:
        # The OAuth2 exchange does not return customer IDs, so the service
        # discovers them on the first write and hands them here. Persisting
        # makes it a one-time cost instead of a discovery round-trip on
        # every turn. Only the IDs are written: the tokens belong to the
        # shared refresh, and this turn's copy of them may be stale.
        await save_customer_ids(user_id, customer_ids)

    service = build_service(
        cred,
        api_base=settings.appfolio_vendor_api_base,
        # A 401 refreshes through the shared, locked OAuth path: one POST
        # across parallel calls and workers, and a dead grant retires the
        # credential and notifies the user once.
        refresh_rejected_jwt=oauth_service.build_rejected_token_refresher(
            user_id, INTEGRATION_NAME
        ),
        on_customer_ids_resolved=_persist_customer_ids,
    )
    tools: list[Tool] = []
    tools.extend(build_work_order_tools(service))
    tools.extend(build_work_order_write_tools(service))
    tools.extend(build_note_tools(service, ctx))
    tools.extend(build_invoice_tools(service, ctx))
    return tools


async def _appfolio_vendor_auth_check(ctx: ToolContext) -> str | None:
    """Return ``None`` when the user has a usable AppFolio credential.

    When no credential is on file, returns a reason string so the
    registry surfaces ``appfolio_vendor`` under "Not connected" in the
    LLM's capability list. The reason routes the user to the web app: the
    single-use magic link must never be pasted into chat (issue #1337).
    """
    cred = await load_credential(ctx.user.id)
    if cred is not None and cred.jwt:
        return None
    return (
        "AppFolio Vendor Portal is not connected. The user connects it in the"
        " Clawbolt web app on the Integrations page, where they paste the magic"
        " link from their AppFolio sign-in email. Do not accept the magic link"
        " over chat; direct the user to the web app instead."
    )


def _register() -> None:
    from backend.app.agent.tools.registry import SubToolInfo, default_registry

    default_registry.register(
        _DATA_FACTORY,
        _appfolio_vendor_factory,
        core=False,
        summary=(
            "AppFolio Vendor Portal: view and search work orders, update their "
            "status (e.g. mark complete), read and add notes (with photos), "
            "and create or upload invoices"
        ),
        display_name="AppFolio Vendor Portal",
        dashboard_description=(
            "View work orders, update status, add notes, and create invoices "
            "in AppFolio Vendor Portal"
        ),
        dashboard_group="Integrations",
        dashboard_group_order=2,
        sub_tools=[
            SubToolInfo(
                ToolName.APPFOLIO_LIST_WORK_ORDERS,
                "List AppFolio work orders by status",
                default_permission="always",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_SEARCH_WORK_ORDERS,
                "Search AppFolio work orders by text",
                default_permission="always",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_GET_WORK_ORDER,
                "Get full details for one AppFolio work order",
                default_permission="always",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_UPDATE_WORK_ORDER_STATUS,
                "Update the status code on an AppFolio work order",
                default_permission="ask",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_UNDO_WORK_ORDER_STATUS,
                "Revert a recent AppFolio work order status change",
                default_permission="ask",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_LIST_NOTES,
                "List notes on an AppFolio work order",
                default_permission="always",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_ADD_NOTE,
                "Add a note (with optional photos) to an AppFolio work order",
                default_permission="ask",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_UPDATE_NOTE,
                "Edit an existing AppFolio work-order note",
                default_permission="ask",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_CREATE_INVOICE,
                "Build a line-itemized AppFolio invoice with optional photos",
                default_permission="ask",
            ),
            SubToolInfo(
                ToolName.APPFOLIO_UPLOAD_INVOICE_PDF,
                "Upload a pre-built invoice PDF to AppFolio",
                default_permission="ask",
            ),
        ],
        auth_check=_appfolio_vendor_auth_check,
    )


_register()
