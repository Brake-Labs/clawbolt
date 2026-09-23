"""Heartbeat management tools for the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from backend.app.agent.approval import ApprovalPolicy, PermissionLevel
from backend.app.agent.heartbeat import SCHEDULED_TASK_PREFIX
from backend.app.agent.markdown_registry import BudgetExceededError
from backend.app.agent.stores import HeartbeatStore
from backend.app.agent.tools.base import Tool, ToolErrorKind, ToolResult, ToolTags
from backend.app.agent.tools.names import ToolName

if TYPE_CHECKING:
    from backend.app.agent.tools.registry import ToolContext


class GetHeartbeatParams(BaseModel):
    """Parameters for the get_heartbeat tool (no parameters)."""


class UpdateHeartbeatParams(BaseModel):
    """Parameters for the update_heartbeat tool."""

    text: str = Field(description="The full new HEARTBEAT.md markdown.")


def create_heartbeat_tools(user_id: str) -> list[Tool]:
    """Create heartbeat-related tools for the agent."""

    async def get_heartbeat() -> ToolResult:
        """Read the user's heartbeat notes."""
        store = HeartbeatStore(user_id)
        text = await store.read_heartbeat_md_async()
        if not text:
            return ToolResult(content="No heartbeat notes set.")
        return ToolResult(content=text)

    async def update_heartbeat(text: str) -> ToolResult:
        """Update the user's heartbeat notes.

        Reads the current content first so the result shows what changed.
        """
        store = HeartbeatStore(user_id)
        previous = await store.read_heartbeat_md_async()
        try:
            await store.write_heartbeat_md(text)
        except BudgetExceededError as exc:
            return ToolResult(
                content=str(exc),
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )
        if previous:
            return ToolResult(content=f"Heartbeat notes updated.\n\nPrevious content:\n{previous}")
        return ToolResult(content="Heartbeat notes updated (was empty).")

    return [
        Tool(
            name=ToolName.GET_HEARTBEAT,
            tags={ToolTags.READ_ONLY},
            description="Read HEARTBEAT.md, the notes that drive your periodic check-ins.",
            function=get_heartbeat,
            params_model=GetHeartbeatParams,
        ),
        Tool(
            name=ToolName.UPDATE_HEARTBEAT,
            description=(
                "Overwrite HEARTBEAT.md, the notes that drive your periodic check-ins. "
                "It is not a scheduler: a reminder at a clock time ('at 2pm') goes to "
                "calendar_create_event as Proactive Messaging says, never here. Call "
                "get_heartbeat first and pass the whole file: the current items plus "
                "the requested change. Never re-add items missing from the current "
                "file. Write recurring items as windows ('every morning', 'Mondays'), "
                "not clock times."
            ),
            function=update_heartbeat,
            params_model=UpdateHeartbeatParams,
            usage_hint=(
                "update_heartbeat: change only what the user asked for; do not prune "
                "items that look stale. Exception: when the current message starts "
                f"with '{SCHEDULED_TASK_PREFIX}', remove the one-time dated line you "
                "just handled. Recurring items ('every morning', 'Mondays', 'weekly') "
                "always stay."
            ),
            approval_policy=ApprovalPolicy(
                default_level=PermissionLevel.ALWAYS,
                description_builder=lambda args: "Update heartbeat notes",
            ),
            # Match the per-path key used by workspace_tools for HEARTBEAT.md
            # so a heartbeat update and a workspace write to the same file
            # serialize against each other.
            concurrency_group="workspace_path:HEARTBEAT.md",
        ),
    ]


def _heartbeat_factory(ctx: ToolContext) -> list[Tool]:
    """Factory for heartbeat tools, used by the registry."""
    return create_heartbeat_tools(ctx.user.id)


def _register() -> None:
    from backend.app.agent.tools.registry import SubToolInfo, default_registry

    default_registry.register(
        "heartbeat",
        _heartbeat_factory,
        core=True,
        summary="View and edit heartbeat notes",
        display_name="Heartbeat",
        dashboard_description="View and edit heartbeat notes",
        dashboard_always_enabled=True,
        sub_tools=[
            SubToolInfo(ToolName.GET_HEARTBEAT, "Read heartbeat notes"),
            SubToolInfo(ToolName.UPDATE_HEARTBEAT, "Update heartbeat notes"),
        ],
    )


_register()
