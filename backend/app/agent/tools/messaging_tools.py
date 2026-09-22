from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from backend.app.agent.approval import ApprovalPolicy, PermissionLevel
from backend.app.agent.tools.base import Tool, ToolErrorKind, ToolResult, ToolTags
from backend.app.agent.tools.names import ToolName

if TYPE_CHECKING:
    from backend.app.agent.tools.registry import ToolContext
    from backend.app.bus import OutboundMessage


class SendMediaReplyParams(BaseModel):
    """Parameters for the send_media_reply tool."""

    message: str = Field(description="The message text")
    media_url: str = Field(description="URL of the media to attach")


def media_url_error(media_url: str) -> str | None:
    """Why ``send_media_reply`` would refuse *media_url*, or None if it would send.

    Shared by the tool body and its ``precheck`` so the model comparison
    report sees the refusal production applies: an empty or ``about:blank`` URL
    sends nothing, and counting it as a message to the user reports a write
    that never happens.
    """
    if not media_url or not media_url.strip():
        return "media_url cannot be empty."
    url = media_url.strip()
    is_url = url.startswith("http://") or url.startswith("https://")
    if not is_url and not Path(url).is_file():
        return (
            f"media_url '{url}' is not a valid URL (must start with "
            f"http:// or https://) and does not exist as a local file."
        )
    return None


def _send_media_precheck(args: dict[str, Any]) -> str | None:
    media_url = args.get("media_url")
    return media_url_error(media_url if isinstance(media_url, str) else "")


def create_messaging_tools(
    publish_outbound: Callable[[OutboundMessage], Awaitable[None]],
    channel: str,
    to_address: str,
) -> list[Tool]:
    """Create messaging tools for the agent."""

    async def send_media_reply(message: str, media_url: str) -> ToolResult:
        """Send a reply with a media attachment."""
        from backend.app.bus import OutboundMessage as OMsg

        error = media_url_error(media_url)
        if error is not None:
            return ToolResult(
                content=f"Error: {error}",
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )
        url = media_url.strip()
        outbound = OMsg(channel=channel, chat_id=to_address, content=message, media=[url])
        await publish_outbound(outbound)
        return ToolResult(content="Sent media message")

    return [
        Tool(
            name=ToolName.SEND_MEDIA_REPLY,
            description="Send a reply with a media attachment (e.g., PDF estimate).",
            function=send_media_reply,
            params_model=SendMediaReplyParams,
            tags={ToolTags.SENDS_REPLY},
            usage_hint=("When sending estimates or files, use this to send media to the user."),
            approval_policy=ApprovalPolicy(
                default_level=PermissionLevel.ALWAYS,
                description_builder=lambda args: "Send a file attachment",
            ),
            # Preserve send order if the model batches multiple replies in one
            # turn. The user's channel is the shared resource here.
            concurrency_group="user_outbound",
            precheck=_send_media_precheck,
        ),
    ]


def _messaging_factory(ctx: ToolContext) -> list[Tool]:
    """Factory for messaging tools, used by the registry."""
    assert ctx.publish_outbound is not None
    return create_messaging_tools(
        ctx.publish_outbound, channel=ctx.channel, to_address=ctx.to_address
    )


def _register() -> None:
    from backend.app.agent.tools.registry import SubToolInfo, default_registry

    default_registry.register(
        "messaging",
        _messaging_factory,
        requires_outbound=True,
        display_name="Messaging",
        dashboard_description="Send text and media replies to the user",
        dashboard_always_enabled=True,
        sub_tools=[
            SubToolInfo(
                ToolName.SEND_MEDIA_REPLY,
                "Send replies with media attachments",
                default_permission="always",
                hidden_in_permissions=True,
            ),
        ],
    )


_register()
