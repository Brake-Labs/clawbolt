"""Agent-invoked media tools for vision and deliberate discard decisions.

The media pipeline stages inbound bytes; the agent decides per-photo whether
to analyze, save, route, or discard via tool calls.

``analyze_photo`` runs vision on a staged photo and caches the result
per-handle for the session so re-asking returns the same answer instantly.

``discard_media`` releases staged bytes and is idempotent. Always gated by
``ApprovalPolicy.ASK`` so the user confirms before the bytes are dropped;
the tool description instructs the agent to only call it when the current
turn's text explicitly asks to skip saving.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from backend.app.agent import media_staging
from backend.app.agent.approval import ApprovalPolicy, PermissionLevel
from backend.app.agent.tools.base import Tool, ToolErrorKind, ToolResult, ToolTags
from backend.app.agent.tools.names import ToolName
from backend.app.media.pipeline import run_vision_on_media

if TYPE_CHECKING:
    from backend.app.agent.tools.registry import ToolContext

logger = logging.getLogger(__name__)


class AnalyzePhotoParams(BaseModel):
    """Parameters for the analyze_photo tool."""

    handle: str = Field(
        description="Media handle from the attachment label, e.g. 'media_ab12cd'.",
    )
    context: str = Field(
        default="",
        description="Short context to guide the analysis. Omit to use the current message.",
    )


class DiscardMediaParams(BaseModel):
    """Parameters for the discard_media tool."""

    handle: str = Field(
        description="Media handle to discard.",
    )
    reason: str = Field(
        description="The user's request, quoted (e.g. 'user said \"don\\'t save this one\"').",
    )


def create_media_tools(
    user_id: str,
    turn_text: str,
    analyze_cache: dict[str, str],
) -> list[Tool]:
    """Build the agent-native media tool set bound to this turn's context."""

    async def analyze_photo(handle: str, context: str = "") -> ToolResult:
        cached = analyze_cache.get(handle)
        if cached is not None:
            return ToolResult(content=cached)

        entry = await media_staging.get_by_handle(handle)
        if entry is None:
            return ToolResult(
                content=(
                    f"No staged media found for handle {handle!r}. "
                    "It may have expired or already been discarded."
                ),
                is_error=True,
                error_kind=ToolErrorKind.NOT_FOUND,
            )

        stored_user_id, _original_url, content, mime = entry
        if stored_user_id != user_id:
            return ToolResult(
                content=f"Handle {handle!r} does not belong to the current user.",
                is_error=True,
                error_kind=ToolErrorKind.PERMISSION,
            )

        # Vision only supports images. PDFs and other documents would crash
        # the image compressor or confuse the vision LLM.
        if not mime.startswith("image/"):
            return ToolResult(
                content=(
                    f"Handle {handle!r} is {mime}, not an image. "
                    "analyze_photo only works on photos."
                ),
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )

        # Extend TTL on reference so long agent sessions don't evict mid-turn.
        await media_staging.touch(handle, user_id=user_id)

        effective_context = context or turn_text
        description = await run_vision_on_media(content, mime, effective_context, user_id=user_id)
        analyze_cache[handle] = description
        logger.info("analyze_photo ran vision for %s (chars=%d)", handle, len(description))
        return ToolResult(content=description)

    async def discard_media(handle: str, reason: str) -> ToolResult:
        # Defense in depth: same cross-user ownership check as analyze_photo.
        # Handles are unguessable in practice but we still scope every
        # destructive operation to the current user.
        entry = await media_staging.get_by_handle(handle)
        if entry is not None and entry[0] != user_id:
            return ToolResult(
                content=f"Handle {handle!r} does not belong to the current user.",
                is_error=True,
                error_kind=ToolErrorKind.PERMISSION,
            )

        removed = await media_staging.evict_by_handle(handle)
        if not removed:
            # Idempotent: a second call (or a call after expiry) reports
            # success so the agent does not get stuck retrying.
            return ToolResult(
                content=f"Media {handle!r} is not staged (already discarded or expired)."
            )
        analyze_cache.pop(handle, None)
        logger.info("discard_media evicted %s (reason=%r)", handle, reason)
        return ToolResult(content=f"Discarded {handle} (reason: {reason})")

    return [
        Tool(
            name=ToolName.ANALYZE_PHOTO,
            tags={ToolTags.READ_ONLY},
            description=(
                "Run vision analysis on a staged photo. Default: do not call. Use it "
                "only when the user asked you to look at the image or you need its "
                "contents to help. Pass the handle as shown in the conversation. "
                "Results are cached per handle, so a repeat call is cheap and returns "
                "the same result; a different photo needs a new handle, so the user "
                "must send it again. CompanyCam and AppFolio tools accept the same "
                "handle."
            ),
            function=analyze_photo,
            params_model=AnalyzePhotoParams,
        ),
        Tool(
            name=ToolName.DISCARD_MEDIA,
            description=(
                "Discard a staged photo. Use only when the user's current message "
                "explicitly asks to drop it; the user is asked to confirm. "
                "Idempotent."
            ),
            function=discard_media,
            params_model=DiscardMediaParams,
            approval_policy=ApprovalPolicy(
                default_level=PermissionLevel.ASK,
                description_builder=lambda args: (
                    f"Discard staged media {args.get('handle', '?')}"
                    f" ({args.get('reason', 'no reason given')})"
                ),
            ),
        ),
    ]


def _media_factory(ctx: ToolContext) -> list[Tool]:
    """Factory for agent-native media tools.

    Always returns the two tools so the Anthropic prompt-cache key (which
    includes the tools block) stays stable across text-only and media
    turns. Gating on per-message media presence flipped the tool count
    between turns, busting the cache prefix and rewriting the ~135k-token
    system prompt at ~$0.65 per miss (issue #1170). The runtime tools
    handle missing handles gracefully (analyze_photo returns NOT_FOUND,
    discard_media is idempotent), so always-on is safe.
    """
    # Per-turn analysis cache. Scoped to the factory call so it lives for the
    # duration of the agent loop for this message.
    analyze_cache: dict[str, str] = {}
    return create_media_tools(ctx.user.id, ctx.turn_text, analyze_cache)


def _register() -> None:
    from backend.app.agent.tools.registry import SubToolInfo, default_registry

    default_registry.register(
        "media",
        _media_factory,
        core=True,
        summary="Describe and discard staged photos (agent-native storage)",
        dashboard_description="Describe and discard staged photos (agent-native storage)",
        dashboard_always_enabled=True,
        sub_tools=[
            SubToolInfo(
                ToolName.ANALYZE_PHOTO,
                "Run vision analysis on a staged photo",
                default_permission="always",
            ),
            SubToolInfo(
                ToolName.DISCARD_MEDIA,
                "Discard a staged photo per user request",
                default_permission="ask",
            ),
        ],
    )


_register()
