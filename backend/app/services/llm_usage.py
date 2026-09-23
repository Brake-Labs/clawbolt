"""LLM usage tracking helper.

Extracts token counts from amessages responses and persists them to the
``llm_usage_logs`` table for cost monitoring. Cost itself is computed
in ``LLMUsageStore.log`` via ``services.llm_pricing``.
"""

from __future__ import annotations

import logging

from any_llm.types.completion import ChatCompletion
from any_llm.types.messages import MessageResponse

from backend.app.agent.stores import LLMUsageStore

logger = logging.getLogger(__name__)


async def log_llm_usage(
    user_id: str,
    model: str,
    response: MessageResponse,
    purpose: str,
    provider: str = "",
    endpoint: str = "",
    priced: bool = True,
) -> None:
    """Extract token usage from an LLM response and save to the usage log.

    *provider* is the any-llm provider id (``"anthropic"``, ``"openai"``,
    ``"google"``, etc.) under which *model* was invoked. We thread it
    through rather than guessing from the model name so cost lookup,
    persistence, and downstream analytics all see the same authoritative
    string. Empty string is allowed for legacy callers that haven't been
    updated yet; cost lookup will fall through to autodetect in that case.

    *endpoint* and *priced* come from the resolved ``LLMTarget``. Behind a
    gateway the provider names the dialect rather than whoever billed the
    tokens, so ``priced=False`` records the usage with no cost rather than
    with a price-list figure that describes a vendor who never saw the call.
    """
    # Everything, extraction included, sits inside the guard: usage logging
    # is bookkeeping, and a response with a malformed ``usage`` must not take
    # down the call that produced it (a vision description, an agent reply).
    try:
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens

        cache_creation_input_tokens = response.usage.cache_creation_input_tokens
        cache_read_input_tokens = response.usage.cache_read_input_tokens
        # Per-lifetime split of the cache writes. Absent (None) for providers
        # that do not report it, and stored as NULL rather than 0 so "not
        # reported" never reads as "nothing written at that lifetime".
        by_ttl = response.usage.cache_creation
        cache_creation_5m = by_ttl.ephemeral_5m_input_tokens if by_ttl else None
        cache_creation_1h = by_ttl.ephemeral_1h_input_tokens if by_ttl else None

        store = LLMUsageStore(user_id)
        await store.log_async(
            model,
            prompt_tokens,
            completion_tokens,
            purpose,
            provider=provider,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            endpoint=endpoint,
            priced=priced,
            cache_creation_5m_input_tokens=cache_creation_5m,
            cache_creation_1h_input_tokens=cache_creation_1h,
        )
    except Exception:
        logger.exception("Failed to log LLM usage for user %s", user_id)
        return

    logger.info(
        "LLM usage logged: user=%s target=%s model=%s purpose=%s "
        "tokens=%d cache_create=%s (5m=%s 1h=%s) cache_read=%s",
        user_id,
        endpoint or provider or "?",
        model,
        purpose,
        total_tokens,
        cache_creation_input_tokens,
        cache_creation_5m,
        cache_creation_1h,
        cache_read_input_tokens,
    )


async def log_chat_completion_usage(
    user_id: str,
    model: str,
    response: ChatCompletion,
    purpose: str,
    provider: str = "",
    endpoint: str = "",
    priced: bool = True,
) -> None:
    """Record usage for a chat-completions (``acompletion``) response.

    The Messages-API sibling of :func:`log_llm_usage` for the few calls that
    need ``response_format`` structured output. Chat completions report no
    cache-write split, so the cache columns stay NULL. Never raises.
    """
    try:
        usage = response.usage
        if usage is None:
            logger.warning("No usage on %s response for user %s; nothing to log", purpose, user_id)
            return
        await LLMUsageStore(user_id).log_async(
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
            purpose,
            provider=provider,
            endpoint=endpoint,
            priced=priced,
        )
    except Exception:
        logger.exception("Failed to log LLM usage for user %s", user_id)
        return
    logger.info(
        "LLM usage logged: user=%s target=%s model=%s purpose=%s tokens=%d",
        user_id,
        endpoint or provider or "?",
        model,
        purpose,
        usage.prompt_tokens + usage.completion_tokens,
    )
