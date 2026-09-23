"""LLM cost computation backed by the ``genai-prices`` library.

Thin wrapper around `pydantic/genai-prices
<https://github.com/pydantic/genai-prices>`_, the community-maintained
source of truth for provider pricing. Adding a new model means
``uv lock --upgrade-package genai-prices``; we do not keep our own
rate table in code.

The library's price data is bundled at install time, so this module
makes no network call. Pricing data refreshes when we bump the
``genai-prices`` dependency.

The caller passes the any-llm provider id alongside the model name,
since both are known at every dispatch site (see
``settings.llm_provider``). Authoritative provider routing means we
never have to guess "which vendor does this model name belong to" from
prefixes; we just forward what the agent loop already knew.

Used by ``LLMUsageStore.log`` to populate the ``cost`` column on every
``llm_usage_logs`` row instead of leaving it hardcoded at 0.

Behind an LLM gateway the model string is a route name, not a vendor model
id. :func:`resolve_price_ref` is the one place that maps it back, and every
consumer (the usage logger, the model comparison report, the cost backfill)
prices through it so they agree. In order, first hit wins:

1. the name as sent (``claude-opus-5-5``);
2. the part after the last ``:``, for ``<route>:<model id>`` names
   (``clawbolt-anthropic:claude-opus-5-5``);
3. an operator alias for the name as sent (``LLM_PRICING_ALIASES``:
   ``clawbolt-prod=claude-opus-5``);
4. an alias for the part after the last ``:``.

Every lookup is scoped to *provider*, the endpoint's dialect: a Claude id
behind an ``openai``-dialect gateway does not price, and an alias helps only
if its target is listed under that dialect.

Only the lookup changes. Callers store the model string exactly as sent.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from genai_prices import Usage, calc_price

from backend.app.config import settings

logger = logging.getLogger(__name__)


# Cost column on ``llm_usage_logs`` is ``Numeric(12, 6)``; quantise to
# match so the DB never rejects an insert with too many fractional
# digits.
_QUANT = Decimal("0.000001")

# What an operator sees next to a model that could not be priced.
UNPRICED_HINT = (
    "Bump the genai-prices dependency if it is a new vendor model, or map a "
    "gateway alias to a model it knows with LLM_PRICING_ALIASES "
    "(e.g. clawbolt-prod=claude-opus-5)."
)


def _library_knows(model_ref: str, provider: str) -> bool:
    try:
        calc_price(
            Usage(input_tokens=1, output_tokens=0),
            model_ref=model_ref,
            provider_id=provider or None,
        )
    except (LookupError, ValueError):
        return False
    return True


def _candidates(model: str) -> list[str]:
    """Names to try for *model*, in order, without duplicates."""
    aliases = settings.llm_pricing_alias_map
    bare = model.rsplit(":", 1)[1] if ":" in model else ""
    # Real names before any alias, so an alias keyed on a route name cannot
    # shadow the vendor model id the route carries.
    names = [model, bare, aliases.get(model, ""), aliases.get(bare, "")]
    return list(dict.fromkeys(name for name in names if name))


def resolve_price_ref(model: str, *, provider: str = "") -> str | None:
    """The model ref ``genai-prices`` prices *model* as, or ``None``.

    See the module docstring for the order. A name the library already
    knows is returned unchanged, so an alias can fill a gap but never
    reprice a real model id.
    """
    for candidate in _candidates(model):
        if _library_knows(candidate, provider):
            return candidate
    return None


def is_known_model(model: str, *, provider: str = "") -> bool:
    """Whether ``genai-prices`` can price this (provider, model) pair.

    Used by the logger to decide when to emit a "no pricing entry"
    warning. Goes through :func:`resolve_price_ref`, the same lookup
    ``compute_cost`` uses, so a true result implies ``compute_cost`` will
    not fall back to zero.
    """
    return resolve_price_ref(model, provider=provider) is not None


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    provider: str = "",
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    cache_creation_1h_input_tokens: int | None = None,
) -> Decimal:
    """Return the dollar cost for a single LLM call as a 6-decimal Decimal.

    *provider* is the any-llm provider id (``"anthropic"``, ``"openai"``,
    etc.) under which *model* was invoked. Pass it explicitly: skipping
    autodetection is faster and disambiguates models whose name appears
    under more than one provider. *model* is resolved through
    :func:`resolve_price_ref`, so a gateway route name prices as the
    model behind it.

    Charges via ``genai-prices``, which understands per-provider
    accounting quirks (Anthropic's input bucket includes cached tokens,
    cache-write surcharges, cache-read discounts, OpenAI's prompt-cache
    discount, etc.) so the caller does not have to.

    *cache_creation_1h_input_tokens* is the part of
    *cache_creation_input_tokens* written with the 1-hour lifetime, when the
    provider reports the split. The library prices it at the model's 1-hour
    write rate and the rest at the 5-minute rate; ``None`` prices every
    write at the 5-minute rate.

    Returns ``Decimal('0.000000')`` for (provider, model) pairs that do
    not resolve; the caller should log a warning (see ``UNPRICED_HINT``)
    so a missing model produces a fix rather than silent zero-cost rows
    forever.
    """
    model_ref = resolve_price_ref(model, provider=provider)
    if model_ref is None:
        return Decimal("0.000000")
    cache_creation = cache_creation_input_tokens or 0
    cache_read = cache_read_input_tokens or 0
    # Anthropic (and several other providers) price the full input
    # bucket including cached tokens; the library expects
    # ``input_tokens`` to be the total. Per-bucket multipliers come
    # from the cache-specific fields.
    total_input = input_tokens + cache_creation + cache_read
    # ``cache_write_1h_tokens`` is a subset of ``cache_write_tokens``; the
    # library charges it the 1-hour rate and the remainder the 5-minute rate.
    one_hour = min(cache_creation_1h_input_tokens or 0, cache_creation)
    try:
        result = calc_price(
            Usage(
                input_tokens=total_input,
                cache_write_tokens=cache_creation,
                cache_write_1h_tokens=one_hour or None,
                cache_read_tokens=cache_read,
                output_tokens=output_tokens,
            ),
            model_ref=model_ref,
            provider_id=provider or None,
        )
    except (LookupError, ValueError):
        return Decimal("0.000000")
    return result.total_price.quantize(_QUANT)
