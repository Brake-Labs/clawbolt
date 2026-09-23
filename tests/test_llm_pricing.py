"""Tests for LLM cost computation.

The pricing module is a thin wrapper around ``genai-prices``; we test
its behavior contract (returns Decimal, 6-decimal quantisation,
unknown-model fallback, Anthropic input-token bucketing convention,
explicit provider routing) rather than asserting exact dollar amounts.
Pinning specific rates would defeat the point of switching to a library
that updates prices when providers change them.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from genai_prices import Usage, calc_price

from backend.app.config import settings
from backend.app.services.llm_pricing import compute_cost, is_known_model, resolve_price_ref

# ---------------------------------------------------------------------------
# Known / unknown model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",  # model deployed in prod; missing pricing here logs $0 usage rows
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",  # dated alias prefix-matches
    ],
)
def test_is_known_model_for_supported_models(model: str) -> None:
    assert is_known_model(model, provider="anthropic") is True


def test_is_known_model_for_unmapped_model() -> None:
    assert is_known_model("not-a-real-model-99", provider="anthropic") is False
    assert is_known_model("", provider="anthropic") is False


def test_is_known_model_works_without_provider_via_autodetect() -> None:
    """Empty provider falls through to ``calc_price`` autodetection.
    Used by legacy callers that haven't been updated yet."""
    assert is_known_model("claude-sonnet-4-6") is True
    assert is_known_model("not-a-real-model") is False


# ---------------------------------------------------------------------------
# compute_cost contract
# ---------------------------------------------------------------------------


def test_compute_cost_returns_decimal() -> None:
    cost = compute_cost("claude-sonnet-4-6", 1000, 500, provider="anthropic")
    assert isinstance(cost, Decimal)


def test_compute_cost_zero_tokens_costs_nothing() -> None:
    assert compute_cost("claude-sonnet-4-6", 0, 0, provider="anthropic") == Decimal("0.000000")


def test_unknown_model_falls_through_to_zero() -> None:
    """Conservative fallback: prefer 'we don't know' over a bad estimate."""
    assert compute_cost("not-a-real-model", 1_000_000, 1_000_000, provider="anthropic") == Decimal(
        "0.000000"
    )


def test_unknown_provider_falls_through_to_zero() -> None:
    """A custom local-provider id (e.g. self-hosted ollama) genai-prices
    doesn't know about must not crash; we just record cost=0."""
    assert compute_cost("claude-sonnet-4-6", 1000, 500, provider="my-local-shim") == Decimal(
        "0.000000"
    )


def test_compute_cost_handles_none_cache_columns() -> None:
    """Per-row cache columns can be NULL (older logs, providers that
    don't expose them). Treat as zero, not as a TypeError."""
    cost = compute_cost(
        "claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=10,
        provider="anthropic",
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert cost > Decimal("0")


def test_compute_cost_quantises_to_six_decimals() -> None:
    """Numeric(12, 6) on the column constrains us to 6 fractional
    digits. The library may return more precision than that."""
    cost = compute_cost("claude-sonnet-4-6", 1, 0, provider="anthropic")
    # Exponent is -6 (six fractional digits) regardless of how the
    # library rounded internally.
    exponent = cost.as_tuple().exponent
    assert isinstance(exponent, int)
    assert exponent == -6


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------


def test_provider_is_passed_to_library_not_inferred_from_model() -> None:
    """When the caller passes a provider explicitly, the library uses
    it directly (no name-prefix guessing on our end). A model-name
    string on its own is no longer enough information for routing."""
    # Same model id, two different providers: behavior should depend on
    # the provider we pass, not on a heuristic over the name.
    pc = calc_price(
        Usage(input_tokens=100, output_tokens=50),
        model_ref="claude-sonnet-4-6",
        provider_id="anthropic",
    )
    our = compute_cost("claude-sonnet-4-6", 100, 50, provider="anthropic")
    assert our == pc.total_price.quantize(Decimal("0.000001"))


def test_compute_cost_works_without_provider_via_autodetect() -> None:
    """Empty provider string falls through to ``calc_price`` autodetect.
    This is the legacy fallback for old callers."""
    cost = compute_cost("claude-sonnet-4-6", 1000, 500)
    assert cost > Decimal("0")


# ---------------------------------------------------------------------------
# Anthropic input-token bucketing convention
# ---------------------------------------------------------------------------


def test_compute_cost_aggregates_input_buckets_for_anthropic() -> None:
    """``genai-prices`` follows Anthropic's wire-protocol convention:
    ``Usage.input_tokens`` is the total prompt size (uncached + cache
    creation + cache reads), with cache write/read columns charging
    their own rates on top.

    Our `compute_cost` API takes them split out (matching what we
    persist) and is responsible for the aggregation. This test pins
    that wiring by checking ``compute_cost`` matches a hand-built
    library call.
    """
    library_result = calc_price(
        Usage(
            input_tokens=362 + 2206,  # plain + cache_write
            cache_write_tokens=2206,
            cache_read_tokens=0,
            output_tokens=92,
        ),
        model_ref="claude-sonnet-4-6",
        provider_id="anthropic",
    ).total_price.quantize(Decimal("0.000001"))

    our_result = compute_cost(
        "claude-sonnet-4-6",
        input_tokens=362,
        output_tokens=92,
        provider="anthropic",
        cache_creation_input_tokens=2206,
        cache_read_input_tokens=0,
    )
    assert our_result == library_result


def test_compute_cost_treats_cache_read_distinctly() -> None:
    """Cache reads are charged at a discount in Anthropic's pricing.
    The library handles the multiplier; we just need to pass them
    through the right field. Compare two equal-token calls where one
    has the cache-read shape and one has only plain input."""
    cache_heavy = compute_cost(
        "claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        provider="anthropic",
        cache_creation_input_tokens=0,
        cache_read_input_tokens=10_000,
    )
    plain = compute_cost(
        "claude-sonnet-4-6",
        input_tokens=10_000,
        output_tokens=0,
        provider="anthropic",
    )
    # Cache reads are cheaper than plain input.
    assert cache_heavy < plain
    # And both are non-zero.
    assert cache_heavy > Decimal("0")
    assert plain > Decimal("0")


def test_known_anthropic_models_all_produce_nonzero_cost() -> None:
    """Smoke test: a 1k input + 500 output call returns >0 cost for
    every Anthropic SKU we currently invoke. Catches a future
    library refactor that quietly drops one of these."""
    for model in (
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",  # model deployed in prod; guards against $0 usage rows
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
    ):
        cost = compute_cost(model, 1000, 500, provider="anthropic")
        assert cost > Decimal("0"), f"{model} priced at zero"


# ---------------------------------------------------------------------------
# Claude 5 (genai-prices >= 0.1.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5"])
def test_claude_5_models_are_priced(model: str) -> None:
    assert is_known_model(model, provider="anthropic") is True
    assert compute_cost(model, 1000, 500, provider="anthropic") > Decimal("0")


def test_claude_opus_5_5_is_not_priced_as_claude_opus_5() -> None:
    """0.1.7 prefix-matched ``claude-opus-5-5`` onto ``claude-opus-5`` and
    billed it at the older, dearer rate. Its own entry is cheaper."""
    newer = compute_cost("claude-opus-5-5", 1_000_000, 1_000_000, provider="anthropic")
    older = compute_cost("claude-opus-5", 1_000_000, 1_000_000, provider="anthropic")
    assert Decimal("0") < newer < older


# ---------------------------------------------------------------------------
# Gateway route prefixes and operator aliases
# ---------------------------------------------------------------------------


def test_route_prefixed_name_prices_like_the_bare_model() -> None:
    """A gateway route name is ``<route>:<model id>``. The price is the model's."""
    prefixed = compute_cost(
        "clawbolt-anthropic:claude-opus-5-5",
        input_tokens=1200,
        output_tokens=300,
        provider="anthropic",
        cache_creation_input_tokens=400,
        cache_read_input_tokens=5000,
    )
    bare = compute_cost(
        "claude-opus-5-5",
        input_tokens=1200,
        output_tokens=300,
        provider="anthropic",
        cache_creation_input_tokens=400,
        cache_read_input_tokens=5000,
    )
    assert prefixed == bare
    assert prefixed > Decimal("0")
    assert is_known_model("clawbolt-anthropic:claude-opus-5-5", provider="anthropic") is True
    assert resolve_price_ref("clawbolt-anthropic:claude-opus-5-5", provider="anthropic") == (
        "claude-opus-5-5"
    )


def test_only_the_part_after_the_last_colon_is_the_model() -> None:
    assert resolve_price_ref("gw:eu:claude-sonnet-5", provider="anthropic") == "claude-sonnet-5"


def test_a_name_the_library_knows_is_not_rewritten() -> None:
    assert resolve_price_ref("claude-sonnet-5", provider="anthropic") == "claude-sonnet-5"


def test_route_prefix_with_an_unknown_model_is_still_unpriced() -> None:
    model = "clawbolt-anthropic:not-a-real-model-99"
    assert is_known_model(model, provider="anthropic") is False
    assert resolve_price_ref(model, provider="anthropic") is None
    assert compute_cost(model, 1000, 1000, provider="anthropic") == Decimal("0.000000")


def test_an_alias_resolves_to_its_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_pricing_aliases", "clawbolt-prod=claude-opus-5")
    assert is_known_model("clawbolt-prod", provider="anthropic") is True
    assert resolve_price_ref("clawbolt-prod", provider="anthropic") == "claude-opus-5"
    assert compute_cost("clawbolt-prod", 1000, 500, provider="anthropic") == compute_cost(
        "claude-opus-5", 1000, 500, provider="anthropic"
    )


def test_an_alias_applies_to_the_bare_name_behind_a_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_pricing_aliases", "clawbolt-prod=claude-opus-5")
    assert resolve_price_ref("gw:clawbolt-prod", provider="anthropic") == "claude-opus-5"


def test_an_alias_does_not_override_a_name_the_library_knows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aliases fill gaps; a stale one must not reprice a real model id."""
    monkeypatch.setattr(settings, "llm_pricing_aliases", "claude-sonnet-5=claude-opus-5")
    assert resolve_price_ref("claude-sonnet-5", provider="anthropic") == "claude-sonnet-5"


def test_an_alias_on_a_route_name_does_not_override_the_model_it_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A route alias set before genai-prices knew the model must not outlive
    that gap once the model id after the ``:`` prices on its own."""
    monkeypatch.setattr(
        settings, "llm_pricing_aliases", "clawbolt-anthropic:claude-opus-5-5=claude-opus-5"
    )
    assert (
        resolve_price_ref("clawbolt-anthropic:claude-opus-5-5", provider="anthropic")
        == "claude-opus-5-5"
    )


def test_an_alias_to_an_unknown_target_stays_unpriced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_pricing_aliases", "clawbolt-prod=not-a-real-model-99")
    assert is_known_model("clawbolt-prod", provider="anthropic") is False
    assert compute_cost("clawbolt-prod", 1000, 500, provider="anthropic") == Decimal("0.000000")


def test_without_an_alias_the_gateway_alias_is_unpriced() -> None:
    assert is_known_model("clawbolt-prod", provider="anthropic") is False
    assert compute_cost("clawbolt-prod", 1000, 500, provider="anthropic") == Decimal("0.000000")


# ---------------------------------------------------------------------------
# Cache-write lifetime split
# ---------------------------------------------------------------------------


def _base_input_cost(model: str, tokens: int) -> Decimal:
    return calc_price(
        Usage(input_tokens=tokens, output_tokens=0), model_ref=model, provider_id="anthropic"
    ).total_price


def test_one_hour_cache_writes_bill_at_twice_base_input() -> None:
    """Anthropic bills a 1h write at 2x base input and a 5m write at 1.25x."""
    model, tokens = "claude-opus-4-5", 1_000_000
    base = _base_input_cost(model, tokens)
    five_min = compute_cost(model, 0, 0, provider="anthropic", cache_creation_input_tokens=tokens)
    one_hour = compute_cost(
        model,
        0,
        0,
        provider="anthropic",
        cache_creation_input_tokens=tokens,
        cache_creation_1h_input_tokens=tokens,
    )
    assert five_min == (base * Decimal("1.25")).quantize(Decimal("0.000001"))
    assert one_hour == (base * 2).quantize(Decimal("0.000001"))


@pytest.mark.parametrize(
    ("one_hour_tokens", "expected"),
    [(0, Decimal("5.000000")), (400_000, Decimal("6.200000")), (1_000_000, Decimal("8.000000"))],
)
def test_opus_5_5_cache_write_cost_by_one_hour_share(
    one_hour_tokens: int, expected: Decimal
) -> None:
    """1M written tokens on claude-opus-5-5 ($4/M base input): all 5m is 1.25x,
    all 1h is 2x, and a 40% 1h split lands in between. Pins the 1h count as a
    subset of the total, not an addition to it."""
    cost = compute_cost(
        "claude-opus-5-5",
        0,
        0,
        provider="anthropic",
        cache_creation_input_tokens=1_000_000,
        cache_creation_1h_input_tokens=one_hour_tokens,
    )
    assert cost == expected


def test_mixed_lifetimes_charge_only_the_one_hour_share_extra() -> None:
    model = "claude-opus-4-5"
    all_five_min = compute_cost(
        model, 100, 10, provider="anthropic", cache_creation_input_tokens=4_000
    )
    mixed = compute_cost(
        model,
        100,
        10,
        provider="anthropic",
        cache_creation_input_tokens=4_000,
        cache_creation_1h_input_tokens=1_000,
    )
    extra = _base_input_cost(model, 1_000) * Decimal("0.75")
    assert mixed == (all_five_min + extra).quantize(Decimal("0.000001"))


def test_unreported_split_prices_like_before() -> None:
    """``None`` (provider did not report the split) prices every write as 5m."""
    plain = compute_cost(
        "claude-opus-4-5", 10, 10, provider="anthropic", cache_creation_input_tokens=2_000
    )
    unreported = compute_cost(
        "claude-opus-4-5",
        10,
        10,
        provider="anthropic",
        cache_creation_input_tokens=2_000,
        cache_creation_1h_input_tokens=None,
    )
    assert plain == unreported


def test_route_prefixed_name_gets_the_one_hour_rate() -> None:
    """The 1h split prices through the resolved ref like everything else."""
    costs = {
        model: compute_cost(
            model,
            0,
            0,
            provider="anthropic",
            cache_creation_input_tokens=1_000_000,
            cache_creation_1h_input_tokens=400_000,
        )
        for model in ("clawbolt-anthropic:claude-opus-5-5", "claude-opus-5-5")
    }
    assert set(costs.values()) == {Decimal("6.200000")}
