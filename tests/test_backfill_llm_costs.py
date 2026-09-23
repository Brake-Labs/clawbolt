"""Tests for the $0-usage-row backfill script.

Exercises ``scripts.backfill_llm_costs.backfill_costs`` against the test
database: a priceable ``cost=0`` row gets repaired, an unpriceable one is
left alone, an already-costed row is never re-priced, and dry-run writes
nothing.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select

from backend.app.config import settings
from backend.app.database import db_session_async
from backend.app.models import LLMEndpoint, LLMUsageLog, User
from backend.app.services.llm_pricing import compute_cost
from scripts.backfill_llm_costs import backfill_costs


async def _add_usage_row(
    user: User,
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cost: Decimal,
) -> int:
    async with db_session_async() as db:
        row = LLMUsageLog(
            user_id=user.id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            purpose="agent_main",
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row.id


async def _cost_of(row_id: int) -> Decimal:
    async with db_session_async() as db:
        return (
            await db.execute(select(LLMUsageLog.cost).where(LLMUsageLog.id == row_id))
        ).scalar_one()


async def test_apply_reprices_zero_cost_row(test_user: User) -> None:
    """A $0 row for a now-priced model is updated in place."""
    row_id = await _add_usage_row(
        test_user,
        model="claude-opus-4-8",
        provider="anthropic",
        input_tokens=7100,
        output_tokens=530,
        cost=Decimal("0.000000"),
    )

    result = await backfill_costs(model="claude-opus-4-8", provider="anthropic", apply=True)

    assert result.updated == 1
    assert result.total_added > Decimal("0")
    assert await _cost_of(row_id) > Decimal("0")


async def test_dry_run_writes_nothing(test_user: User) -> None:
    """Dry-run reports the row but leaves the stored cost at $0."""
    row_id = await _add_usage_row(
        test_user,
        model="claude-opus-4-8",
        provider="anthropic",
        input_tokens=7100,
        output_tokens=530,
        cost=Decimal("0.000000"),
    )

    result = await backfill_costs(model="claude-opus-4-8", provider="anthropic", apply=False)

    assert result.updated == 1
    assert result.applied is False
    assert await _cost_of(row_id) == Decimal("0.000000")


async def test_unpriceable_row_stays_zero(test_user: User) -> None:
    """A model genai-prices does not know is left at $0, not crashed on."""
    row_id = await _add_usage_row(
        test_user,
        model="not-a-real-model-99",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=1000,
        cost=Decimal("0.000000"),
    )

    result = await backfill_costs(apply=True)

    assert result.updated == 0
    assert await _cost_of(row_id) == Decimal("0.000000")


async def test_already_costed_row_is_untouched(test_user: User) -> None:
    """Rows that already carry a cost are never re-priced."""
    sentinel = Decimal("0.123456")
    row_id = await _add_usage_row(
        test_user,
        model="claude-opus-4-8",
        provider="anthropic",
        input_tokens=7100,
        output_tokens=530,
        cost=sentinel,
    )

    result = await backfill_costs(model="claude-opus-4-8", provider="anthropic", apply=True)

    assert result.updated == 0
    assert await _cost_of(row_id) == sentinel


async def test_zero_token_row_is_skipped(test_user: User) -> None:
    """A $0 row with no tokens is not a mispricing; leave it alone."""
    row_id = await _add_usage_row(
        test_user,
        model="claude-opus-4-8",
        provider="anthropic",
        input_tokens=0,
        output_tokens=0,
        cost=Decimal("0.000000"),
    )

    result = await backfill_costs(model="claude-opus-4-8", provider="anthropic", apply=True)

    assert result.scanned == 0
    assert result.updated == 0
    assert await _cost_of(row_id) == Decimal("0.000000")


async def _row_state(row_id: int) -> tuple[Decimal, bool, str]:
    async with db_session_async() as db:
        row = (await db.execute(select(LLMUsageLog).where(LLMUsageLog.id == row_id))).scalar_one()
        return row.cost, row.pricing_available, row.model


async def _add_unpriced_row(user: User, *, model: str, endpoint: str = "") -> int:
    """A row as the logger wrote it before the model could be priced."""
    async with db_session_async() as db:
        row = LLMUsageLog(
            user_id=user.id,
            endpoint=endpoint,
            provider="anthropic",
            model=model,
            pricing_available=False,
            input_tokens=7100,
            output_tokens=530,
            total_tokens=7630,
            cost=Decimal("0.000000"),
            purpose="agent_main",
            cache_creation_input_tokens=900,
            cache_read_input_tokens=40_000,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row.id


async def _add_endpoint(name: str, *, pricing: str) -> None:
    async with db_session_async() as db:
        db.add(LLMEndpoint(name=name, dialect="anthropic", pricing=pricing))
        await db.commit()


async def test_apply_reprices_a_route_prefixed_row_and_marks_it_priced(test_user: User) -> None:
    row_id = await _add_unpriced_row(test_user, model="clawbolt-anthropic:claude-opus-5-5")

    result = await backfill_costs(apply=True)

    cost, priced, model = await _row_state(row_id)
    assert result.updated == 1
    assert cost == result.total_added
    assert cost > Decimal("0")
    assert priced is True
    # The stored model is what was sent; only the price lookup changed.
    assert model == "clawbolt-anthropic:claude-opus-5-5"


async def test_apply_reprices_an_aliased_row(
    test_user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "llm_pricing_aliases", "clawbolt-prod=claude-opus-5")
    row_id = await _add_unpriced_row(test_user, model="clawbolt-prod")

    result = await backfill_costs(apply=True)

    cost, priced, model = await _row_state(row_id)
    assert result.updated == 1
    assert cost > Decimal("0")
    assert priced is True
    assert model == "clawbolt-prod"


async def test_dry_run_changes_neither_cost_nor_pricing_flag(test_user: User) -> None:
    row_id = await _add_unpriced_row(test_user, model="clawbolt-anthropic:claude-opus-5-5")

    result = await backfill_costs(apply=False)

    assert result.applied is False
    assert result.updated == 1
    assert result.total_added > Decimal("0")
    assert await _row_state(row_id) == (
        Decimal("0.000000"),
        False,
        "clawbolt-anthropic:claude-opus-5-5",
    )


async def test_rows_from_an_unpriced_endpoint_are_left_alone(test_user: User) -> None:
    """The endpoint said the (provider, model) pair does not name who billed
    the tokens. A backfill must not overrule that with a price-list hit."""
    await _add_endpoint("gw-unpriced", pricing="unpriced")
    row_id = await _add_unpriced_row(test_user, model="claude-opus-5-5", endpoint="gw-unpriced")

    result = await backfill_costs(apply=True)

    assert result.updated == 0
    assert result.skipped_endpoint == 1
    assert await _row_state(row_id) == (Decimal("0.000000"), False, "claude-opus-5-5")


async def test_rows_from_a_deleted_endpoint_are_left_alone(test_user: User) -> None:
    """Whether a deleted endpoint was priced is no longer knowable."""
    row_id = await _add_unpriced_row(test_user, model="claude-opus-5-5", endpoint="gone")

    result = await backfill_costs(apply=True)

    assert result.updated == 0
    assert result.skipped_endpoint == 1
    assert await _row_state(row_id) == (Decimal("0.000000"), False, "claude-opus-5-5")


async def test_rows_from_a_priced_endpoint_are_repriced(test_user: User) -> None:
    await _add_endpoint("gw-priced", pricing="auto")
    row_id = await _add_unpriced_row(test_user, model="claude-opus-5-5", endpoint="gw-priced")

    result = await backfill_costs(apply=True)

    assert result.updated == 1
    cost, priced, _ = await _row_state(row_id)
    assert cost > Decimal("0")
    assert priced is True


async def test_a_row_whose_model_is_still_unknown_keeps_its_flag(test_user: User) -> None:
    row_id = await _add_unpriced_row(test_user, model="clawbolt-anthropic:not-a-real-model-99")

    result = await backfill_costs(apply=True)

    assert result.updated == 0
    assert await _row_state(row_id) == (
        Decimal("0.000000"),
        False,
        "clawbolt-anthropic:not-a-real-model-99",
    )


async def test_backfill_charges_the_one_hour_write_premium(test_user: User) -> None:
    """A repaired row prices its 1h cache writes the way the live logger does."""
    async with db_session_async() as db:
        row = LLMUsageLog(
            user_id=test_user.id,
            provider="anthropic",
            model="claude-opus-4-5",
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
            cost=Decimal("0.000000"),
            purpose="agent_main",
            cache_creation_input_tokens=10_000,
            cache_creation_5m_input_tokens=0,
            cache_creation_1h_input_tokens=10_000,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        row_id = row.id

    await backfill_costs(model="claude-opus-4-5", provider="anthropic", apply=True)

    one_hour = compute_cost(
        "claude-opus-4-5",
        100,
        10,
        provider="anthropic",
        cache_creation_input_tokens=10_000,
        cache_creation_1h_input_tokens=10_000,
    )
    five_min = compute_cost(
        "claude-opus-4-5", 100, 10, provider="anthropic", cache_creation_input_tokens=10_000
    )
    assert await _cost_of(row_id) == one_hour
    assert one_hour > five_min
