"""What the live side billed over the window a run sampled.

The candidate's cost is a number the report computes from tokens it watched
come back. On its own it invites the reading that the deployment would pay
that instead of what it pays today, which nothing on the page supported: the
production figure was simply absent.

``llm_usage_logs`` already carries per-call tokens, the cache fields and
``pricing_available``, keyed by user and time, so the other side of that
comparison is one query away. It is read here rather than in ``report`` so
the aggregation stays a pure function of the turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models import LLMUsageLog

logger = logging.getLogger(__name__)


@dataclass
class ProductionUsage:
    """What the live side actually billed over the window the run sampled.

    It is not a like-for-like total and must not be rendered as one. The
    window is the sampled turns' own timestamps, so it includes every call
    the live agent made inside it: the tool rounds a replay never reaches,
    heartbeats, compaction. The replay's figure counts one decision per turn.
    It answers "what does this user cost today, over the same days", which is
    the question an operator weighing a switch is actually asking.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_cost: Decimal | None = None
    """Summed over the priced rows, or ``None`` when none of them is priced."""
    unpriced_calls: int = 0
    """Rows whose ``pricing_available`` is False, so their cost is not one.

    Non-zero means the dollar figure covers only part of the window, which
    the console says rather than presenting a partial sum as the total.
    """
    window_start: str = ""
    window_end: str = ""

    @property
    def billed_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens


async def read_production_usage(
    db: AsyncSession, user_id: str, *, start: datetime, end: datetime
) -> ProductionUsage:
    """Sum *user_id*'s live LLM spend over ``[start, end]``.

    Summed in SQL rather than by loading rows: a busy week is thousands of
    calls and the report needs six numbers from them. The cost sum is taken
    over the priced rows only, because ``cost`` is recorded as zero behind an
    unpriced gateway and a plain ``SUM`` would read that as free.
    """
    priced = LLMUsageLog.pricing_available.is_(True)
    row = (
        await db.execute(
            select(
                func.count(LLMUsageLog.id),
                func.coalesce(func.sum(LLMUsageLog.input_tokens), 0),
                func.coalesce(func.sum(LLMUsageLog.output_tokens), 0),
                func.coalesce(func.sum(LLMUsageLog.cache_read_input_tokens), 0),
                func.coalesce(func.sum(LLMUsageLog.cache_creation_input_tokens), 0),
                func.sum(case((priced, LLMUsageLog.cost), else_=None)),
                func.coalesce(func.sum(case((priced, 0), else_=1)), 0),
            ).where(
                LLMUsageLog.user_id == user_id,
                LLMUsageLog.created_at >= start,
                LLMUsageLog.created_at <= end,
            )
        )
    ).one()

    calls = int(row[0] or 0)
    cost = row[5]
    return ProductionUsage(
        calls=calls,
        input_tokens=int(row[1] or 0),
        output_tokens=int(row[2] or 0),
        cache_read_tokens=int(row[3] or 0),
        cache_creation_tokens=int(row[4] or 0),
        total_cost=Decimal(str(cost)) if cost is not None else None,
        unpriced_calls=int(row[6] or 0),
        window_start=start.isoformat(),
        window_end=end.isoformat(),
    )
