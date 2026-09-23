"""Vision and approval-classifier calls write their token usage to ``llm_usage_logs``.

Both run outside the agent loop, so they log their own rows rather than
riding on the loop's per-round logging.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import CacheCreation
from any_llm.types.completion import ChatCompletion
from any_llm.types.messages import MessageResponse
from sqlalchemy import select

import backend.app.database as _db_module
from backend.app.agent import media_staging
from backend.app.agent.approval import ApprovalDecision, classify_approval_response
from backend.app.agent.observer import PURPOSE_APPROVAL_CLASSIFICATION, PURPOSE_VISION
from backend.app.agent.stores import LLMUsageStore
from backend.app.agent.tools.media_tools import create_media_tools
from backend.app.agent.tools.names import ToolName
from backend.app.config import settings
from backend.app.media.pipeline import VISION_FALLBACK, run_vision_on_media
from backend.app.media.vision import analyze_image
from backend.app.models import LLMUsageLog, User
from backend.app.services.llm_pricing import compute_cost
from backend.app.services.llm_service import LLMTarget
from backend.app.services.llm_usage import log_llm_usage
from tests.mocks.llm import make_vision_response

PRICED_MODEL = "claude-sonnet-4-20250514"


def _vision_response(description: str = "A cracked cedar deck board.") -> MessageResponse:
    response = make_vision_response(description)
    response.usage.input_tokens = 1_600
    response.usage.output_tokens = 420
    response.usage.cache_creation_input_tokens = 200
    response.usage.cache_read_input_tokens = 50
    response.usage.cache_creation = CacheCreation(
        ephemeral_5m_input_tokens=200, ephemeral_1h_input_tokens=0
    )
    return response


async def _rows(user_id: str) -> list[LLMUsageLog]:
    async with _db_module.db_session_async() as db:
        result = await db.execute(select(LLMUsageLog).filter_by(user_id=user_id))
        return list(result.scalars().all())


@pytest.fixture()
def vision_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare-provider vision role, as a single-tenant install configures it."""
    monkeypatch.setattr(settings, "vision_endpoint", "")
    monkeypatch.setattr(settings, "vision_provider", "anthropic")
    monkeypatch.setattr(settings, "vision_model", PRICED_MODEL)


@pytest.mark.usefixtures("vision_role")
async def test_vision_call_writes_one_usage_row(test_user: User) -> None:
    with patch(
        "backend.app.media.vision.amessages_streamed",
        new=AsyncMock(return_value=_vision_response()),
    ):
        text = await analyze_image(b"jpeg", "image/jpeg", user_id=test_user.id)

    assert text == "A cracked cedar deck board."
    rows = await _rows(test_user.id)
    assert len(rows) == 1
    row = rows[0]
    assert row.purpose == PURPOSE_VISION == "vision"
    assert row.provider == "anthropic"
    assert row.model == PRICED_MODEL
    assert row.endpoint == ""
    assert row.input_tokens == 1_600
    assert row.output_tokens == 420
    assert row.total_tokens == 2_020
    assert row.cache_creation_input_tokens == 200
    assert row.cache_read_input_tokens == 50
    assert row.cache_creation_5m_input_tokens == 200
    assert row.cache_creation_1h_input_tokens == 0
    expected = compute_cost(
        PRICED_MODEL,
        1_600,
        420,
        provider="anthropic",
        cache_creation_input_tokens=200,
        cache_read_input_tokens=50,
        cache_creation_1h_input_tokens=0,
    )
    assert expected > 0
    assert row.cost == expected
    assert row.pricing_available is True


async def test_vision_through_a_named_endpoint_logs_it_and_prices_the_route(
    test_user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway route name prices as the model behind it, as in the agent loop."""
    monkeypatch.setattr(settings, "llm_pricing_aliases", "clawbolt-vision=claude-opus-5")
    target = LLMTarget(provider="anthropic", model="clawbolt-vision", endpoint="gw")
    with (
        patch("backend.app.media.vision.resolve_target", new=AsyncMock(return_value=target)),
        patch(
            "backend.app.media.vision.amessages_streamed",
            new=AsyncMock(return_value=_vision_response()),
        ),
    ):
        await analyze_image(b"jpeg", "image/jpeg", user_id=test_user.id)

    (row,) = await _rows(test_user.id)
    assert row.endpoint == "gw"
    assert row.model == "clawbolt-vision"
    assert row.cost > 0
    assert row.cost == compute_cost(
        "clawbolt-vision",
        1_600,
        420,
        provider="anthropic",
        cache_creation_input_tokens=200,
        cache_read_input_tokens=50,
        cache_creation_1h_input_tokens=0,
    )


async def test_vision_on_an_unpriced_endpoint_records_zero_cost(test_user: User) -> None:
    target = LLMTarget(provider="anthropic", model=PRICED_MODEL, endpoint="gw", priced=False)
    with (
        patch("backend.app.media.vision.resolve_target", new=AsyncMock(return_value=target)),
        patch(
            "backend.app.media.vision.amessages_streamed",
            new=AsyncMock(return_value=_vision_response()),
        ),
    ):
        await analyze_image(b"jpeg", "image/jpeg", user_id=test_user.id)

    (row,) = await _rows(test_user.id)
    assert row.cost == 0
    assert row.pricing_available is False


@pytest.mark.usefixtures("vision_role")
async def test_failing_usage_write_does_not_break_vision(test_user: User) -> None:
    """The description still reaches the caller, not the fallback text."""
    with (
        patch(
            "backend.app.media.vision.amessages_streamed",
            new=AsyncMock(return_value=_vision_response("Rotted sill plate.")),
        ),
        patch.object(
            LLMUsageStore, "log_async", new=AsyncMock(side_effect=RuntimeError("db down"))
        ),
    ):
        text = await run_vision_on_media(b"jpeg", "image/jpeg", "", user_id=test_user.id)

    assert text == "Rotted sill plate."
    assert text != VISION_FALLBACK


async def test_malformed_usage_does_not_raise(test_user: User) -> None:
    response = make_vision_response()
    response.usage = None  # type: ignore[assignment]
    await log_llm_usage(test_user.id, PRICED_MODEL, response, PURPOSE_VISION)
    assert await _rows(test_user.id) == []


@pytest.mark.usefixtures("vision_role")
async def test_analyze_photo_attributes_vision_spend_to_the_user(test_user: User) -> None:
    handle = await media_staging.stage(test_user.id, "url-usage", b"bytes", "image/jpeg")
    assert handle is not None
    tools = create_media_tools(test_user.id, "what is this", {})
    analyze = next(t for t in tools if t.name == ToolName.ANALYZE_PHOTO)

    with patch(
        "backend.app.media.vision.amessages_streamed",
        new=AsyncMock(return_value=_vision_response()),
    ):
        result = await analyze.function(handle=handle)

    assert result.is_error is False
    (row,) = await _rows(test_user.id)
    assert row.purpose == PURPOSE_VISION
    assert row.output_tokens == 420


def _classifier_response(decision: str = "approved") -> ChatCompletion:
    parsed = MagicMock()
    parsed.decision = decision
    message = MagicMock()
    message.parsed = parsed
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=310, completion_tokens=12)
    return response


async def test_approval_classification_writes_a_usage_row(
    test_user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "compaction_model", PRICED_MODEL)
    monkeypatch.setattr(settings, "compaction_provider", "anthropic")
    with patch(
        "backend.app.agent.approval.acompletion",
        new=AsyncMock(return_value=_classifier_response()),
    ):
        decision = await classify_approval_response("sure go ahead", user_id=test_user.id)

    assert decision == ApprovalDecision.APPROVED
    (row,) = await _rows(test_user.id)
    assert row.purpose == PURPOSE_APPROVAL_CLASSIFICATION
    assert row.model == PRICED_MODEL
    assert row.provider == "anthropic"
    assert row.input_tokens == 310
    assert row.output_tokens == 12
    assert row.cost == compute_cost(PRICED_MODEL, 310, 12, provider="anthropic")
    assert row.cost > 0


async def test_failing_usage_write_does_not_break_approval_classification(
    test_user: User,
) -> None:
    with (
        patch(
            "backend.app.agent.approval.acompletion",
            new=AsyncMock(return_value=_classifier_response("denied")),
        ),
        patch.object(
            LLMUsageStore, "log_async", new=AsyncMock(side_effect=RuntimeError("db down"))
        ),
    ):
        decision = await classify_approval_response("nah", user_id=test_user.id)

    assert decision == ApprovalDecision.DENIED
