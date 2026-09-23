"""qb_update must never drop fields or lines the agent left out.

A full QBO update nulls every writable field the body omits, and a ``Line``
array in any update (sparse included) replaces every stored line. These
tests drive ``qb_update`` through ``QuickBooksOnlineService`` over an
``httpx.MockTransport`` and assert on the body QuickBooks would receive.
All data is synthetic.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from backend.app.agent.tools.base import Tool, ToolErrorKind
from backend.app.integrations.quickbooks import service as service_module
from backend.app.integrations.quickbooks.factory import (
    QBUpdateParams,
    _format_results,
    create_quickbooks_tools,
)
from backend.app.integrations.quickbooks.service import QuickBooksOnlineService


def _stored_estimate() -> dict[str, Any]:
    """An Estimate as QBO returns it from a read, with the fields a full
    update would wipe."""
    return {
        "domain": "QBO",
        "sparse": False,
        "Id": "2001",
        "SyncToken": "4",
        "MetaData": {"CreateTime": "2026-09-01T10:00:00-07:00"},
        "DocNumber": "EST-2001",
        "TxnDate": "2026-09-01",
        "ExpirationDate": "2026-10-01",
        "DueDate": "2026-10-15",
        "TxnStatus": "Pending",
        "CustomerRef": {"value": "58", "name": "Test Customer"},
        "BillEmail": {"Address": "test.customer@example.com"},
        "CustomerMemo": {"value": "Thanks for the opportunity."},
        "SalesTermRef": {"value": "3", "name": "Net 30"},
        "ShipAddr": {"Id": "9", "Line1": "100 Oak St", "City": "Springfield"},
        "CustomField": [
            {"DefinitionId": "1", "Name": "Crew", "Type": "StringType", "StringValue": "B"}
        ],
        "TxnTaxDetail": {"TxnTaxCodeRef": {"value": "3"}, "TotalTax": 0},
        "TotalAmt": 612.5,
        "Line": [
            {
                "Id": "1",
                "LineNum": 1,
                "Description": "Labor - kitchen remodel",
                "Amount": 400.0,
                "DetailType": "SalesItemLineDetail",
                "SalesItemLineDetail": {
                    "ItemRef": {"value": "1", "name": "Services"},
                    "Qty": 8,
                    "UnitPrice": 50,
                    "TaxCodeRef": {"value": "NON"},
                },
            },
            {
                "Id": "2",
                "LineNum": 2,
                "Amount": 212.5,
                "DetailType": "SalesItemLineDetail",
                "SalesItemLineDetail": {
                    "ItemRef": {"value": "5", "name": "Materials"},
                    "Qty": 1,
                    "UnitPrice": 212.5,
                    "TaxCodeRef": {"value": "TAX"},
                    "ServiceDate": "2026-09-10",
                },
            },
            {"Amount": 612.5, "DetailType": "SubTotalLineDetail", "SubTotalLineDetail": {}},
        ],
    }


class FakeQBO:
    """In-process QBO: answers reads of one stored Estimate and records
    every update body."""

    def __init__(self, stored: dict[str, Any]) -> None:
        self.stored = stored
        self.posts: list[dict[str, Any]] = []
        self.reads = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/estimate/2001"):
            self.reads += 1
            return httpx.Response(200, json={"Estimate": copy.deepcopy(self.stored)})
        if request.method == "POST" and request.url.path.endswith("/estimate"):
            body = json.loads(request.content)
            self.posts.append(body)
            echoed = {**copy.deepcopy(self.stored), **body, "SyncToken": "5"}
            return httpx.Response(200, json={"Estimate": echoed})
        return httpx.Response(404, json={"Fault": {"Error": [{"code": "610"}]}})


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service_module.httpx, "AsyncClient", factory)


@pytest.fixture
def qbo(monkeypatch: pytest.MonkeyPatch) -> FakeQBO:
    fake = FakeQBO(_stored_estimate())
    _patch_transport(monkeypatch, fake.handler)
    return fake


def _tools() -> list[Tool]:
    svc = QuickBooksOnlineService(
        client_id="cid",
        client_secret="csec",
        realm_id="9999",
        access_token="tok",
        refresh_token="rt",
        environment="sandbox",
    )
    return create_quickbooks_tools(svc)


def _tool(name: str) -> Tool:
    return next(t for t in _tools() if t.name == name)


async def _update(**args: Any) -> Any:
    """Call qb_update the way the agent loop does: through the params model."""
    tool = _tool("qb_update")
    validated = tool.params_model.model_validate(args).model_dump()
    return await tool.function(**validated)


def _lines_by_id(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(ln["Id"]): ln for ln in body["Line"] if "Id" in ln}


# -- Fields -----------------------------------------------------------------


async def test_memo_only_update_is_sparse_and_leaves_lines_and_fields_alone(
    qbo: FakeQBO,
) -> None:
    result = await _update(
        entity_type="Estimate",
        data={"Id": "2001", "SyncToken": "4", "CustomerMemo": {"value": "Updated memo"}},
    )

    assert result.is_error is False, result.content
    assert len(qbo.posts) == 1
    body = qbo.posts[0]
    assert body == {
        "Id": "2001",
        "SyncToken": "4",
        "sparse": True,
        "CustomerMemo": {"value": "Updated memo"},
    }
    assert "SyncToken: 5" in result.content


async def test_stale_synctoken_is_a_conflict_and_nothing_is_sent(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={"Id": "2001", "SyncToken": "3", "CustomerMemo": {"value": "x"}},
    )

    assert result.is_error is True
    assert result.error_kind == ToolErrorKind.VALIDATION
    assert "changed in QuickBooks since it was read" in result.content
    assert "SyncToken is now 4" in result.content
    assert qbo.posts == []


async def test_missing_synctoken_is_rejected(qbo: FakeQBO) -> None:
    result = await _update(entity_type="Estimate", data={"Id": "2001", "DueDate": "2026-11-01"})
    assert result.is_error is True
    assert "SyncToken is missing" in result.content
    assert qbo.posts == []


# -- Lines ------------------------------------------------------------------


async def test_changing_one_line_keeps_the_others_and_their_detail(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [
                {
                    "Id": "1",
                    "Amount": 500.0,
                    "SalesItemLineDetail": {"Qty": 10},
                }
            ],
        },
    )

    assert result.is_error is False, result.content
    body = qbo.posts[0]
    assert body["sparse"] is True
    stored = _lines_by_id(_stored_estimate())
    sent = _lines_by_id(body)
    assert set(sent) == {"1", "2"}
    # Changed line: new Amount and Qty; item, price and tax code kept.
    assert sent["1"]["Amount"] == 500.0
    assert sent["1"]["Description"] == "Labor - kitchen remodel"
    assert sent["1"]["SalesItemLineDetail"] == {
        "ItemRef": {"value": "1", "name": "Services"},
        "Qty": 10,
        "UnitPrice": 50,
        "TaxCodeRef": {"value": "NON"},
    }
    # Untouched line goes back exactly as stored.
    assert sent["2"] == stored["2"]
    # Top-level fields are not in the body, so QBO keeps them.
    for kept in ("BillEmail", "CustomerMemo", "DueDate", "SalesTermRef", "ShipAddr"):
        assert kept not in body


async def test_update_that_omits_lines_does_not_delete_them(qbo: FakeQBO) -> None:
    """The old SKILL.md example: new lines with no Ids. They are added,
    and every stored line is still sent."""
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [
                {
                    "Amount": 75.0,
                    "DetailType": "SalesItemLineDetail",
                    "Description": "Permit fee",
                    "SalesItemLineDetail": {"Qty": 1, "UnitPrice": 75.0},
                }
            ],
        },
    )

    assert result.is_error is False, result.content
    body = qbo.posts[0]
    assert _lines_by_id(body) == _lines_by_id(_stored_estimate())
    new = [ln for ln in body["Line"] if "Id" not in ln and ln["DetailType"] != "SubTotalLineDetail"]
    assert [ln["Description"] for ln in new] == ["Permit fee"]
    # Appended before QBO's trailing subtotal line.
    assert body["Line"][-1]["DetailType"] == "SubTotalLineDetail"
    assert body["Line"][-2]["Description"] == "Permit fee"


async def test_line_is_removed_only_by_explicit_delete(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={"Id": "2001", "SyncToken": "4"},
        delete_line_ids=[2],
    )

    assert result.is_error is False, result.content
    body = qbo.posts[0]
    assert set(_lines_by_id(body)) == {"1"}
    assert _lines_by_id(body)["1"] == _lines_by_id(_stored_estimate())["1"]


async def test_unknown_line_id_is_rejected(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={"Id": "2001", "SyncToken": "4", "Line": [{"Id": "7", "Amount": 1.0}]},
    )
    assert result.is_error is True
    assert "Line Id 7 is not on Estimate 2001" in result.content
    assert "1, 2" in result.content
    assert qbo.posts == []


async def test_deleting_every_line_is_rejected(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={"Id": "2001", "SyncToken": "4"},
        delete_line_ids=["1", "2"],
    )
    assert result.is_error is True
    assert "no line items" in result.content
    assert qbo.posts == []


def test_delete_line_ids_accepts_numbers() -> None:
    parsed = QBUpdateParams.model_validate(
        {"entity_type": "Estimate", "data": {"Id": "1"}, "delete_line_ids": [2, "3"]}
    )
    assert parsed.delete_line_ids == ["2", "3"]


async def test_resending_the_queried_record_adds_no_line(qbo: FakeQBO) -> None:
    """A full payload copied from qb_query, one quantity edited. Its lines
    carry Ids except QuickBooks' trailing subtotal, which must not come
    back as a second subtotal row; unchanged fields and totals stay out."""
    full = _stored_estimate()
    full["Line"][0]["SalesItemLineDetail"]["Qty"] = 12
    full["Line"][0]["Amount"] = 600.0

    result = await _update(entity_type="Estimate", data=full)

    assert result.is_error is False, result.content
    body = qbo.posts[0]
    assert set(body) == {"Id", "SyncToken", "sparse", "Line"}
    assert [ln["DetailType"] for ln in body["Line"]].count("SubTotalLineDetail") == 1
    assert [ln.get("Id") for ln in body["Line"]] == ["1", "2", None]
    assert body["Line"][0]["Amount"] == 600.0


async def test_line_list_resent_without_ids_is_rejected(qbo: FakeQBO) -> None:
    """The old habit: the whole line list again, no Ids. Appending it would
    double the estimate, so nothing is sent."""
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [
                {
                    "Amount": 600.0,
                    "DetailType": "SalesItemLineDetail",
                    "Description": "Labor - kitchen remodel",
                    "SalesItemLineDetail": {"Qty": 12, "UnitPrice": 50},
                },
            ],
        },
    )
    assert result.is_error is True
    assert "repeats line 1" in result.content
    assert qbo.posts == []


async def test_a_deleted_line_may_be_added_back(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [
                {
                    "Amount": 400.0,
                    "DetailType": "SalesItemLineDetail",
                    "Description": "Labor - kitchen remodel",
                    "SalesItemLineDetail": {"ItemRef": {"value": "7"}, "Qty": 8, "UnitPrice": 50},
                },
            ],
        },
        delete_line_ids=["1"],
    )
    assert result.is_error is False, result.content
    assert [ln.get("Id") for ln in qbo.posts[0]["Line"]] == ["2", None, None]


async def test_quantity_change_keeps_amount_consistent(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [{"Id": "1", "SalesItemLineDetail": {"Qty": 10}}],
        },
    )
    assert result.is_error is False, result.content
    assert _lines_by_id(qbo.posts[0])["1"]["Amount"] == 500.0


async def test_partial_address_keeps_the_rest_of_the_address(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={"Id": "2001", "SyncToken": "4", "ShipAddr": {"Line1": "200 Elm St"}},
    )
    assert result.is_error is False, result.content
    assert qbo.posts[0]["ShipAddr"] == {"Id": "9", "Line1": "200 Elm St", "City": "Springfield"}


async def test_new_street_address_does_not_keep_the_old_suite(qbo: FakeQBO) -> None:
    qbo.stored["ShipAddr"] = {
        "Id": "9",
        "Line1": "100 Oak St",
        "Line2": "Suite 5",
        "City": "Springfield",
        "Lat": "39.78",
        "Long": "-89.65",
    }
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "ShipAddr": {"Line1": "200 Elm St", "City": "Shelbyville"},
        },
    )
    assert result.is_error is False, result.content
    assert qbo.posts[0]["ShipAddr"] == {
        "Id": "9",
        "Line1": "200 Elm St",
        "Line2": "",
        "City": "Shelbyville",
    }


async def test_a_second_flat_fee_at_the_same_price_is_added(qbo: FakeQBO) -> None:
    """Same item, qty and price as a stored line but its own description:
    a separate charge, not a repeat."""
    qbo.stored["Line"].insert(
        2,
        {
            "Id": "3",
            "Description": "Permit fee",
            "Amount": 75.0,
            "DetailType": "SalesItemLineDetail",
            "SalesItemLineDetail": {"ItemRef": {"value": "1"}, "Qty": 1, "UnitPrice": 75},
        },
    )
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [
                {
                    "Amount": 75.0,
                    "DetailType": "SalesItemLineDetail",
                    "Description": "Disposal fee",
                    "SalesItemLineDetail": {"ItemRef": {"value": "1"}, "Qty": 1, "UnitPrice": 75},
                },
            ],
        },
    )
    assert result.is_error is False, result.content
    assert [ln.get("Id") for ln in qbo.posts[0]["Line"]] == ["1", "2", "3", None, None]


async def test_an_undescribed_line_with_the_same_numbers_is_a_repeat(qbo: FakeQBO) -> None:
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [
                {
                    "Amount": 212.5,
                    "DetailType": "SalesItemLineDetail",
                    "SalesItemLineDetail": {"Qty": 1, "UnitPrice": 212.5},
                },
            ],
        },
    )
    assert result.is_error is True
    assert "repeats line 2" in result.content
    assert qbo.posts == []


async def test_a_line_named_like_a_heading_is_not_a_repeat(qbo: FakeQBO) -> None:
    qbo.stored["Line"].insert(
        0,
        {"Id": "4", "Description": "Kitchen", "DetailType": "DescriptionOnly"},
    )
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [
                {
                    "Amount": 40.0,
                    "DetailType": "SalesItemLineDetail",
                    "Description": "Kitchen",
                    "SalesItemLineDetail": {"Qty": 1, "UnitPrice": 40},
                },
            ],
        },
    )
    assert result.is_error is False, result.content


async def test_recomputed_amount_rounds_half_up(qbo: FakeQBO) -> None:
    """2.5 x 19.99 is 49.975: QuickBooks has 49.98, float round() 49.97."""
    result = await _update(
        entity_type="Estimate",
        data={
            "Id": "2001",
            "SyncToken": "4",
            "Line": [{"Id": "2", "SalesItemLineDetail": {"Qty": 2.5, "UnitPrice": 19.99}}],
        },
    )
    assert result.is_error is False, result.content
    assert _lines_by_id(qbo.posts[0])["2"]["Amount"] == 49.98


async def test_preview_lists_only_lines_that_change(qbo: FakeQBO) -> None:
    """A line resent as stored is not a change; a ref shows its name."""
    tool = _tool("qb_update")
    assert tool.approval_policy is not None
    assert tool.approval_policy.preview_builder is not None
    text = await tool.approval_policy.preview_builder(
        {
            "entity_type": "Estimate",
            "data": {
                "Id": "2001",
                "SyncToken": "4",
                "SalesTermRef": {"value": "4", "name": "Net 15"},
                "Line": [
                    {"Id": "1", "Amount": 400.0},
                    {"Id": "2", "SalesItemLineDetail": {"Qty": 2}},
                ],
            },
        }
    )
    assert text is not None
    assert "  SalesTermRef: Net 30 (3) -> Net 15 (4)" in text
    assert "Change line 1" not in text
    assert "Change line 2" in text


# -- Approval prompt ------------------------------------------------------------


async def test_approval_preview_shows_the_effective_change(qbo: FakeQBO) -> None:
    tool = _tool("qb_update")
    assert tool.approval_policy is not None
    assert tool.approval_policy.preview_builder is not None

    text = await tool.approval_policy.preview_builder(
        {
            "entity_type": "Estimate",
            "data": {
                "Id": "2001",
                "SyncToken": "4",
                "CustomerMemo": {"value": "Updated memo"},
                "Line": [
                    {"Id": "2", "Amount": 300.0, "SalesItemLineDetail": {"UnitPrice": 300.0}},
                    {
                        "Amount": 75.0,
                        "DetailType": "SalesItemLineDetail",
                        "Description": "Permit fee",
                        "SalesItemLineDetail": {"Qty": 1, "UnitPrice": 75.0},
                    },
                ],
            },
            "delete_line_ids": ["1"],
        }
    )

    assert text is not None
    lines = text.splitlines()
    assert lines[0] == "Update Estimate #2001 in QuickBooks (#EST-2001, Test Customer)"
    assert "  CustomerMemo: Thanks for the opportunity. -> Updated memo" in lines
    assert "  Change line 2: (no description) | item Materials | qty 1 x $212.50 = $212.50" in lines
    assert "      now: (no description) | item Materials | qty 1 x $300.00 = $300.00" in lines
    assert "  Add line: Permit fee | qty 1 x $75.00 = $75.00" in lines
    assert (
        "  Remove line 1: Labor - kitchen remodel | item Services | qty 8 x $50.00 = $400.00"
        in lines
    )
    assert "  Line items total: $612.50 -> $375.00" in lines
    assert lines[-1] == "  Everything else stays as it is."
    # A preview is a read, never a write.
    assert qbo.posts == []


async def test_approval_preview_falls_back_when_the_request_would_be_refused(
    qbo: FakeQBO,
) -> None:
    tool = _tool("qb_update")
    assert tool.approval_policy is not None
    assert tool.approval_policy.preview_builder is not None
    stale = {"entity_type": "Estimate", "data": {"Id": "2001", "SyncToken": "1"}}
    assert await tool.approval_policy.preview_builder(stale) is None


def test_static_approval_text_does_not_claim_a_total() -> None:
    """Without the stored record the prompt lists the request, not a total
    computed from a partial line list."""
    tool = _tool("qb_update")
    assert tool.approval_policy is not None
    assert tool.approval_policy.description_builder is not None
    text = tool.approval_policy.description_builder(
        {
            "entity_type": "Estimate",
            "data": {
                "Id": "2001",
                "SyncToken": "4",
                "Line": [
                    {
                        "Id": "1",
                        "Amount": 500.0,
                        "DetailType": "SalesItemLineDetail",
                        "SalesItemLineDetail": {"Qty": 10},
                    }
                ],
            },
            "delete_line_ids": ["2"],
        }
    )
    assert text.splitlines()[0] == "Update Estimate #2001 in QuickBooks"
    assert "  Change line 1: Amount 500.0, Qty 10" in text
    assert "  Remove line 2" in text
    assert "for $" not in text


# -- Query rendering ---------------------------------------------------------------


def test_single_record_query_renders_every_line_detail() -> None:
    out = _format_results([_stored_estimate()])

    assert "Line (3):" in out
    assert (
        "    Id: 1 | LineNum: 1 | DetailType: SalesItemLineDetail | Amount: 400.0 | "
        'Description: "Labor - kitchen remodel" | ItemRef: Services (1) | Qty: 8 | '
        "UnitPrice: 50 | TaxCodeRef: NON"
    ) in out
    assert (
        "    Id: 2 | LineNum: 2 | DetailType: SalesItemLineDetail | Amount: 212.5 | "
        "ItemRef: Materials (5) | Qty: 1 | UnitPrice: 212.5 | TaxCodeRef: TAX | "
        "ServiceDate: 2026-09-10"
    ) in out
    assert "    DetailType: SubTotalLineDetail | Amount: 612.5" in out
    # The row's own fields stay on the row line, above the lines.
    row = out.splitlines()[1]
    assert row.startswith("- Id: 2001 | SyncToken: 4")
    assert "TotalAmt: 612.5" in row


def test_discount_line_renders_its_detail() -> None:
    row = _stored_estimate()
    row["Line"].append(
        {
            "Id": "3",
            "Amount": 20.0,
            "DetailType": "DiscountLineDetail",
            "DiscountLineDetail": {
                "PercentBased": True,
                "DiscountPercent": 10,
                "DiscountAccountRef": {"value": "86", "name": "Discounts given"},
            },
        }
    )
    out = _format_results([row])
    assert (
        "    Id: 3 | DetailType: DiscountLineDetail | Amount: 20.0 | PercentBased: True | "
        "DiscountPercent: 10 | DiscountAccountRef: Discounts given (86)"
    ) in out


def test_results_over_three_rows_stay_compact() -> None:
    out = _format_results([_stored_estimate() for _ in range(4)])
    assert "Line (" not in out
    assert "LineNum" not in out
    assert "Line: [Labor - kitchen remodel $400.00; 212.5; 612.5]" in out
