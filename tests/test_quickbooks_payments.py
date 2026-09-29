"""Recording a customer payment against invoices through qb_create.

A payment on the wrong invoice, recorded twice, or for more than was owed
misstates the books and nothing flags it until reconciliation. These tests
pin each refusal, and that nothing is written when one fires.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.integrations.quickbooks.factory import create_quickbooks_tools
from tests.test_quickbooks_write import FakeQBService


class PaymentsFakeQB(FakeQBService):
    """Fake with stored invoices and a payment history to query."""

    def __init__(self) -> None:
        super().__init__()
        self.payments: list[dict[str, Any]] = []
        self.banks: list[dict[str, Any]] = [{"Id": "35", "Name": "Business Checking"}]
        self.queries: list[str] = []
        self.records[("Account", "35")] = {
            "Id": "35",
            "Name": "Business Checking",
            "AccountType": "Bank",
        }
        self.records[("Account", "4")] = {
            "Id": "4",
            "Name": "Undeposited Funds",
            "AccountType": "Other Current Asset",
        }
        self.records[("Invoice", "644")] = {
            "Id": "644",
            "DocNumber": "1044",
            "CustomerRef": {"value": "20", "name": "Test Customer"},
            "TotalAmt": 4824.53,
            "Balance": 4824.53,
        }
        self.records[("Invoice", "650")] = {
            "Id": "650",
            "DocNumber": "1050",
            "CustomerRef": {"value": "31", "name": "Other Customer"},
            "TotalAmt": 500.0,
            "Balance": 500.0,
        }
        self.records[("Invoice", "660")] = {
            "Id": "660",
            "DocNumber": "1060",
            "CustomerRef": {"value": "20", "name": "Test Customer"},
            "TotalAmt": 300.0,
            "Balance": 0,
        }

    async def query(self, query_str: str) -> list[dict[str, Any]]:
        self.queries.append(query_str)
        return self.banks if "FROM Account" in query_str else self.payments


def _tool(svc: FakeQBService, name: str = "qb_create") -> Any:
    return next(t for t in create_quickbooks_tools(svc) if t.name == name)


def _payment(
    amount: float = 4824.53,
    invoice_id: str = "644",
    customer_id: str = "20",
    ref: str = "1234",
) -> dict[str, Any]:
    return {
        "CustomerRef": {"value": customer_id},
        "TotalAmt": amount,
        "TxnDate": "2026-09-29",
        "PaymentMethodRef": {"value": "2", "name": "Check"},
        "PaymentRefNum": ref,
        "Line": [{"Amount": amount, "LinkedTxn": [{"TxnId": invoice_id, "TxnType": "Invoice"}]}],
    }


async def test_records_a_payment_that_pays_the_invoice_in_full() -> None:
    svc = PaymentsFakeQB()
    result = await _tool(svc).function(entity_type="Payment", data=_payment())

    assert result.is_error is False
    assert "Invoice 1044 balance: $0.00" in result.content
    deposit = {"DepositToAccountRef": {"value": "35", "name": "Business Checking"}}
    assert svc.created == [("Payment", {**_payment(), **deposit})]
    assert result.receipt is not None
    assert result.receipt.action == "Recorded QuickBooks payment from"


async def test_a_partial_payment_reports_what_is_still_owed() -> None:
    svc = PaymentsFakeQB()
    result = await _tool(svc).function(entity_type="Payment", data=_payment(amount=1000.0))

    assert result.is_error is False
    assert "Invoice 1044 balance: $3,824.53" in result.content


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (_payment(invoice_id="650"), "belongs to customer 31"),
        (_payment(amount=5000.0), "more than invoice 1044's open balance"),
        (_payment(invoice_id="660", amount=300.0), "already paid in full"),
        ({**_payment(), "TotalAmt": 5000.0}, "must equal the line amounts"),
        ({**_payment(), "Line": []}, "needs a Line for each invoice"),
        (
            {
                **_payment(),
                "Line": [{"Amount": 4824.53, "LinkedTxn": [{"TxnId": "7", "TxnType": "Estimate"}]}],
            },
            "against invoices only",
        ),
        ({**_payment(), "CustomerRef": {"value": "Test Customer"}}, "numeric customer Id"),
    ],
)
async def test_refuses_a_payment_that_would_misstate_the_books(
    data: dict[str, Any], reason: str
) -> None:
    svc = PaymentsFakeQB()
    result = await _tool(svc).function(entity_type="Payment", data=data)

    assert result.is_error is True
    assert reason in result.content
    assert svc.created == []


async def test_refuses_a_reference_number_already_recorded_for_the_customer() -> None:
    """Regression guard: recording the same check twice counts the money twice."""
    svc = PaymentsFakeQB()
    svc.payments = [
        {"Id": "88", "PaymentRefNum": "1234", "TotalAmt": 4824.53, "TxnDate": "2026-09-28"}
    ]
    result = await _tool(svc).function(entity_type="Payment", data=_payment())

    assert result.is_error is True
    assert "Payment 88 with reference 1234 is already recorded" in result.content
    assert svc.created == []
    assert svc.queries[0] == (
        "SELECT * FROM Payment WHERE CustomerRef = '20' ORDERBY TxnDate DESC MAXRESULTS 100"
    )


async def test_approval_preview_shows_each_invoice_balance_before_and_after() -> None:
    policy = _tool(PaymentsFakeQB()).approval_policy
    assert policy is not None and policy.preview_builder is not None

    preview = await policy.preview_builder({"entity_type": "Payment", "data": _payment()})

    assert preview == (
        "Record payment in QuickBooks for $4,824.53\n"
        "  From: Test Customer\n"
        "  received 2026-09-29, Check, ref 1234\n"
        "  Deposit to: Business Checking\n"
        "  Invoice 1044: $4,824.53 applied, balance $4,824.53 -> $0.00 (paid in full)"
    )


async def test_approval_falls_back_to_the_request_when_the_check_fails() -> None:
    policy = _tool(PaymentsFakeQB()).approval_policy
    assert policy is not None and policy.preview_builder is not None
    args = {"entity_type": "Payment", "data": _payment(invoice_id="650")}

    assert await policy.preview_builder(args) is None
    assert policy.description_builder is not None
    assert "Invoice Id 650" in policy.description_builder(args)


async def test_other_entity_types_keep_their_approval_text() -> None:
    policy = _tool(PaymentsFakeQB()).approval_policy
    assert policy is not None and policy.preview_builder is not None
    args = {"entity_type": "Customer", "data": {"DisplayName": "New Co"}}

    assert await policy.preview_builder(args) is None
    assert policy.description_builder is not None
    assert policy.description_builder(args) == "Create Customer in QuickBooks"


async def test_payment_methods_are_queryable() -> None:
    svc = PaymentsFakeQB()
    result = await _tool(svc, "qb_query").function(query="SELECT * FROM PaymentMethod")
    assert result.is_error is False


async def test_several_bank_accounts_means_asking_which() -> None:
    svc = PaymentsFakeQB()
    svc.banks = [{"Id": "35", "Name": "Business Checking"}, {"Id": "36", "Name": "Savings"}]
    result = await _tool(svc).function(entity_type="Payment", data=_payment())

    assert result.is_error is True
    assert "Id 35: Business Checking; Id 36: Savings" in result.content
    assert svc.created == []


async def test_a_named_bank_account_is_used() -> None:
    svc = PaymentsFakeQB()
    svc.banks = [{"Id": "35", "Name": "Business Checking"}, {"Id": "36", "Name": "Savings"}]
    data = {**_payment(), "DepositToAccountRef": {"value": "35"}}
    result = await _tool(svc).function(entity_type="Payment", data=data)

    assert result.is_error is False
    assert svc.created[0][1]["DepositToAccountRef"] == {"value": "35", "name": "Business Checking"}


async def test_refuses_to_deposit_anywhere_but_a_bank_account() -> None:
    svc = PaymentsFakeQB()
    data = {**_payment(), "DepositToAccountRef": {"value": "4"}}
    result = await _tool(svc).function(entity_type="Payment", data=data)

    assert result.is_error is True
    assert "not an active bank account" in result.content
    assert svc.created == []


async def test_no_bank_account_means_nothing_is_recorded() -> None:
    svc = PaymentsFakeQB()
    svc.banks = []
    result = await _tool(svc).function(entity_type="Payment", data=_payment())

    assert result.is_error is True
    assert "no active bank account" in result.content
    assert svc.created == []
