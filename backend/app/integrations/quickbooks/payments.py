"""Checks for recording a customer payment against invoices.

QuickBooks has no "mark paid" switch. An invoice is paid when a Payment
record is linked to it and the payment's amounts bring its Balance to zero.
A payment recorded against the wrong invoice, twice, or for more than was
owed misstates the books in a way nobody notices until reconciliation, so
every rule here is checked against the stored invoices before the write,
not left to the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.integrations.quickbooks.service import QuickBooksService

# Half a cent: amounts are compared in dollars as QuickBooks returns them.
_CENT_TOLERANCE = 0.005


class PaymentRejected(Exception):
    """The payment would misstate the books; the message says how to fix it."""


@dataclass(frozen=True)
class AppliedInvoice:
    """One invoice a payment pays down, as QuickBooks holds it now."""

    invoice_id: str
    doc_number: str
    customer_name: str
    balance: float
    amount: float

    @property
    def balance_after(self) -> float:
        return round(self.balance - self.amount, 2)


def _ref_value(ref: Any) -> str:
    if isinstance(ref, dict):
        return str(ref.get("value") or "").strip()
    return ""


def _money(value: Any, what: str) -> float:
    try:
        amount = round(float(value), 2)
    except (TypeError, ValueError):
        raise PaymentRejected(f"{what} must be a number, got {value!r}.") from None
    if amount <= 0:
        raise PaymentRejected(f"{what} must be more than $0.00.")
    return amount


def _invoice_links(data: dict[str, Any]) -> list[tuple[str, float]]:
    """``(invoice_id, amount)`` for each payment line, in order."""
    lines = data.get("Line")
    if not isinstance(lines, list) or not lines:
        raise PaymentRejected(
            "A payment needs a Line for each invoice it pays, with Amount and "
            'LinkedTxn [{"TxnId": "<invoice Id>", "TxnType": "Invoice"}].'
        )
    links: list[tuple[str, float]] = []
    for line in lines:
        linked = line.get("LinkedTxn") if isinstance(line, dict) else None
        if not isinstance(linked, list) or len(linked) != 1 or not isinstance(linked[0], dict):
            raise PaymentRejected(
                "Each payment line links exactly one invoice: "
                '"LinkedTxn": [{"TxnId": "<invoice Id>", "TxnType": "Invoice"}].'
            )
        txn = linked[0]
        if txn.get("TxnType") != "Invoice":
            raise PaymentRejected(
                f"Payments are recorded against invoices only, not {txn.get('TxnType')!r}."
            )
        invoice_id = str(txn.get("TxnId") or "").strip()
        if not invoice_id.isdigit():
            raise PaymentRejected(f"LinkedTxn TxnId must be a numeric invoice Id: {invoice_id!r}.")
        links.append((invoice_id, _money(line.get("Amount"), f"Amount for invoice {invoice_id}")))
    ids = [invoice_id for invoice_id, _ in links]
    if len(set(ids)) != len(ids):
        raise PaymentRejected("Each invoice appears on one payment line only.")
    return links


async def check_payment(
    qb_service: QuickBooksService, data: dict[str, Any]
) -> list[AppliedInvoice]:
    """Validate a Payment payload against the invoices it pays.

    Returns each invoice with its balance before and after, for the approval
    prompt and the result. Raises ``PaymentRejected`` when the payment names
    no customer, does not add up, pays an invoice of another customer, pays
    more than an invoice's open balance, or reuses a reference number
    already recorded for this customer.
    """
    customer_id = _ref_value(data.get("CustomerRef"))
    if not customer_id.isdigit():
        raise PaymentRejected('CustomerRef must be {"value": "<numeric customer Id>"}.')
    total = _money(data.get("TotalAmt"), "TotalAmt")
    links = _invoice_links(data)
    applied_total = round(sum(amount for _, amount in links), 2)
    if abs(applied_total - total) > _CENT_TOLERANCE:
        raise PaymentRejected(
            f"TotalAmt ${total:,.2f} must equal the line amounts, which add to "
            f"${applied_total:,.2f}. A payment that leaves money unapplied sits "
            "as a customer credit; ask the user where the difference goes."
        )

    applied: list[AppliedInvoice] = []
    for invoice_id, amount in links:
        invoice = await qb_service.read_entity("Invoice", invoice_id)
        invoice_customer = _ref_value(invoice.get("CustomerRef"))
        customer_name = str((invoice.get("CustomerRef") or {}).get("name") or "")
        doc = str(invoice.get("DocNumber") or invoice_id)
        if invoice_customer != customer_id:
            raise PaymentRejected(
                f"Invoice {doc} belongs to customer {invoice_customer or '(unknown)'}"
                f"{f' ({customer_name})' if customer_name else ''}, not customer "
                f"{customer_id}. Check which customer paid."
            )
        try:
            balance = round(float(invoice.get("Balance") or 0), 2)
        except (TypeError, ValueError):
            balance = 0.0
        if balance <= 0:
            raise PaymentRejected(f"Invoice {doc} is already paid in full (balance $0.00).")
        if amount - balance > _CENT_TOLERANCE:
            raise PaymentRejected(
                f"${amount:,.2f} is more than invoice {doc}'s open balance of "
                f"${balance:,.2f}. Ask the user whether the rest belongs to another invoice."
            )
        applied.append(
            AppliedInvoice(
                invoice_id=invoice_id,
                doc_number=doc,
                customer_name=customer_name,
                balance=balance,
                amount=amount,
            )
        )

    ref_num = str(data.get("PaymentRefNum") or "").strip()
    if ref_num:
        recorded = await qb_service.query(
            f"SELECT * FROM Payment WHERE CustomerRef = '{customer_id}' "
            "ORDERBY TxnDate DESC MAXRESULTS 100"
        )
        for payment in recorded:
            if str(payment.get("PaymentRefNum") or "").strip() == ref_num:
                raise PaymentRejected(
                    f"Payment {payment.get('Id')} with reference {ref_num} is already "
                    f"recorded for this customer (${float(payment.get('TotalAmt') or 0):,.2f} "
                    f"on {payment.get('TxnDate')}). Recording it again counts the money twice."
                )
    return applied


def describe_payment(data: dict[str, Any], applied: list[AppliedInvoice] | None) -> str:
    """Approval text: who paid, how, and what each invoice's balance becomes.

    Without *applied* (the invoices could not be read), lists the invoices by
    the Ids the request names and states no balances.
    """
    try:
        total = f" for ${float(data.get('TotalAmt') or 0):,.2f}"
    except (TypeError, ValueError):
        total = ""
    rows = [f"Record payment in QuickBooks{total}"]
    customer = next((a.customer_name for a in applied or [] if a.customer_name), "")
    if not customer:
        customer = (
            str((data.get("CustomerRef") or {}).get("name") or "")
            if isinstance(data.get("CustomerRef"), dict)
            else ""
        )
    if customer:
        rows.append(f"  From: {customer}")
    how: list[str] = []
    if data.get("TxnDate"):
        how.append(f"received {data['TxnDate']}")
    method = data.get("PaymentMethodRef")
    if isinstance(method, dict) and method.get("name"):
        how.append(str(method["name"]))
    if data.get("PaymentRefNum"):
        how.append(f"ref {data['PaymentRefNum']}")
    if how:
        rows.append("  " + ", ".join(how))
    if applied:
        for invoice in applied:
            status = "paid in full" if invoice.balance_after <= 0 else "partly paid"
            rows.append(
                f"  Invoice {invoice.doc_number}: ${invoice.amount:,.2f} applied, balance "
                f"${invoice.balance:,.2f} -> ${max(invoice.balance_after, 0):,.2f} ({status})"
            )
    else:
        for line in data.get("Line") or []:
            if not isinstance(line, dict):
                continue
            linked = line.get("LinkedTxn") or [{}]
            txn = linked[0] if isinstance(linked, list) and linked else {}
            rows.append(f"  Invoice Id {txn.get('TxnId', '?')}: ${line.get('Amount', '?')} applied")
    return "\n".join(rows)
