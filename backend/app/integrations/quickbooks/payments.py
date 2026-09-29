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


@dataclass(frozen=True)
class PaymentPlan:
    """What a checked payment will do: the invoices it pays and where it lands."""

    applied: list[AppliedInvoice]
    deposit_account_id: str
    deposit_account_name: str


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


async def _deposit_account(qb_service: QuickBooksService, data: dict[str, Any]) -> tuple[str, str]:
    """The bank account the payment is deposited to, as ``(Id, Name)``.

    Payments go straight to a bank account rather than Undeposited Funds,
    which only makes sense for someone who batches deposits or matches a
    bank feed. A named account must be an active bank account. With none
    named, the company's only bank account is used; with several, the user
    has to say which. Account balances never reach the model: only names
    and Ids are returned, and ``Account`` stays out of ``qb_query``.
    """
    wanted = _ref_value(data.get("DepositToAccountRef"))
    if wanted:
        if not wanted.isdigit():
            raise PaymentRejected('DepositToAccountRef must be {"value": "<numeric account Id>"}.')
        account = await qb_service.read_entity("Account", wanted)
        if account.get("AccountType") != "Bank" or account.get("Active") is False:
            raise PaymentRejected(f"Account {wanted} is not an active bank account.")
        return wanted, str(account.get("Name") or wanted)
    banks = await qb_service.query(
        "SELECT * FROM Account WHERE AccountType = 'Bank' AND Active = true MAXRESULTS 20"
    )
    if len(banks) == 1:
        return str(banks[0].get("Id")), str(banks[0].get("Name") or "")
    if not banks:
        raise PaymentRejected(
            "QuickBooks has no active bank account to deposit the payment into. "
            "Tell the user; one has to be set up in QuickBooks first."
        )
    options = "; ".join(f"Id {b.get('Id')}: {b.get('Name')}" for b in banks)
    raise PaymentRejected(
        f"QuickBooks has several bank accounts ({options}). Ask the user which one "
        'this payment went into, then pass DepositToAccountRef {"value": "<Id>"}.'
    )


async def check_payment(qb_service: QuickBooksService, data: dict[str, Any]) -> PaymentPlan:
    """Validate a Payment payload against the invoices it pays.

    Returns each invoice with its balance before and after, and the bank
    account the payment lands in, for the approval prompt and the write.
    Raises ``PaymentRejected`` when the payment names no customer, does not
    add up, pays an invoice of another customer, pays more than an
    invoice's open balance, has no bank account to land in, or reuses a
    reference number already recorded for this customer.
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
    account_id, account_name = await _deposit_account(qb_service, data)
    return PaymentPlan(
        applied=applied, deposit_account_id=account_id, deposit_account_name=account_name
    )


def describe_payment(data: dict[str, Any], plan: PaymentPlan | None) -> str:
    """Approval text: who paid, how, where it lands, and each invoice's new balance.

    Without *plan* (the check could not run), lists the invoices by the Ids
    the request names and states no balances.
    """
    applied = plan.applied if plan else None
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
    if plan:
        rows.append(f"  Deposit to: {plan.deposit_account_name}")
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
