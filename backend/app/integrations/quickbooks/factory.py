"""QuickBooks Online tools for the agent."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field, field_validator

from backend.app.agent.approval import ApprovalPolicy, PermissionLevel
from backend.app.agent.tools.base import Tool, ToolErrorKind, ToolReceipt, ToolResult, ToolTags
from backend.app.agent.tools.names import ToolName
from backend.app.config import settings
from backend.app.integrations.quickbooks.merge import (
    MergedUpdate,
    UpdateRejected,
    build_update,
    describe_line,
    describe_merged_update,
)
from backend.app.integrations.quickbooks.service import (
    QuickBooksOnlineService,
    QuickBooksService,
)
from backend.app.services.oauth import (
    ReconnectRequired,
    TokenRefreshUnavailable,
    oauth_service,
)

if TYPE_CHECKING:
    from backend.app.agent.tools.registry import ToolContext

logger = logging.getLogger(__name__)

# Maximum number of rows to include in the tool response to keep context lean.
_MAX_ROWS = 50

# Entities allowed in qb_query to prevent exfiltration of sensitive data.
_QUERYABLE_ENTITIES = {
    "INVOICE",
    "ESTIMATE",
    "CUSTOMER",
    "ITEM",
    "PAYMENT",
    "BILL",
    "VENDOR",
    "SALESRECEIPT",
    "CREDITMEMO",
    "PURCHASEORDER",
    "TIMEACTIVITY",
    "DEPOSIT",
    "TRANSFER",
    "JOURNALENTRY",
}

# Human-readable labels for queryable entities.
_ENTITY_LABELS: dict[str, str] = {
    "INVOICE": "invoices",
    "ESTIMATE": "estimates",
    "CUSTOMER": "customers",
    "ITEM": "items",
    "PAYMENT": "payments",
    "BILL": "bills",
    "VENDOR": "vendors",
    "SALESRECEIPT": "sales receipts",
    "CREDITMEMO": "credit memos",
    "PURCHASEORDER": "purchase orders",
    "TIMEACTIVITY": "time entries",
    "DEPOSIT": "deposits",
    "TRANSFER": "transfers",
    "JOURNALENTRY": "journal entries",
}

# Entity types that qb_create is allowed to create.
_CREATABLE_ENTITIES = {"Customer", "Estimate", "Invoice", "Item"}

# Keys that identify an existing record; never valid on a create.
_RECORD_IDENTITY_KEYS = frozenset({"Id", "SyncToken", "sparse", "MetaData", "domain"})

# Entity types that qb_update is allowed to update.
_UPDATABLE_ENTITIES = {"Customer", "Estimate", "Invoice", "Item"}

# Entity types that qb_send is allowed to send via email.
_SENDABLE_ENTITIES = {"Invoice", "Estimate"}

# Serialization key shared by every QuickBooks write tool.
#
# The agent runs all approved tool calls from one model turn concurrently.
# The model routinely emits ``qb_create`` and ``qb_send`` in the same turn,
# predicting the id the create will return. Without a shared group the send
# races the create and QBO answers ``code=610 Object Not Found`` because the
# transaction genuinely does not exist yet. These tools all mutate the same
# QBO company and read each other's ids, so they share one static key rather
# than a per-entity resolver: ordering across entity types matters too (an
# Invoice line can reference an Item created in the same turn).
_QB_WRITE_CONCURRENCY_GROUP = "quickbooks_write"

# Intuit fault codes that mean "the entity is not there", as opposed to a
# transient backend failure. Surfaced as NOT_FOUND so the model looks the id
# up again instead of blindly retrying a "service unavailable".
_INTUIT_NOT_FOUND_CODES = {"610"}

# Intuit fault codes for a query QuickBooks refused to run: 4000 "Error
# parsing query" (QueryParserError) and 4001 "Invalid query"
# (QueryValidationError, e.g. "Property BillAddr not found for Entity
# Customer"). The query is wrong, not the service. Surfaced as VALIDATION so
# the model fixes the query and retries instead of telling the user
# QuickBooks is down.
_INTUIT_INVALID_QUERY_CODES = {"4000", "4001"}

_INVALID_QUERY_HINT = (
    "QuickBooks is working; this query is invalid. Fix it and retry: use SELECT * "
    "instead of naming fields, or simplify the WHERE clause."
)


class QBQueryParams(BaseModel):
    """Parameters for the qb_query tool."""

    query: str = Field(
        description=(
            "SELECT only, e.g. SELECT * FROM Invoice MAXRESULTS 20. Use SELECT * for "
            "addresses, emails, and phones. Enum values often "
            "gotten wrong:\n"
            "  Estimate.TxnStatus = 'Pending' | 'Accepted' | 'Closed' | 'Rejected'\n"
            "  Invoice.EmailStatus = 'NotSet' | 'NeedToSend' | 'EmailSent'\n"
            "Filter Bill / Invoice / Estimate by the numeric Balance, not a status enum."
        )
    )


_TXNSTATUS_VALID_BY_ENTITY: dict[str, tuple[str, ...]] = {
    "Estimate": ("Pending", "Accepted", "Closed", "Rejected"),
}
"""Known string-enum value sets per entity. Used to enrich 400 errors so a
hallucinated value like ``TxnStatus='In Progress'`` comes back with the
right list instead of the model retrying the same wrong guess."""


def _format_intuit_fault(exc: httpx.HTTPStatusError, *, entity: str | None = None) -> str:
    """Convert a QBO HTTPStatusError into a model-readable error message.

    Intuit returns a structured ``Fault.Error[]`` payload that the LLM
    has trouble parsing reliably. We pull out ``Message`` + ``Detail``
    + ``code`` and concatenate them. When ``Detail`` mentions a value
    that maps to a known enum (``TxnStatus`` on Estimate so far), we
    append the valid set so the next turn does not retry the same bad
    guess.

    Falls back to the raw JSON (then the raw exception string) when the
    response is not JSON or is shaped differently than expected.
    """
    raw_body: Any
    try:
        raw_body = exc.response.json()
    except Exception:
        return f"HTTP {exc.response.status_code} from QuickBooks: {exc.response.text or exc!s}"

    fault = (raw_body or {}).get("Fault") if isinstance(raw_body, dict) else None
    errors = (fault or {}).get("Error") if isinstance(fault, dict) else None
    if not isinstance(errors, list) or not errors:
        return f"HTTP {exc.response.status_code} from QuickBooks: {json.dumps(raw_body)[:500]}"

    parts: list[str] = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        code = err.get("code", "")
        message = (err.get("Message") or "").strip()
        detail = (err.get("Detail") or "").strip()
        # Most-specific first: Detail usually contains the bad value;
        # Message is the category.
        line = " | ".join(p for p in (f"code={code}" if code else "", message, detail) if p)
        if line:
            parts.append(line)

    body = "; ".join(parts) or json.dumps(raw_body)[:500]

    # Enrich when the failure looks like a hallucinated enum value. The
    # set is small and curated so the false-positive rate stays low.
    hint = _enum_hint(body, entity)
    if hint:
        body = f"{body}\nHint: {hint}"
    return body


def _intuit_fault_codes(exc: httpx.HTTPStatusError) -> set[str]:
    """Collect the ``Fault.Error[].code`` values from a QBO error response.

    Returns an empty set when the body is not a recognizable Intuit fault.
    """
    try:
        raw_body = exc.response.json()
    except Exception:
        return set()
    fault = (raw_body or {}).get("Fault") if isinstance(raw_body, dict) else None
    errors = (fault or {}).get("Error") if isinstance(fault, dict) else None
    if not isinstance(errors, list):
        return set()
    return {str(err["code"]) for err in errors if isinstance(err, dict) and err.get("code")}


def _log_tool_failure(exc: Exception, message: str, *args: object) -> None:
    """Log a failed QuickBooks call; a dead connection is expected, not a crash.

    ``ReconnectRequired`` is the user's connection lapsing, already handled
    (token retired, user notified), so it gets a warning without a traceback
    rather than an ERROR that pages as a code failure.
    """
    if isinstance(exc, ReconnectRequired):
        logger.warning(message + ": reconnect required: %s", *args, exc)
    else:
        logger.exception(message, *args)


def _refresh_unavailable_result(action: str) -> ToolResult:
    """A 401 left standing because a peer held the refresh lock.

    That says nothing about the grant and retrying shortly should succeed,
    so it is a retryable SERVICE error, logged without a traceback.
    """
    logger.warning("QuickBooks token refresh could not run during %s", action)
    return ToolResult(
        content=f"QuickBooks could not renew its access while trying to {action}.",
        is_error=True,
        error_kind=ToolErrorKind.SERVICE,
        hint="Another refresh of this connection was in progress. Retry this call shortly.",
    )


def _fault_error_kind(exc: Exception) -> ToolErrorKind:
    """Classify a QBO failure for the LLM error hint.

    A 610 is a 4xx about a missing object, not an outage. Reporting it as
    SERVICE tells the model "temporarily unavailable, try a different
    approach", which is the wrong instruction: the id is wrong (or not
    visible yet) and the model should look it up rather than retry blind.

    A dead connection (the refresh token expired or was revoked, or QBO
    still refuses a freshly refreshed token) arrives as
    ``ReconnectRequired``: AUTH, so the model tells the user to reconnect
    rather than wait out an outage.

    A 400 carrying 4000 or 4001 is a query QuickBooks could not parse or
    validate: VALIDATION. Matching on the Intuit code rather than the bare
    status keeps a 400 with no ``Fault`` body out of this bucket. Everything
    else (429, 5xx, a token endpoint 5xx, network errors, other faults) stays
    SERVICE.
    """
    if isinstance(exc, ReconnectRequired):
        return ToolErrorKind.AUTH
    if not isinstance(exc, httpx.HTTPStatusError):
        return ToolErrorKind.SERVICE
    codes = _intuit_fault_codes(exc)
    if codes & _INTUIT_NOT_FOUND_CODES:
        return ToolErrorKind.NOT_FOUND
    if exc.response.status_code == 400 and codes & _INTUIT_INVALID_QUERY_CODES:
        return ToolErrorKind.VALIDATION
    return ToolErrorKind.SERVICE


def _enum_hint(error_body: str, entity: str | None) -> str:
    """Map a QBO error blob to a one-line hint when it looks enum-shaped.

    Intentionally conservative: matches on substrings the LLM is likely
    to also see. Returns empty string when no hint applies.
    """
    if not entity:
        return ""
    lowered = error_body.lower()
    if "txnstatus" in lowered and entity in _TXNSTATUS_VALID_BY_ENTITY:
        valid = ", ".join(_TXNSTATUS_VALID_BY_ENTITY[entity])
        return f"Valid TxnStatus for {entity}: {valid}"
    return ""


def _coerce_data_to_dict(value: Any) -> Any:
    """Parse JSON-encoded strings into dicts so the LLM can pass either shape.

    The LLM occasionally over-quotes deeply nested QBO payloads and emits
    `data` as a JSON string rather than a JSON object. Accept both forms
    on the first round to avoid a wasted retry.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "data must be a JSON object or a JSON-encoded object string; "
                f"could not parse string as JSON: {exc.msg}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"data must be a JSON object; got a JSON-encoded {type(parsed).__name__}"
            )
        return parsed
    return value


class QBCreateParams(BaseModel):
    """Parameters for the qb_create tool."""

    entity_type: str = Field(description="Customer, Estimate, Invoice, or Item.")
    data: dict[str, Any] = Field(description="QBO API payload as a JSON object.")

    _coerce_data = field_validator("data", mode="before")(_coerce_data_to_dict)


class QBUpdateParams(BaseModel):
    """Parameters for the qb_update tool."""

    entity_type: str = Field(description="Customer, Estimate, Invoice, or Item.")
    data: dict[str, Any] = Field(
        description=(
            "Only what changes, plus Id and SyncToken from a qb_query of this record. "
            "Fields you omit are kept. Line entries: with a line's Id, only the keys "
            "you send change on that line; without Id, the line is added. Lines you "
            "omit are kept."
        )
    )
    delete_line_ids: list[str] = Field(
        default_factory=list,
        description="Ids of existing lines to remove. The only way a line is removed.",
    )

    _coerce_data = field_validator("data", mode="before")(_coerce_data_to_dict)

    @field_validator("delete_line_ids", mode="before")
    @classmethod
    def _stringify_line_ids(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [str(v) for v in value]
        return value


class QBSendParams(BaseModel):
    """Parameters for the qb_send tool."""

    entity_type: str = Field(
        description="Invoice or Estimate.",
        default="Invoice",
    )
    entity_id: str = Field(description="QuickBooks entity ID (numeric).")
    email: str = Field(
        description="Recipient email address.",
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    )


# Results with more rows than this render compact (see ``_format_results``).
# One to three rows are what a lookup by Id, DocNumber or name returns, and
# qb_update needs that record whole, SyncToken included.
_COMPACT_ABOVE_ROWS = 3

# Compact rows cut these free-text fields to this many characters. They are
# the bulk of a SELECT * list row and none of them identifies anything.
_COMPACT_TEXT_CHARS = 90
_COMPACT_SHORTENED_FIELDS = frozenset(
    {"Line", "CustomerMemo", "PrivateNote", "Notes", "Description", "PurchaseDesc"}
)

# Compact rows omit these. Each is company-wide configuration or delivery
# plumbing that repeats on every row of a SELECT * and answers no question the
# agent is asked about a list. Ids, SyncToken, DocNumber, refs, amounts,
# dates, statuses and emails are never named here.
_COMPACT_DROPPED_FIELDS = frozenset(
    {
        "DeliveryInfo",
        "ShipFromAddr",
        "CurrencyRef",
        "ApplyTaxAfterDiscount",
        "FreeFormAddress",
        "GlobalTaxCalculation",
        "V4IDPseudonym",
        "PrintOnCheckName",
        "ClientEntityId",
        "IsProject",
        "BillWithParent",
        "PreferredDeliveryMethod",
        "AllowIPNPayment",
        "AllowOnlinePayment",
        "AllowOnlineCreditCardPayment",
        "AllowOnlineACHPayment",
    }
)

_COMPACT_NOTE = (
    f"(More than {_COMPACT_ABOVE_ROWS} rows, so each is compact: filler fields "
    f"dropped, long text cut at {_COMPACT_TEXT_CHARS} chars and marked [+N chars]. "
    "Before quoting cut text or calling qb_update, query the record alone: "
    "WHERE Id = '<Id>'.)"
)


def _render_value(key: str, val: Any) -> str | None:
    """Render one QBO field value, or None when it has nothing to show."""
    if isinstance(val, dict):
        if "name" in val or "value" in val:
            name = val.get("name", "")
            ref_val = val.get("value", "")
            if name and ref_val:
                return f"{name} ({ref_val})"
            return str(name or ref_val) if name or ref_val else None
        if "Address" in val:
            return str(val["Address"])
        if "FreeFormNumber" in val:
            return str(val["FreeFormNumber"])
        if "URI" in val:
            return str(val["URI"])
        if any(k in val for k in ("Line1", "City", "PostalCode")):
            addr_bits = [
                str(val[k])
                for k in ("Line1", "Line2", "City", "CountrySubDivisionCode", "PostalCode")
                if val.get(k)
            ]
            return ", ".join(addr_bits) if addr_bits else None
        # Fail loud on unknown dict shapes so future QBO fields surface
        # (verbose but visible) rather than disappearing.
        return json.dumps(val)
    if isinstance(val, list):
        if key == "Line" and val:
            items = []
            for item in val:
                if not isinstance(item, dict):
                    continue
                desc = item.get("Description", "")
                amt = item.get("Amount")
                items.append(f"{desc} ${amt:,.2f}" if amt is not None and desc else str(amt))
            return f"[{'; '.join(items)}]"
        return json.dumps(val)
    return str(val)


def _is_empty(key: str, val: Any) -> bool:
    """True for values that say nothing: null, blank, empty containers, and a
    CustomField list whose definitions carry no value."""
    if val is None or val == "" or val == [] or val == {}:
        return True
    return (
        key == "CustomField"
        and isinstance(val, list)
        and all(
            isinstance(item, dict) and not any(k.endswith("Value") for k in item) for item in val
        )
    )


# Line keys rendered first, in this order, by ``_render_line``.
_LINE_HEAD_KEYS = ("Id", "LineNum", "DetailType", "Amount", "Description")


def _render_line(line: dict[str, Any]) -> str:
    """Render one transaction line with everything qb_update needs.

    Id, LineNum, DetailType, Amount and Description first, then every field
    of the line's detail object (ItemRef, Qty, UnitPrice, TaxCodeRef,
    ServiceDate, DiscountPercent, ...), then any other line field. Nothing
    is dropped, so each line can be told apart and changed by its Id.
    """
    parts: list[str] = []
    for key in _LINE_HEAD_KEYS:
        if key not in line or line[key] is None:
            continue
        val = line[key]
        if key == "Description":
            parts.append(f"Description: {json.dumps(val)}")
        else:
            parts.append(f"{key}: {val}")
    detail_key = line.get("DetailType")
    for key, val in line.items():
        if key in _LINE_HEAD_KEYS:
            continue
        if key == detail_key and isinstance(val, dict):
            for dkey, dval in val.items():
                text = _render_value(dkey, dval)
                if text is not None and not _is_empty(dkey, dval):
                    parts.append(f"{dkey}: {text}")
            continue
        text = _render_value(key, val)
        if text is not None and not _is_empty(key, val):
            parts.append(f"{key}: {text}")
    return " | ".join(parts)


def _format_row(row: dict[str, Any], *, compact: bool) -> str:
    parts: list[str] = []
    line_block: list[str] = []
    for key, val in row.items():
        if key in ("domain", "sparse", "MetaData"):
            continue
        if not compact and key == "Line" and isinstance(val, list):
            # Whole records feed qb_update, so every line keeps its Id and
            # detail. Rendered after the row so its fields stay on one line.
            items = [item for item in val if isinstance(item, dict)]
            line_block.append(f"  Line ({len(items)}):")
            line_block.extend(f"    {_render_line(item)}" for item in items)
            continue
        if compact:
            if key in _COMPACT_DROPPED_FIELDS or _is_empty(key, val):
                continue
            if key == "TxnTaxDetail" and isinstance(val, dict):
                # Keep the total; the per-rate breakdown is for the full record.
                if "TotalTax" in val:
                    parts.append(f"TotalTax: {val['TotalTax']}")
                continue
        text = _render_value(key, val)
        if text is None:
            continue
        if compact and key in _COMPACT_SHORTENED_FIELDS and len(text) > _COMPACT_TEXT_CHARS:
            cut = len(text) - _COMPACT_TEXT_CHARS
            text = f"{text[:_COMPACT_TEXT_CHARS]}... [+{cut} chars]"
        parts.append(f"{key}: {text}")
    return "\n".join(["- " + " | ".join(parts), *line_block])


def _format_results(rows: list[dict[str, Any]]) -> str:
    """Format QBO query results into a readable string for the LLM.

    Up to ``_COMPACT_ABOVE_ROWS`` rows render every field. Larger results are
    lists the agent scans, so each row drops ``_COMPACT_DROPPED_FIELDS`` and
    empty values and cuts long free text, with a note on how to get a record
    whole. Everything that identifies or prices a record stays on every row.
    """
    if not rows:
        return "Query returned 0 results."

    compact = len(rows) > _COMPACT_ABOVE_ROWS
    lines = [f"Query returned {len(rows)} result(s):"]
    lines.extend(_format_row(row, compact=compact) for row in rows[:_MAX_ROWS])

    if len(rows) > _MAX_ROWS:
        lines.append(f"... and {len(rows) - _MAX_ROWS} more (add MAXRESULTS to narrow)")
    if compact:
        lines.append(_COMPACT_NOTE)

    return "\n".join(lines)


def _extract_query_entity(args: dict[str, Any]) -> str | None:
    """Extract the entity name from a QBO query string (e.g. 'Invoice' from 'SELECT * FROM Invoice')."""
    query = str(args.get("query", ""))
    match = re.search(r"\bFROM\s+(\w+)", query, re.IGNORECASE)
    return match.group(1) if match else None


def _describe_qb_query(args: dict[str, Any]) -> str:
    """Build a human-readable description for a QuickBooks query."""
    query = str(args.get("query", ""))
    match = re.search(r"\bFROM\s+(\w+)", query, re.IGNORECASE)
    if not match:
        return "Look up data in QuickBooks"
    entity = match.group(1).upper()
    label = _ENTITY_LABELS.get(entity, match.group(1).lower() + "s")
    return f"Look up {label} in QuickBooks"


def _extract_entity_type(args: dict[str, Any]) -> str | None:
    """Extract the entity_type argument."""
    return str(args["entity_type"]) if args.get("entity_type") else None


# Entity types whose payload carries a ``Line`` array we render in the
# approval prompt. Customer payloads do not have line items, so they
# stay on the short legacy form.
_LINE_ITEMIZED_ENTITIES: frozenset[str] = frozenset({"Invoice", "Estimate"})


def _qb_approval_header(verb: str, entity_type: str, entity_id: Any, total: float | None) -> str:
    """Build the first line of a qb_create / qb_update approval prompt.

    Update headers include ``#{Id}`` (when available) so audit-log review
    can trace which row was edited; Create headers do not because the
    id only exists after QBO assigns one on the POST response. When
    ``total`` is set, the header appends ``for ${total:,.2f}``.

    Thousands separator matches ``_receipt_target`` so the same invoice
    formats consistently across the approval prompt and the post-write
    ToolReceipt.
    """
    pieces = [f"{verb} {entity_type}"]
    if verb == "Update" and entity_id:
        pieces[0] = f"{pieces[0]} #{entity_id}"
    pieces.append("in QuickBooks")
    if total is not None:
        pieces.append(f"for ${total:,.2f}")
    return " ".join(pieces)


def _format_qb_write_approval_description(verb: str, args: dict[str, Any]) -> str:
    """Render a multi-line approval prompt that names every line item.

    The default builder used to print only ``"Create Invoice in
    QuickBooks"``, which let a billing action slip through with the
    wrong per-line math because the user could not see quantity, unit
    price, or total before approving. Surfacing each line as
    ``qty x $unit = $line_total`` (and the grand total) makes the
    approval prompt the last place a mistake can be caught before
    QuickBooks stores it. Mirrors the AppFolio fix in #1292.

    Falls back to the short form (``"<verb> <entity_type> in
    QuickBooks"``) for entity types without line items (Customer) and
    on any malformed payload, so the prompt never crashes; the agent's
    own typed validation will reject a bad call after approval anyway.

    ``verb`` is "Create" or "Update". For Update the header includes
    the entity Id so an admin reviewing audit logs can trace which row
    was edited.

    Trusts that ``args`` came from ``QBCreateParams`` / ``QBUpdateParams``
    (Pydantic-validated upstream), so ``args["data"]`` is a dict;
    line-shape tolerance below is for the LLM's freeform payload
    *inside* ``data``, not for the param envelope itself.
    """
    entity_type = str(args.get("entity_type") or "entity")
    data: dict[str, Any] = args.get("data") or {}
    entity_id = data.get("Id")

    short_header = _qb_approval_header(verb, entity_type, entity_id, total=None)

    if entity_type not in _LINE_ITEMIZED_ENTITIES:
        return short_header

    lines_raw = data.get("Line")
    if not isinstance(lines_raw, list) or not lines_raw:
        return short_header

    parsed: list[tuple[str, float | None, float | None, float]] = []
    grand_total = 0.0
    for line in lines_raw:
        if not isinstance(line, dict):
            return short_header
        # QBO computes the subtotal line itself; counting a copied one would
        # show double the real total. A discount line's Amount is the size of
        # the discount, so it reduces the total rather than adding to it.
        if line.get("DetailType") == "SubTotalLineDetail" or "SubTotalLineDetail" in line:
            continue
        try:
            amount = float(line.get("Amount", 0) or 0)
        except (TypeError, ValueError):
            return short_header
        if line.get("DetailType") == "DiscountLineDetail" or "DiscountLineDetail" in line:
            amount = -abs(amount)
        description = str(line.get("Description") or "(no description)")
        detail = line.get("SalesItemLineDetail")
        qty: float | None = None
        unit_price: float | None = None
        if isinstance(detail, dict):
            try:
                if detail.get("Qty") is not None:
                    qty = float(detail["Qty"])
                if detail.get("UnitPrice") is not None:
                    unit_price = float(detail["UnitPrice"])
            except (TypeError, ValueError):
                qty = None
                unit_price = None
        grand_total += amount
        parsed.append((description, qty, unit_price, amount))

    if not parsed:
        return short_header

    rendered = [_qb_approval_header(verb, entity_type, entity_id, total=grand_total)]
    for idx, (description, qty, unit_price, amount) in enumerate(parsed, start=1):
        short_desc = description if len(description) <= 80 else description[:77] + "..."
        if qty is not None and unit_price is not None:
            # ``:g`` drops trailing ``.0`` so ``5.0`` reads as ``5`` while
            # leaving genuine fractional quantities (e.g. ``1.5``) intact.
            rendered.append(
                f"  {idx}. {short_desc} | qty {qty:g} x ${unit_price:,.2f} = ${amount:,.2f}"
            )
        else:
            rendered.append(f"  {idx}. {short_desc} | ${amount:,.2f}")
    return "\n".join(rendered)


def _describe_line_keys(line: dict[str, Any]) -> str:
    """The keys a line change sets, detail keys included. Only these change;
    the rest of the stored line is kept."""
    sent: list[str] = []
    for key, val in line.items():
        if key in ("Id", "DetailType"):
            continue
        if key.endswith("LineDetail") and isinstance(val, dict):
            sent.extend(f"{k} {_render_value(k, v)}" for k, v in val.items())
        else:
            sent.append(f"{key} {_render_value(key, val)}")
    return ", ".join(sent) or "no change"


def _describe_qb_update_request(args: dict[str, Any]) -> str:
    """Approval text for qb_update from the request alone.

    Used when the live preview (``preview_qb_update``) cannot read the
    record. Lists what the request sets, changes, adds and removes; with no
    stored values to compare against, it names no totals.
    """
    entity_type = str(args.get("entity_type") or "entity")
    data: dict[str, Any] = args.get("data") or {}
    header = _qb_approval_header("Update", entity_type, data.get("Id"), total=None)
    rows: list[str] = []
    for key, val in data.items():
        if key in ("Id", "SyncToken", "sparse", "domain", "MetaData", "Line"):
            continue
        text = _render_value(key, val)
        rows.append(f"  Set {key}: {text if text is not None else '(empty)'}")
    lines = data.get("Line")
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, dict):
                continue
            if line.get("Id") not in (None, ""):
                rows.append(f"  Change line {line['Id']}: {_describe_line_keys(line)}")
            else:
                rows.append(f"  Add line: {describe_line(line)}")
    for lid in args.get("delete_line_ids") or []:
        rows.append(f"  Remove line {lid}")
    if not rows:
        return header
    rows.append("  Everything else stays as it is.")
    return "\n".join([header, *rows])


# Entity types that have a public QBO web UI page we can deep-link to.
_WEB_LINKABLE_ENTITIES: dict[str, str] = {
    "Invoice": "invoice",
    "Estimate": "estimate",
    "Customer": "customerdetail",
}


def _build_qbo_url(qb_service: QuickBooksService, entity_type: str, entity_id: str) -> str | None:
    """Build a deep link into the QuickBooks Online web UI for an entity.

    Returns ``None`` for entity types without a known web UI path. The link
    always points at the real entity ID returned by the API, so no LLM text
    is involved.
    """
    path = _WEB_LINKABLE_ENTITIES.get(entity_type)
    if not path or not entity_id or entity_id == "?":
        return None
    qbo_service = qb_service if isinstance(qb_service, QuickBooksOnlineService) else None
    if qbo_service is None:
        return None
    is_prod = "sandbox" not in qbo_service._api_base
    host = "app.qbo.intuit.com" if is_prod else "app.sandbox.qbo.intuit.com"
    if entity_type == "Customer":
        return f"https://{host}/app/{path}?nameId={entity_id}"
    return f"https://{host}/app/{path}?txnId={entity_id}"


def _receipt_target(entity_type: str, result: dict[str, Any]) -> str:
    """Best-effort human-readable target string for a QB entity result."""
    total = result.get("TotalAmt")
    name = result.get("DisplayName") or ""
    doc_num = result.get("DocNumber") or ""
    entity_id = result.get("Id", "")
    if entity_type in ("Invoice", "Estimate"):
        customer_ref = result.get("CustomerRef") or {}
        customer = customer_ref.get("name", "") if isinstance(customer_ref, dict) else ""
        bits: list[str] = []
        if customer:
            bits.append(customer)
        if total is not None:
            bits.append(f"${total:,.2f}")
        if not bits and doc_num:
            bits.append(f"#{doc_num}")
        if not bits:
            bits.append(f"ID {entity_id}")
        return ", ".join(bits)
    if entity_type == "Customer":
        return name or f"ID {entity_id}"
    if entity_type == "Item":
        item_name = result.get("Name") or ""
        return item_name or f"ID {entity_id}"
    return name or doc_num or f"ID {entity_id}"


def _extract_send_email(args: dict[str, Any]) -> str | None:
    """Extract the email recipient from qb_send arguments."""
    return str(args["email"]) if args.get("email") else None


def create_quickbooks_tools(
    qb_service: QuickBooksService,
) -> list[Tool]:
    """Create QuickBooks-related tools for the agent."""

    # Entity ids created during this agent run, keyed by entity type. The
    # tool list is rebuilt per inbound message, so this is scoped to one
    # user turn (spanning every LLM round of that turn) and never leaks
    # between messages or users.
    #
    # qb_send consults it to refuse an id the model predicted rather than
    # read back from a qb_create result. A predicted id that happens to
    # match a different live transaction would email one customer another
    # customer's invoice, so the guess is rejected, not retried.
    created_ids: dict[str, set[str]] = {}

    async def qb_query(query: str) -> ToolResult:
        """Run a read-only query against QuickBooks Online."""
        import re as _re

        normalized = query.strip()
        if not normalized.upper().startswith("SELECT"):
            return ToolResult(
                content="Only SELECT queries are supported.",
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )

        entity_match = _re.search(r"\bFROM\s+(\w+)", normalized, _re.IGNORECASE)
        if not entity_match:
            return ToolResult(
                content="Query must include a FROM clause (e.g. SELECT * FROM Invoice).",
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )
        if entity_match.group(1).upper() not in _QUERYABLE_ENTITIES:
            return ToolResult(
                content=f"Querying '{entity_match.group(1)}' is not allowed. "
                f"Allowed entities: {', '.join(sorted(_QUERYABLE_ENTITIES))}",
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )

        # Resolve the entity name from the SELECT clause once so the
        # error formatter can attach the right enum-hint set when QBO
        # returns a 400.
        entity_name = entity_match.group(1).capitalize()

        try:
            rows = await qb_service.query(normalized)
        except Exception as exc:
            if isinstance(exc, TokenRefreshUnavailable):
                return _refresh_unavailable_result("query")
            error_kind = _fault_error_kind(exc)
            if error_kind is ToolErrorKind.VALIDATION:
                # The model's query was wrong; no stack trace needed.
                logger.warning("QuickBooks rejected query: %s", exc)
            else:
                _log_tool_failure(exc, "QuickBooks query failed")
            if isinstance(exc, httpx.HTTPStatusError):
                error_str = _format_intuit_fault(exc, entity=entity_name)
            else:
                error_str = str(exc)
            return ToolResult(
                content=f"QuickBooks query error: {error_str}",
                is_error=True,
                error_kind=error_kind,
                hint=_INVALID_QUERY_HINT if error_kind is ToolErrorKind.VALIDATION else "",
            )

        return ToolResult(content=_format_results(rows))

    async def qb_create(entity_type: str, data: dict[str, Any]) -> ToolResult:
        """Create an entity in QuickBooks Online."""
        if entity_type not in _CREATABLE_ENTITIES:
            return ToolResult(
                content=f"Creating '{entity_type}' is not allowed. "
                f"Allowed: {', '.join(sorted(_CREATABLE_ENTITIES))}",
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )

        # A payload copied from a queried record carries that record's
        # identity. QBO treats a create body with Id and SyncToken as an
        # update, so a whole estimate pasted into an Invoice create would
        # overwrite whichever invoice shares its Id. Strip the identity.
        data = {k: v for k, v in data.items() if k not in _RECORD_IDENTITY_KEYS}

        lines = data.get("Line")
        if isinstance(lines, list):
            # Lines copied from a queried record (estimate to invoice) carry
            # that record's line Ids, which name lines of another transaction.
            data = {
                **data,
                "Line": [
                    {k: v for k, v in ln.items() if k != "Id"} if isinstance(ln, dict) else ln
                    for ln in lines
                ],
            }

        try:
            result = await qb_service.create_entity(entity_type, data)
        except Exception as exc:
            if isinstance(exc, TokenRefreshUnavailable):
                return _refresh_unavailable_result(f"create a {entity_type}")
            _log_tool_failure(exc, "QB create %s failed", entity_type)
            if isinstance(exc, httpx.HTTPStatusError):
                error_str = _format_intuit_fault(exc, entity=entity_type)
            else:
                error_str = str(exc)
            return ToolResult(
                content=f"Failed to create {entity_type}: {error_str}",
                is_error=True,
                error_kind=_fault_error_kind(exc),
            )

        entity_id = result.get("Id", "?")
        doc_num = result.get("DocNumber", "")
        total = result.get("TotalAmt")
        display_name = result.get("DisplayName", "")
        item_name = result.get("Name", "")

        if entity_id != "?":
            created_ids.setdefault(entity_type, set()).add(str(entity_id))

        # LLM-facing content stays terse and data-only so the model has no
        # receipt-shaped phrasing to bullet-point back to the user. The
        # auto-appended ToolReceipt is the canonical user-facing rendering
        # (see regression test guarding ``qb_send`` content).
        parts = ["ok", f"Id: {entity_id}"]
        if doc_num:
            parts.append(f"DocNumber: {doc_num}")
        if total is not None:
            parts.append(f"Total: ${total:.2f}")
        if display_name:
            parts.append(f"Name: {display_name}")
        if not display_name and item_name:
            parts.append(f"Name: {item_name}")

        return ToolResult(
            content=" | ".join(parts),
            receipt=ToolReceipt(
                action=f"Created QuickBooks {entity_type.lower()} for",
                target=_receipt_target(entity_type, result),
                url=_build_qbo_url(qb_service, entity_type, str(entity_id)),
            ),
        )

    async def _merge_with_stored(
        entity_type: str, data: dict[str, Any], delete_line_ids: list[str]
    ) -> MergedUpdate:
        """Read the record as QuickBooks holds it now and merge the change in."""
        entity_id = str(data.get("Id") or "").strip()
        if not entity_id:
            raise UpdateRejected(
                "data.Id is missing. Query the record first and pass its Id and SyncToken."
            )
        current = await qb_service.read_entity(entity_type, entity_id)
        return build_update(entity_type, current, data, delete_line_ids)

    async def qb_update(
        entity_type: str,
        data: dict[str, Any],
        delete_line_ids: list[str] | None = None,
    ) -> ToolResult:
        """Update an existing entity in QuickBooks Online.

        Never sends the agent's payload as-is. The record is re-read and the
        change merged into it (see ``merge.build_update``), so a field or a
        line the agent left out is kept rather than nulled or deleted.
        """
        if entity_type not in _UPDATABLE_ENTITIES:
            return ToolResult(
                content=f"Updating '{entity_type}' is not allowed. "
                f"Allowed: {', '.join(sorted(_UPDATABLE_ENTITIES))}",
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )

        try:
            merged = await _merge_with_stored(entity_type, data, delete_line_ids or [])
            result = await qb_service.update_entity(entity_type, merged.payload)
        except UpdateRejected as exc:
            return ToolResult(
                content=f"Did not update {entity_type}: {exc}",
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )
        except Exception as exc:
            if isinstance(exc, TokenRefreshUnavailable):
                return _refresh_unavailable_result(f"update a {entity_type}")
            _log_tool_failure(exc, "QB update %s failed", entity_type)
            if isinstance(exc, httpx.HTTPStatusError):
                error_str = _format_intuit_fault(exc, entity=entity_type)
            else:
                error_str = str(exc)
            return ToolResult(
                content=f"Failed to update {entity_type}: {error_str}",
                is_error=True,
                error_kind=_fault_error_kind(exc),
            )

        entity_id = result.get("Id", "?")
        doc_num = result.get("DocNumber", "")
        total = result.get("TotalAmt")
        display_name = result.get("DisplayName", "")
        item_name = result.get("Name", "")
        sync_token = result.get("SyncToken")

        parts = ["ok", f"Id: {entity_id}"]
        if sync_token is not None:
            # A second update of this record needs the new token.
            parts.append(f"SyncToken: {sync_token}")
        if doc_num:
            parts.append(f"DocNumber: {doc_num}")
        if total is not None:
            parts.append(f"Total: ${total:.2f}")
        if display_name:
            parts.append(f"Name: {display_name}")
        if not display_name and item_name:
            parts.append(f"Name: {item_name}")

        return ToolResult(
            content=" | ".join(parts),
            receipt=ToolReceipt(
                action=f"Updated QuickBooks {entity_type.lower()} for",
                target=_receipt_target(entity_type, result),
                url=_build_qbo_url(qb_service, entity_type, str(entity_id)),
            ),
        )

    async def preview_qb_update(args: dict[str, Any]) -> str | None:
        """Approval text for qb_update against the stored record.

        Shows each changed field and line as it is and as it will be, and
        the line total before and after. ``None`` (a failed read, a request
        the tool will refuse) falls back to ``_describe_qb_update_request``.
        """
        entity_type = str(args.get("entity_type") or "")
        data = args.get("data")
        if entity_type not in _UPDATABLE_ENTITIES or not isinstance(data, dict):
            return None
        try:
            merged = await _merge_with_stored(
                entity_type, data, list(args.get("delete_line_ids") or [])
            )
        except Exception:
            logger.info("qb_update preview unavailable", exc_info=True)
            return None
        return describe_merged_update(entity_type, merged)

    async def qb_send(entity_type: str, entity_id: str, email: str) -> ToolResult:
        """Send an invoice or estimate via QuickBooks email."""
        if entity_type not in _SENDABLE_ENTITIES:
            return ToolResult(
                content=f"Sending '{entity_type}' is not allowed. "
                f"Allowed: {', '.join(sorted(_SENDABLE_ENTITIES))}",
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )

        # When this turn created entities of the same type, the only ids the
        # model can legitimately send are the ones qb_create handed back.
        # Anything else is a prediction, and sending a predicted id delivers
        # some other customer's transaction to this recipient.
        turn_created = created_ids.get(entity_type)
        if turn_created and entity_id.strip() not in turn_created:
            known = ", ".join(sorted(turn_created))
            logger.warning(
                "qb_send_unverified_entity_id tool=qb_send entity_type=%s requested=%s created=%s",
                entity_type,
                entity_id,
                known,
            )
            return ToolResult(
                content=(
                    f"Refusing to send {entity_type} {entity_id}: this turn created "
                    f"{entity_type} {known}, and {entity_id} is not among them. "
                    f"Send one of the ids returned by qb_create, or look the "
                    f"intended {entity_type.lower()} up with qb_query first."
                ),
                is_error=True,
                error_kind=ToolErrorKind.VALIDATION,
            )

        try:
            await qb_service.send_entity_email(entity_type, entity_id, email)
        except Exception as exc:
            if isinstance(exc, TokenRefreshUnavailable):
                return _refresh_unavailable_result(f"send a {entity_type}")
            _log_tool_failure(exc, "QB send %s email failed", entity_type)
            if isinstance(exc, httpx.HTTPStatusError):
                error_str = _format_intuit_fault(exc, entity=entity_type)
            else:
                error_str = str(exc)
            return ToolResult(
                content=f"Failed to send {entity_type.lower()}: {error_str}",
                is_error=True,
                error_kind=_fault_error_kind(exc),
            )

        # Drop the verb + recipient from the LLM-facing content: that
        # phrasing was getting bullet-pointed in prose right before the
        # auto-receipt rendered the same action structurally, producing
        # a double-bullet for one underlying call. Bug observed in
        # production 2026-05-13 on a contractor's invoice send.
        return ToolResult(
            content=f"ok | {entity_type} Id: {entity_id}",
            receipt=ToolReceipt(
                action=f"Emailed QuickBooks {entity_type.lower()} to",
                target=email,
                url=_build_qbo_url(qb_service, entity_type, entity_id),
            ),
        )

    return [
        Tool(
            name=ToolName.QB_QUERY,
            tags={ToolTags.READ_ONLY},
            description=(
                "Run a read-only QuickBooks Online query (SQL-like SELECT ... FROM "
                "<Entity>) for invoices, estimates, customers, items, payments, and "
                "more; the QuickBooks skill has the syntax and entities. Results over "
                f"{_COMPACT_ABOVE_ROWS} rows are compact (filler dropped, long text "
                "cut); query WHERE Id = '<Id>' for a whole record."
            ),
            function=qb_query,
            params_model=QBQueryParams,
            approval_policy=ApprovalPolicy(
                default_level=PermissionLevel.ASK,
                resource_extractor=_extract_query_entity,
                description_builder=_describe_qb_query,
            ),
        ),
        Tool(
            name=ToolName.QB_CREATE,
            description=(
                "Create a Customer, Estimate, Invoice, or Item in QuickBooks Online "
                "from a QBO API payload built as the QuickBooks skill describes."
            ),
            function=qb_create,
            params_model=QBCreateParams,
            concurrency_group=_QB_WRITE_CONCURRENCY_GROUP,
            approval_policy=ApprovalPolicy(
                default_level=PermissionLevel.ASK,
                resource_extractor=_extract_entity_type,
                description_builder=lambda args: _format_qb_write_approval_description(
                    "Create", args
                ),
            ),
        ),
        Tool(
            name=ToolName.QB_UPDATE,
            description=(
                "Update a Customer, Estimate, Invoice, or Item in QuickBooks Online. "
                "The tool merges data into the stored record; see the QuickBooks skill."
            ),
            function=qb_update,
            params_model=QBUpdateParams,
            concurrency_group=_QB_WRITE_CONCURRENCY_GROUP,
            approval_policy=ApprovalPolicy(
                default_level=PermissionLevel.ASK,
                resource_extractor=_extract_entity_type,
                description_builder=_describe_qb_update_request,
                preview_builder=preview_qb_update,
            ),
        ),
        Tool(
            name=ToolName.QB_SEND,
            description=(
                "Email an invoice or estimate to a customer through QuickBooks. Pass "
                "the Id returned by qb_create or qb_query; never predict one. Confirm "
                "the email address with the user first."
            ),
            function=qb_send,
            params_model=QBSendParams,
            concurrency_group=_QB_WRITE_CONCURRENCY_GROUP,
            approval_policy=ApprovalPolicy(
                default_level=PermissionLevel.ASK,
                resource_extractor=_extract_send_email,
                description_builder=lambda args: (
                    f"Send {args.get('entity_type', 'entity')} "
                    f"to {args.get('email', 'recipient')} via QuickBooks"
                ),
                # Lets the user grant a blanket "always all" approval covering
                # every recipient instead of approving each email separately.
                resource_noun="recipients",
            ),
        ),
    ]


async def _get_quickbooks_service_for_user(user_id: str) -> QuickBooksService | None:
    """Build a QuickBooks service using OAuth tokens for the given user."""
    token = await oauth_service.get_valid_token(user_id, "quickbooks")
    if not (token and token.access_token and token.realm_id):
        return None
    return QuickBooksOnlineService(
        realm_id=token.realm_id,
        access_token=token.access_token,
        environment=settings.quickbooks_environment,
        refresh_access_token=oauth_service.build_rejected_token_refresher(user_id, "quickbooks"),
    )


async def _quickbooks_auth_check(ctx: ToolContext) -> str | None:
    """Check whether QuickBooks is configured and the user has authenticated.

    Returns ``None`` when ready, or a reason string when auth is missing.
    Returns ``None`` (not a reason) when the integration is not configured
    at all (admin has not set credentials), so it stays completely hidden.
    """
    if not settings.quickbooks_client_id or not settings.quickbooks_client_secret:
        return None
    token = await oauth_service.load_token(ctx.user.id, "quickbooks")
    if token and token.access_token and token.realm_id:
        return None
    return (
        "QuickBooks is not connected. "
        "Use manage_integration(action='connect', target='quickbooks') "
        "to generate a connection link for the user."
    )


async def _quickbooks_factory(ctx: ToolContext) -> list[Tool]:
    """Factory for QuickBooks tools, used by the registry."""
    if not settings.quickbooks_client_id or not settings.quickbooks_client_secret:
        return []
    qb_service = await _get_quickbooks_service_for_user(ctx.user.id)
    if qb_service is None:
        return []
    return create_quickbooks_tools(qb_service)


def _register() -> None:
    from backend.app.agent.tools.registry import SubToolInfo, default_registry

    default_registry.register(
        "quickbooks",
        _quickbooks_factory,
        core=False,
        summary=(
            "Query, create, and manage QuickBooks Online entities: "
            "invoices, estimates, customers, and more"
        ),
        display_name="QuickBooks Online",
        dashboard_description="Query, create, and manage QuickBooks Online entities",
        dashboard_group="Integrations",
        dashboard_group_order=2,
        sub_tools=[
            SubToolInfo(
                ToolName.QB_QUERY,
                "Run read-only queries against QuickBooks Online",
                default_permission="ask",
            ),
            SubToolInfo(
                ToolName.QB_CREATE, "Create entities in QuickBooks", default_permission="ask"
            ),
            SubToolInfo(
                ToolName.QB_UPDATE,
                "Update existing entities in QuickBooks",
                default_permission="ask",
            ),
            SubToolInfo(
                ToolName.QB_SEND,
                "Send invoices or estimates via QuickBooks email",
                default_permission="ask",
            ),
        ],
        auth_check=_quickbooks_auth_check,
    )


_register()
