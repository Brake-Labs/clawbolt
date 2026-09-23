"""Turn a qb_update request into a payload that cannot drop data by omission.

QuickBooks Online has two update modes. A full update sets every writable
field the body omits to NULL. A sparse update (``"sparse": true``) leaves
omitted fields alone, but ``Line`` is one field: a body that carries it
replaces the whole array, so any existing line it leaves out is deleted, a
line without an ``Id`` is created, and a line sent with its ``Id`` but only
some of its detail loses the rest.

The agent writes a change, not a record. ``build_update`` reads the record
as QuickBooks holds it now and merges the change into it:

* Top-level fields the agent sends replace the stored value. Everything else
  is left out of a sparse body, so QuickBooks keeps it.
* A line with an ``Id`` is merged onto the stored line of that ``Id``: its
  keys replace, and its ``*LineDetail`` object is merged key by key, so
  changing ``Qty`` keeps ``ItemRef`` and ``TaxCodeRef``.
* A line without an ``Id`` is appended.
* A line is removed only when its ``Id`` is in ``delete_line_ids``.

Whenever lines are touched, the whole merged array is sent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

# Entities QuickBooks documents a sparse update for. Item has only a full
# update, so an Item change is sent as the stored record with the change
# applied, which carries every writable field and so nulls nothing.
SPARSE_ENTITIES = frozenset({"Customer", "Estimate", "Invoice"})

# Entities whose ``Line`` array this module merges.
LINE_ENTITIES = frozenset({"Estimate", "Invoice"})

# Keys of the agent's payload that never go to QuickBooks as changes.
_CONTROL_KEYS = frozenset({"Id", "SyncToken", "sparse", "domain", "MetaData"})

# QuickBooks computes this line from the others; it is never a line item.
_SUBTOTAL = "SubTotalLineDetail"


class UpdateRejected(ValueError):
    """The request cannot be applied safely; the message tells the agent why."""

    def __init__(self, message: str, *, conflict: bool = False) -> None:
        super().__init__(message)
        self.conflict = conflict


@dataclass
class MergedUpdate:
    """What ``build_update`` decided.

    ``payload`` is the body to POST. ``before`` and ``after`` are the whole
    record as stored and as it will be, for the approval prompt.
    """

    payload: dict[str, Any]
    before: dict[str, Any]
    after: dict[str, Any]
    changed_fields: list[str]
    changed_line_ids: list[str]
    added_lines: list[dict[str, Any]]
    removed_lines: list[dict[str, Any]]


def _line_id(line: dict[str, Any]) -> str | None:
    raw = line.get("Id")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _is_subtotal(line: dict[str, Any]) -> bool:
    return line.get("DetailType") == _SUBTOTAL


def _merge_line(stored: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(stored)
    new_type = change.get("DetailType")
    old_type = stored.get("DetailType")
    if new_type and old_type and new_type != old_type:
        # A line switching type drops the old type's detail object.
        merged.pop(str(old_type), None)
    for key, value in change.items():
        existing = merged.get(key)
        if key.endswith("LineDetail") and isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = {**existing, **copy.deepcopy(value)}
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_lines(
    entity_type: str,
    entity_id: str,
    stored_lines: list[dict[str, Any]],
    requested: Any,
    delete_line_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (merged lines, changed ids, added lines, removed lines)."""
    if requested is None:
        requested = []
    if not isinstance(requested, list) or not all(isinstance(ln, dict) for ln in requested):
        raise UpdateRejected("Line must be a list of line objects.")

    stored_by_id: dict[str, dict[str, Any]] = {}
    for line in stored_lines:
        lid = _line_id(line)
        if lid is not None:
            stored_by_id[lid] = line
    known = ", ".join(stored_by_id) or "none"

    deletes = [str(d).strip() for d in delete_line_ids]
    unknown_deletes = [d for d in deletes if d not in stored_by_id]
    if unknown_deletes:
        raise UpdateRejected(
            f"delete_line_ids {', '.join(unknown_deletes)} are not lines on "
            f"{entity_type} {entity_id}. Its line Ids are: {known}."
        )

    changes: dict[str, dict[str, Any]] = {}
    added: list[dict[str, Any]] = []
    for line in requested:
        lid = _line_id(line)
        if lid is None:
            added.append(copy.deepcopy(line))
            continue
        if lid not in stored_by_id:
            raise UpdateRejected(
                f"Line Id {lid} is not on {entity_type} {entity_id}. Its line Ids "
                f"are: {known}. Omit Id to add a new line."
            )
        if lid in deletes:
            raise UpdateRejected(f"Line Id {lid} is both changed and in delete_line_ids.")
        if lid in changes:
            raise UpdateRejected(f"Line Id {lid} appears twice in Line.")
        changes[lid] = line

    merged: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for line in stored_lines:
        lid = _line_id(line)
        if lid is not None and lid in deletes:
            removed.append(line)
        elif lid is not None and lid in changes:
            merged.append(_merge_line(line, changes[lid]))
        else:
            merged.append(copy.deepcopy(line))

    # New lines go before QuickBooks' trailing subtotal line, where the UI
    # would put them.
    insert_at = len(merged)
    while insert_at > 0 and _is_subtotal(merged[insert_at - 1]):
        insert_at -= 1
    merged[insert_at:insert_at] = added

    if not any(not _is_subtotal(line) for line in merged):
        raise UpdateRejected(
            f"That would leave {entity_type} {entity_id} with no line items. "
            "QuickBooks needs at least one."
        )

    changed_ids = [lid for lid, change in changes.items() if _line_changes(change)]
    return merged, changed_ids, added, removed


def _line_changes(change: dict[str, Any]) -> bool:
    """True when a line entry carries anything besides its Id."""
    return any(key != "Id" for key in change)


def build_update(
    entity_type: str,
    current: dict[str, Any],
    changes: dict[str, Any],
    delete_line_ids: list[str] | None = None,
) -> MergedUpdate:
    """Merge the agent's *changes* into *current* and build the body to send.

    Raises ``UpdateRejected`` when the request is not safe to apply: no
    SyncToken, a SyncToken older than the stored one (the record changed
    since the agent read it), a line Id that is not on the record, or a
    request that would leave no line items.
    """
    delete_line_ids = list(delete_line_ids or [])
    entity_id = str(current.get("Id", changes.get("Id", "?")))

    sent_token = changes.get("SyncToken")
    stored_token = current.get("SyncToken")
    if sent_token is None or str(sent_token).strip() == "":
        raise UpdateRejected(
            f"SyncToken is missing. Query the record first "
            f"(SELECT * FROM {entity_type} WHERE Id = '{entity_id}') and pass its SyncToken."
        )
    if stored_token is not None and str(sent_token).strip() != str(stored_token):
        raise UpdateRejected(
            f"{entity_type} {entity_id} changed in QuickBooks since it was read "
            f"(SyncToken is now {stored_token}, the update carried {sent_token}). "
            f"Query it again (WHERE Id = '{entity_id}'), check the change still "
            "applies, then retry with the new SyncToken.",
            conflict=True,
        )

    touches_lines = "Line" in changes or bool(delete_line_ids)
    if touches_lines and entity_type not in LINE_ENTITIES:
        raise UpdateRejected(f"{entity_type} has no line items to change.")

    field_changes = {
        key: copy.deepcopy(value)
        for key, value in changes.items()
        if key not in _CONTROL_KEYS and key != "Line"
    }

    after = copy.deepcopy(current)
    after.update(copy.deepcopy(field_changes))
    changed_ids: list[str] = []
    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    merged_lines: list[dict[str, Any]] | None = None
    if touches_lines:
        stored_lines = [ln for ln in current.get("Line") or [] if isinstance(ln, dict)]
        merged_lines, changed_ids, added, removed = _merge_lines(
            entity_type, entity_id, stored_lines, changes.get("Line"), delete_line_ids
        )
        after["Line"] = merged_lines

    if entity_type in SPARSE_ENTITIES:
        payload: dict[str, Any] = {
            "Id": entity_id,
            "SyncToken": str(stored_token if stored_token is not None else sent_token),
            "sparse": True,
            **field_changes,
        }
        if merged_lines is not None:
            payload["Line"] = merged_lines
    else:
        payload = copy.deepcopy(after)
        payload.pop("MetaData", None)
        payload.pop("domain", None)
        payload["sparse"] = False

    return MergedUpdate(
        payload=payload,
        before=current,
        after=after,
        changed_fields=list(field_changes),
        changed_line_ids=changed_ids,
        added_lines=added,
        removed_lines=removed,
    )


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _short(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _field_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("value", "Address", "FreeFormNumber", "name"):
            if key in value and value[key] not in (None, ""):
                return _short(str(value[key]))
        return _short(str(value))
    if value is None or value == "":
        return "(empty)"
    return _short(str(value))


def describe_line(line: dict[str, Any]) -> str:
    """One approval-prompt row for a line: description, item, qty x price, amount."""
    detail_type = str(line.get("DetailType") or "")
    detail = line.get(detail_type) if detail_type else None
    detail = detail if isinstance(detail, dict) else {}
    bits = [_short(str(line.get("Description") or "(no description)"))]
    item = detail.get("ItemRef")
    if isinstance(item, dict) and (item.get("name") or item.get("value")):
        bits.append(f"item {item.get('name') or item.get('value')}")
    qty, price = detail.get("Qty"), detail.get("UnitPrice")
    amount = _money(line.get("Amount", 0))
    if qty is not None and price is not None:
        try:
            bits.append(f"qty {float(qty):g} x {_money(price)} = {amount}")
        except (TypeError, ValueError):
            bits.append(amount)
    else:
        bits.append(amount)
    return " | ".join(bits)


def lines_subtotal(lines: Any) -> float | None:
    """Sum of line amounts before tax, discounts and QuickBooks' subtotal row."""
    if not isinstance(lines, list):
        return None
    total = 0.0
    for line in lines:
        if not isinstance(line, dict) or line.get("DetailType") in (
            _SUBTOTAL,
            "DiscountLineDetail",
        ):
            continue
        try:
            total += float(line.get("Amount") or 0)
        except (TypeError, ValueError):
            return None
    return total


def describe_merged_update(entity_type: str, merged: MergedUpdate) -> str:
    """Approval prompt naming every change against the stored record."""
    before, after = merged.before, merged.after
    entity_id = after.get("Id", "?")
    header = f"Update {entity_type} #{entity_id} in QuickBooks"
    doc = before.get("DocNumber")
    customer = before.get("CustomerRef")
    who = customer.get("name") if isinstance(customer, dict) else None
    label = ", ".join(str(x) for x in (f"#{doc}" if doc else None, who) if x)
    if label:
        header = f"{header} ({label})"

    rows: list[str] = []
    for key in merged.changed_fields:
        rows.append(f"  {key}: {_field_text(before.get(key))} -> {_field_text(after.get(key))}")

    before_lines = {
        str(ln.get("Id")): ln for ln in before.get("Line") or [] if isinstance(ln, dict)
    }
    after_lines = {str(ln.get("Id")): ln for ln in after.get("Line") or [] if isinstance(ln, dict)}
    for lid in merged.changed_line_ids:
        old, new = before_lines.get(lid), after_lines.get(lid)
        if old is None or new is None:
            continue
        rows.append(f"  Change line {lid}: {describe_line(old)}")
        rows.append(f"      now: {describe_line(new)}")
    for line in merged.added_lines:
        rows.append(f"  Add line: {describe_line(line)}")
    for line in merged.removed_lines:
        rows.append(f"  Remove line {line.get('Id')}: {describe_line(line)}")

    if merged.changed_line_ids or merged.added_lines or merged.removed_lines:
        old_total = lines_subtotal(before.get("Line"))
        new_total = lines_subtotal(after.get("Line"))
        if old_total is not None and new_total is not None:
            rows.append(f"  Line items total: {_money(old_total)} -> {_money(new_total)}")

    if not rows:
        rows.append("  No field or line changes.")
    rows.append("  Everything else stays as it is.")
    return "\n".join([header, *rows])
