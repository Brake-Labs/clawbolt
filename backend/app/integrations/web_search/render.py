"""Generic rendering of provider result records as plain text.

Field-name agnostic on purpose. The renderer walks whatever record the provider
returned and prints every leaf it finds, so a provider that starts sending a new
field needs no change here. An earlier design mapped results onto a fixed
four-field model and silently dropped Brave's ``product.price``, which was the
one field a materials estimate actually needed.

Nothing here drops information. There is no field allowlist, no denylist, and no
truncation: every record this module is handed is rendered whole, and no value
is ever cut short. The only skip is a key whose value is null or an empty
string, which carries nothing the model could use.

Shaping belongs to the provider, which knows its own field names. Brave's
(``brave.py``) strips ``<strong>`` markup, drops a short denylist of
presentation-only keys (image and favicon URLs, site chrome, display flags),
and caps a few repeated lists with a ``<key>_not_shown`` count. That is a
denylist, not an allowlist, so the failure above cannot recur: a field nobody
named, ``product.price`` included, still passes through.

The size of a response is therefore set by the result count and the provider's
trim, not by a character budget hidden in here. ``WEB_SEARCH_MAX_RESULTS`` sets
the count for a call that does not ask for one; the agent picks per call
otherwise. If that is ever too much, ask for fewer results rather than adding a
budget here: a cut in this module cannot know which field was the answer.
"""

from typing import Any


def _walk(value: Any, prefix: str, lines: list[str]) -> None:
    """Flatten *value* into ``key: value`` lines, dotting nested keys."""
    if isinstance(value, dict):
        for key, sub in value.items():
            _walk(sub, f"{prefix}.{key}" if prefix else str(key), lines)
        return

    if isinstance(value, list):
        for i, sub in enumerate(value):
            _walk(sub, f"{prefix}[{i}]", lines)
        return

    # Null and empty string say nothing the model can act on. Everything else,
    # including numbers and booleans, is rendered as the provider sent it.
    if value is None or value == "":
        return

    lines.append(f"   {prefix}: {value}")


def render_record(record: dict[str, Any], index: int) -> str:
    """Render one provider record as a numbered block of ``key: value`` lines."""
    lines: list[str] = []
    _walk(record, "", lines)
    body = "\n".join(lines)
    return f"[{index}]\n{body}" if body else f"[{index}]\n   (empty result)"


def render_records(records: list[dict[str, Any]]) -> str:
    """Render every record the provider returned, in order."""
    return "\n\n".join(render_record(record, i) for i, record in enumerate(records, 1))
