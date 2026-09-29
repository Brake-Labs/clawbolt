"""Brave Search web results via the Brave Search API.

https://api.search.brave.com/app/documentation/web-search/get-started

This provider does not reshape Brave's records beyond named trims. It pulls the
result list out of the envelope, strips markup, drops the keys in
``_DROPPED_KEYS``, drops values the record already states elsewhere (a snippet
repeating the description, a nested url equal to the result url, an offer
repeating its product's price, ``page_age`` beside ``age``), and caps the
repeated lists in ``_LIST_CAPS``. Every rule is a denylist or a dedup: any
field Brave sends that is not named here (including ones added after this was
written) reaches the model.
"""

import asyncio
import html
import logging
import re
from typing import Any

import httpx

from backend.app.integrations.web_search.errors import SearchUnavailableError

logger = logging.getLogger(__name__)

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Brave marks query terms inside text fields with <strong> tags. Removing them
# drops markup, not information, and it is applied to every string in the record
# rather than to a named list of fields, so it needs no updating when Brave adds
# one.
_TAG_RE = re.compile(r"<[^>]+>")

# Retried once each. 429 is a rate limit and 5xx is Brave being unwell; both
# are worth exactly one more attempt inside a live message loop, where the user
# is waiting on a reply and a long retry chain reads as a hang. 4xx other than
# 429 is a fault in our request and will fail identically on retry.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# Brave's own ceiling: count=30 returns 422.
_MAX_COUNT = 20

# Recency filters Brave accepts. Brave also takes a YYYY-MM-DDtoYYYY-MM-DD
# range, deliberately not exposed: the agent has no use for a precision it
# cannot justify, and a malformed range is a 422.
_FRESHNESS_CODES = frozenset({"pd", "pw", "pm", "py"})
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.5

# Keys dropped at any depth. Most are chrome for a results UI (image and favicon
# URLs, the site's display name and breadcrumb, schema type tags, boolean
# display flags) and carry no fact the agent can quote. The rest is product
# metadata no trade answer turns on: barcodes, a currency that is the
# searcher's own, and a rating's scale and review count (the score itself,
# ``ratingValue``, stays for comparing products). A denylist on purpose: an allowlist is what
# once dropped ``product.price``, and a field Brave adds later must still pass
# through. Title, url, description, names, prices, offers, snippets, FAQ
# answers and ages are never named here.
_DROPPED_KEYS = frozenset(
    {
        "thumbnail",
        "meta_url",
        "profile",
        "favicon",
        "img",
        "logo",
        "is_source_local",
        "is_source_both",
        "family_friendly",
        "type",
        "subtype",
        "is_live",
        "language",
        "is_tripadvisor",
        "gtin",
        "gtin8",
        "gtin12",
        "gtin13",
        "gtin14",
        "priceCurrency",
        "bestRating",
        "worstRating",
        "reviewCount",
    }
)

# Repeated lists capped to their first N entries, keyed by the list's key or by
# ``parent.key``. A capped list gets a sibling ``<key>_not_shown`` count, so the
# agent knows there was more and can search more narrowly. Offers carry the
# prices the agent quotes in estimates, so their cap stays loose enough to give
# a price range. Extra snippets are capped hardest: they were most of a result's
# size, a result stays in history for the rest of the session, and after
# ``_distinct_snippets`` the first one is Brave's most relevant new passage.
_LIST_CAPS: dict[str, int] = {
    "product_cluster": 3,
    "offers": 3,
    "extra_snippets": 1,
    "faq.items": 2,
}

# A site navigation menu scraped as a snippet: a long run of capitalized words
# with no digits and no sentence punctuation ("Specials & Offers Appliances
# Bath ... Site Map"). The digit and punctuation bars keep product names out: a
# snippet that states a size, a price or a sentence is not a menu.
_MENU_MIN_WORDS = 20
_MENU_CAPITALIZED_SHARE = 0.8
_MENU_CONNECTORS = frozenset({"&", "and", "|", "/", "-", "or"})
_NOT_A_MENU_RE = re.compile(r"[.!?:;$0-9]")


def _clean(value: Any) -> Any:
    """Strip highlight markup and unescape entities, recursively.

    Walks the whole record rather than named fields, so nested objects (Brave's
    ``product``, ``meta_url``, ``profile``) and lists (``extra_snippets``) are
    cleaned without being enumerated here. Non-string leaves pass through
    untouched, including the numbers and booleans Brave sends.
    """
    if isinstance(value, str):
        return html.unescape(_TAG_RE.sub("", value)).strip()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _norm(text: str) -> str:
    """Casefold and collapse whitespace, for containment checks."""
    return " ".join(text.split()).casefold()


def _is_menu(snippet: str) -> bool:
    """True for a snippet that reads as a site navigation menu, not prose."""
    if _NOT_A_MENU_RE.search(snippet):
        return False
    words = [w for w in snippet.split() if w.casefold() not in _MENU_CONNECTORS]
    if len(words) < _MENU_MIN_WORDS:
        return False
    capitalized = sum(1 for w in words if w[0].isupper())
    return capitalized / len(words) >= _MENU_CAPITALIZED_SHARE


def _distinct_snippets(snippets: list[Any], description: str) -> list[Any]:
    """Drop snippets that repeat the description or another snippet, and menus.

    A snippet contained in the description or in another snippet adds no word
    the agent has not already read. Order is kept, so the cap that follows
    still takes Brave's most relevant passage. Non-string entries pass through.
    """
    seen = [_norm(description)] if description else []
    kept: list[Any] = []
    for snippet in snippets:
        if not isinstance(snippet, str):
            kept.append(snippet)
            continue
        norm = _norm(snippet)
        if not norm or _is_menu(snippet) or any(norm in s for s in seen):
            continue
        # A later, longer snippet can contain an earlier one: keep the longer.
        kept = [k for k in kept if not (isinstance(k, str) and _norm(k) in norm)]
        kept.append(snippet)
        seen.append(norm)
    return kept


def _drop(value: Any, result_url: Any, depth: int = 0) -> Any:
    """Drop ``_DROPPED_KEYS`` at any depth, and nested copies of the result url.

    A product or offer url equal to the result's own url is the same link
    printed twice; one that differs (another listing, another seller) is kept.
    """
    if isinstance(value, list):
        return [_drop(v, result_url, depth + 1) for v in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _drop(sub, result_url, depth + 1)
        for key, sub in value.items()
        if key not in _DROPPED_KEYS
        and not (depth > 0 and key == "url" and result_url and sub == result_url)
    }


def _drop_redundant_offers(value: Any) -> Any:
    """Drop offers that only repeat what their parent already says.

    After ``_drop``, a single-seller offer is often ``{"price": "24.98"}`` under
    a product whose own price is ``"24.98"``. An offer whose every field matches
    the parent's adds nothing; an offer with a different price or any field of
    its own is kept, so a price range survives.
    """
    if isinstance(value, list):
        return [_drop_redundant_offers(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: _drop_redundant_offers(v) for k, v in value.items()}
    offers = out.get("offers")
    if isinstance(offers, list):
        kept = [
            o
            for o in offers
            if not (isinstance(o, dict) and all(out.get(k) == v for k, v in o.items()))
        ]
        if kept:
            out["offers"] = kept
        else:
            del out["offers"]
    return out


def _cap(value: Any, parent: str = "") -> Any:
    """Cap ``_LIST_CAPS`` lists, recursively, with a ``<key>_not_shown`` count."""
    if isinstance(value, list):
        return [_cap(v, parent) for v in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, sub in value.items():
        cap = _LIST_CAPS.get(f"{parent}.{key}", _LIST_CAPS.get(key))
        if cap is not None and isinstance(sub, list) and len(sub) > cap:
            out[key] = _cap(sub[:cap], key)
            out[f"{key}_not_shown"] = len(sub) - cap
        else:
            out[key] = _cap(sub, key)
    return out


def _trim(record: dict[str, Any]) -> dict[str, Any]:
    """Shape one cleaned Brave record: drop chrome and repeats, cap lists.

    Every rule removes a key by name or a value the record already states
    elsewhere, so a field this module has never heard of still reaches the
    model.
    """
    out = _drop(record, record.get("url"))
    # ``age`` is the readable form of ``page_age``; keep the raw timestamp only
    # when it is the one date the record has.
    if out.get("age"):
        out.pop("page_age", None)
    snippets = out.get("extra_snippets")
    if isinstance(snippets, list):
        description = out.get("description")
        distinct = _distinct_snippets(snippets, description if isinstance(description, str) else "")
        if distinct:
            out["extra_snippets"] = distinct
        else:
            del out["extra_snippets"]
    return _cap(_drop_redundant_offers(out))


class BraveSearchProvider:
    """General web search backed by the Brave Search API."""

    def __init__(self, api_key: str, *, timeout_seconds: float = 10.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.name = "brave"
        self.display_name = "Brave Search"

    async def _request(self, params: dict[str, str]) -> dict:
        """GET from Brave with bounded exponential backoff.

        Raises ``SearchUnavailableError`` when every attempt is exhausted on a
        retryable status, so the caller can tell a dead backend from a query
        with no results. Other HTTP errors propagate as ``HTTPStatusError`` so
        the tool can distinguish an auth failure (bad key, operator problem)
        from a transient one.
        """
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(_MAX_ATTEMPTS):
                resp = await client.get(_BRAVE_ENDPOINT, params=params, headers=headers)
                if resp.status_code in _RETRY_STATUSES:
                    if attempt == _MAX_ATTEMPTS - 1:
                        # The key is in a header, so the URL is safe to omit
                        # entirely; log the status only.
                        logger.warning(
                            "Brave search failed after %d attempts (status %d)",
                            _MAX_ATTEMPTS,
                            resp.status_code,
                        )
                        raise SearchUnavailableError(
                            f"Brave returned {resp.status_code} after {_MAX_ATTEMPTS} attempts"
                        )
                    delay = _BACKOFF_BASE_SECONDS * (2**attempt)
                    logger.warning(
                        "Brave search status %d, retrying in %.1fs", resp.status_code, delay
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()
        # Unreachable: the loop either returns, raises, or continues.
        raise SearchUnavailableError("Brave search exhausted its retries")

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        freshness: str | None = None,
    ) -> list[dict[str, Any]]:
        # Brave rejects count above 20 with a 422, so a caller asking for more
        # is clamped rather than handed an error it cannot act on.
        count = max(1, min(max_results, _MAX_COUNT))
        params = {
            "q": query,
            "count": str(count),
            # Plain web results only. Brave's news/video/location clusters
            # have their own shapes and answer a different question than the
            # one this tool asks. Product data is unaffected: Brave attaches
            # it to ordinary web results, which is where the prices live.
            "result_filter": "web",
            # No extra_snippets flag: Brave returns those passages either
            # way, measured identical byte counts with and without it, so
            # asking for them only implies a control that does not exist.
        }
        if freshness in _FRESHNESS_CODES:
            # Omitted entirely when unset or unrecognized. Sending an invalid
            # value is a 422, and an unfiltered search is the right fallback:
            # the question may well have an old but correct answer.
            params["freshness"] = freshness

        data = await self._request(params)

        results = (data.get("web") or {}).get("results") or []
        return [_trim(_clean(r)) for r in results[:count] if isinstance(r, dict)]
