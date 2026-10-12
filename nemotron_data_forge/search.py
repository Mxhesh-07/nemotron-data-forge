"""Tavily web search with retries, used to gather fresh grounding sources."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import requests

from . import config
from .config import require_tavily_key
from .errors import SearchError
from .logging_utils import get_logger
from .schemas import Source

logger = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def tavily_search(
    query: str,
    max_results: int = 4,
    api_key: str | None = None,
    search_depth: str = "basic",
    timeout: float = 30.0,
    max_retries: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Source]:
    """Search the web with Tavily and return Source objects.

    Retries transient failures with exponential backoff. Raises SearchError on
    permanent failure.
    """
    if not query or not query.strip():
        raise SearchError("Search query must not be empty")

    key = require_tavily_key(api_key)
    retries = max(1, max_retries if max_retries is not None else config.MAX_RETRIES)
    payload = {
        "api_key": key,
        "query": query.strip(),
        "max_results": max(1, int(max_results)),
        "search_depth": search_depth,
        "include_answer": False,
        "include_raw_content": False,
    }

    last_error: str | None = None
    for attempt in range(retries):
        try:
            response = requests.post(config.TAVILY_URL, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = str(exc)
            _backoff(attempt, retries, sleep, f"Tavily request error: {exc}")
            continue

        if response.status_code in _RETRYABLE_STATUS:
            last_error = f"HTTP {response.status_code}"
            _backoff(attempt, retries, sleep, f"Tavily transient HTTP {response.status_code}")
            continue

        if response.status_code >= 400:
            raise SearchError(
                f"Tavily returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise SearchError(f"Tavily returned invalid JSON: {exc}") from exc

        return _parse_results(data)

    raise SearchError(f"Tavily search failed after {retries} attempts: {last_error}")


def _backoff(attempt: int, retries: int, sleep: Callable[[float], None], message: str) -> None:
    if attempt >= retries - 1:
        return
    delay = min(2.0 ** attempt, 15.0) * (0.5 + random.random() * 0.5)
    logger.warning("%s (attempt %d/%d), retrying in %.1fs", message, attempt + 1, retries, delay)
    sleep(delay)


def _parse_results(data: dict) -> list[Source]:
    sources: list[Source] = []
    for item in (data or {}).get("results", []) or []:
        content = (item.get("content") or "").strip()
        url = (item.get("url") or "").strip()
        if not content or not url:
            continue
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        sources.append(
            Source(
                title=(item.get("title") or url).strip(),
                url=url,
                content=content,
                score=score,
            )
        )
    return sources


# --- Offline sample sources -------------------------------------------------
# Used by demo mode so the pipeline can be exercised without a Tavily key. These
# are short, clearly-labelled blurbs, not real web pages.
SAMPLE_SOURCES: list[Source] = [
    Source(
        title="Sample: Tax basics for freelancers in India",
        url="https://example.com/sample-india-freelance-tax",
        content=(
            "Freelancers in India are generally treated as business owners. Income from "
            "freelancing is taxed under 'Profits and Gains of Business or Profession'. If "
            "gross receipts are below the presumptive-taxation limits, a freelancer may "
            "opt for presumptive taxation and declare a fixed percentage of turnover as "
            "income. Freelancers should track professional expenses such as software, "
            "internet, and a home-office share, as these can reduce taxable income."
        ),
        score=0.9,
    ),
    Source(
        title="Sample: GST registration thresholds",
        url="https://example.com/sample-gst-threshold",
        content=(
            "GST registration is generally required once annual turnover crosses the "
            "threshold for services. Depending on the state, service providers may need "
            "to register earlier, and many freelancers register voluntarily to claim input "
            "credit. Once registered, invoices must show the GSTIN and tax charged. "
            "Export of services is typically zero-rated, but filing returns is still needed."
        ),
        score=0.85,
    ),
    Source(
        title="Sample: Advance tax and deadlines",
        url="https://example.com/sample-advance-tax",
        content=(
            "If total tax liability for the year is above the small threshold, freelancers "
            "must pay advance tax in quarterly instalments. Missing an instalment can "
            "attract interest. Income tax returns for freelancers are usually filed by the "
            "deadline that applies to non-audit cases, with a late-filing option available "
            "after that date."
        ),
        score=0.8,
    ),
]
