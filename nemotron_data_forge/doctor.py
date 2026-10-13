"""Environment and connectivity checks for `dataforge --check`.

Runs a handful of *cheap* live calls so you can confirm your keys and model IDs
before spending on a full dataset run. Everything is injectable so the checks can
be tested entirely offline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import config
from .logging_utils import get_logger
from .schemas import Source

logger = get_logger(__name__)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def check_environment(
    client=None,
    search_fn: Callable[..., list[Source]] | None = None,
    run_search: bool = True,
    require_tavily: bool = True,
) -> list[Check]:
    """Return a list of connectivity/configuration checks.

    ``client`` and ``search_fn`` can be supplied to avoid real network calls
    (used by the tests). ``run_search`` toggles the live Tavily probe.
    """
    checks: list[Check] = []

    has_key = bool((config.NEBIUS_API_KEY or "").strip())
    checks.append(
        Check(
            f"{config.LLM_KEY_ENV} present",
            has_key,
            "" if has_key else f"set {config.LLM_KEY_ENV} in .env",
        )
    )
    if not has_key and client is None:
        return checks

    if client is None:
        from .llm import TokenFactoryClient

        try:
            client = TokenFactoryClient()
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            checks.append(Check("LLM client", False, str(exc)))
            return checks
    checks.append(Check("LLM client", True, f"{config.LLM_PROVIDER_LABEL} - {config.NEBIUS_BASE_URL}"))

    # 1. Model catalogue and configured model IDs.
    try:
        models = client.list_models()
        checks.append(Check("List models", True, f"{len(models)} available"))
        for role, model in (
            ("generator", config.GENERATOR_MODEL),
            ("grader", config.GRADER_MODEL),
            ("embeddings", config.EMBEDDING_MODEL),
        ):
            if model in models:
                checks.append(Check(f"{role} model available", True, model))
            else:
                checks.append(Check(f"{role} model available", False, f"{model} not in catalogue"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("List models", False, str(exc)))

    # 2. A tiny chat round-trip (cheapest model).
    try:
        reply = client.chat(
            config.FAST_MODEL,
            [{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0,
            max_tokens=5,
        )
        checks.append(Check("Chat round-trip", bool(reply and reply.strip()), (reply or "").strip()[:40]))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Chat round-trip", False, str(exc)))

    # 3. Embeddings round-trip.
    try:
        vectors = client.embed(["ping"])
        ok = bool(vectors) and bool(vectors[0])
        checks.append(
            Check("Embeddings round-trip", ok, f"dim {len(vectors[0])}" if ok else "no vector returned")
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Embeddings round-trip", False, str(exc)))

    # 4. Tavily key + optional live search.
    has_tavily = bool((config.TAVILY_API_KEY or "").strip())
    if has_tavily:
        tavily_detail = ""
    elif require_tavily:
        tavily_detail = "set TAVILY_API_KEY in .env (or run with --demo)"
    else:
        tavily_detail = "not required in demo mode"
    checks.append(Check("TAVILY_API_KEY present", has_tavily or not require_tavily, tavily_detail))
    if run_search and has_tavily:
        from .search import tavily_search

        probe = search_fn or tavily_search
        try:
            results = probe("Nebius Token Factory", max_results=1, api_key=config.TAVILY_API_KEY)
            checks.append(Check("Tavily search round-trip", bool(results), f"{len(results)} result(s)"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("Tavily search round-trip", False, str(exc)))

    return checks
