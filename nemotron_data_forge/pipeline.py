"""End-to-end pipeline: Tavily -> Nemotron generate -> Nemotron grade -> dedupe -> export.

The pipeline is resilient: a failure in one source is recorded and the run
continues. Generation and grading run concurrently with a bounded thread pool.
"""

from __future__ import annotations

import datetime as _dt
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import __version__, config
from .dedupe import dedupe
from .generator import generate_examples, rewrite_examples
from .grader import apply_threshold, grade_examples
from .llm import TokenFactoryClient
from .logging_utils import get_logger
from .schemas import (
    Example,
    PipelineResult,
    PipelineStats,
    RunManifest,
    Source,
    SourceOutcome,
)
from .search import SAMPLE_SOURCES, tavily_search

logger = get_logger(__name__)
ProgressFn = Callable[[float, str], None]


class _Meter:
    """Maps a phase's completion (0..1) onto a slice of the global progress bar."""

    def __init__(self, callback: ProgressFn, start: float, end: float) -> None:
        self._cb = callback
        self._start = start
        self._end = end
        self._total = 1
        self._done = 0
        self._lock = threading.Lock()

    def set_total(self, total: int) -> None:
        self._total = max(1, total)

    def tick(self, message: str) -> None:
        with self._lock:
            self._done += 1
            fraction = min(1.0, self._done / self._total)
        self.report(fraction, message)

    def report(self, fraction: float, message: str) -> None:
        pct = self._start + (self._end - self._start) * max(0.0, min(1.0, fraction))
        self._cb(pct, message)


def run_pipeline(
    topic: str,
    fmt: str = "Q&A",
    examples_per_source: int = 3,
    max_sources: int = 4,
    keep_threshold: float = config.DEFAULT_KEEP_THRESHOLD,
    min_axis_score: float | None = None,
    nebius_api_key: str | None = None,
    tavily_api_key: str | None = None,
    generator_model: str | None = None,
    grader_model: str | None = None,
    use_sample_sources: bool = False,
    use_embeddings: bool = True,
    max_concurrency: int | None = None,
    max_rewrite_rounds: int | None = None,
    max_usd: float | None = None,
    client: TokenFactoryClient | None = None,
    search_fn: Callable[..., list[Source]] | None = None,
    on_progress: ProgressFn | None = None,
) -> PipelineResult:
    """Run the full dataset-generation workflow and return the result."""
    started = time.time()
    started_iso = _dt.datetime.now().isoformat(timespec="seconds")
    fmt = config.validate_format(fmt)
    progress: ProgressFn = on_progress or (lambda _p, _m: None)
    worker = max(1, max_concurrency if max_concurrency is not None else config.MAX_CONCURRENCY)
    searcher = search_fn or tavily_search
    errors: list[str] = []

    if client is None:
        client = TokenFactoryClient(api_key=nebius_api_key)

    budget = max(0.0, max_usd if max_usd is not None else config.DEFAULT_MAX_USD)
    budget_hit = False

    def _over_budget() -> bool:
        return budget > 0 and client.usage.total_cost_usd() >= budget

    stats = PipelineStats(
        topic=topic,
        format=fmt,
        models={
            "generator": generator_model or config.GENERATOR_MODEL,
            "grader": grader_model or config.GRADER_MODEL,
            "embeddings": config.EMBEDDING_MODEL if use_embeddings else "disabled",
        },
    )

    # --- 1. Gather sources -------------------------------------------------
    progress(2.5, "Searching the web with Tavily...")
    if use_sample_sources:
        sources: list[Source] = list(SAMPLE_SOURCES[:max_sources])
    else:
        try:
            sources = searcher(topic, max_results=max_sources, api_key=tavily_api_key)
        except Exception as exc:  # noqa: BLE001 - surface as a clean error
            errors.append(f"search failed: {exc}")
            logger.error("Search failed: %s", exc)
            sources = []
    stats.sources_used = len(sources)
    if not sources:
        return _finalize(
            [], [], stats, sources, [], errors, client, started, started_iso,
            budget_usd=budget, budget_hit=budget_hit,
        )

    outcomes: dict[str, SourceOutcome] = {
        s.url: SourceOutcome(url=s.url, title=s.title) for s in sources
    }

    # --- 2. Generate (concurrent per source) -------------------------------
    generated: list[Example] = []
    gen_meter = _Meter(progress, 5.0, 65.0)
    gen_meter.set_total(len(sources))

    def _generate(source: Source) -> list[Example]:
        return generate_examples(
            client, source, topic=topic, fmt=fmt, count=examples_per_source, model=generator_model
        )

    with ThreadPoolExecutor(max_workers=min(worker, len(sources))) as pool:
        future_map = {pool.submit(_generate, s): s for s in sources}
        for future in as_completed(future_map):
            source = future_map[future]
            try:
                produced = future.result()
                generated.extend(produced)
                outcomes[source.url].generated = len(produced)
            except Exception as exc:  # noqa: BLE001
                outcomes[source.url].error = str(exc)
                errors.append(f"generation failed for {source.url}: {exc}")
                logger.error("Generation failed for %s: %s", source.url, exc)
            gen_meter.tick(f"Generated from {outcomes[source.url].title[:40]}...")
            if not budget_hit and _over_budget():
                budget_hit = True
                errors.append(
                    f"budget cap of ${budget:.4f} reached during generation; "
                    "stopped generating new sources"
                )
                logger.warning("Budget cap reached; cancelling queued generation")
                pool.shutdown(wait=False, cancel_futures=True)
                break

    stats.generated = len(generated)

    # --- 3. Grade (concurrent per source) ----------------------------------
    by_url: dict[str, Source] = {s.url: s for s in sources}
    grade_groups: dict[str, list[Example]] = {s.url: [] for s in sources}
    for example in generated:
        grade_groups.setdefault(example.source_url, []).append(example)
    grade_targets = [(url, group) for url, group in grade_groups.items() if group]

    grade_meter = _Meter(progress, 65.0, 84.0)
    grade_meter.set_total(len(grade_targets))

    def _grade(item) -> None:
        url, group = item
        grade_examples(client, by_url[url], group, model=grader_model)

    with ThreadPoolExecutor(max_workers=min(worker, max(1, len(grade_targets)))) as pool:
        future_map = {pool.submit(_grade, item): item[0] for item in grade_targets}
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                future.result()
                outcomes[url].graded = len(grade_groups[url])
            except Exception as exc:  # noqa: BLE001
                outcomes[url].error = outcomes[url].error or str(exc)
                errors.append(f"grading failed for {url}: {exc}")
                logger.error("Grading failed for %s: %s", url, exc)
            grade_meter.tick(f"Graded {outcomes[url].title[:40]}...")

    apply_threshold(generated, keep_threshold, min_axis_score)

    # --- 3b. Rewrite on reject (optional) ----------------------------------
    rewrite_rounds = max(
        0, max_rewrite_rounds if max_rewrite_rounds is not None else config.DEFAULT_REWRITE_ROUNDS
    )
    # Budget gate: grading the already-generated (bounded) set still runs, but
    # the optional enhancement phases are skipped once the cap is reached.
    if budget_hit or _over_budget():
        budget_hit = True
        if rewrite_rounds > 0:
            errors.append("skipped rewrite-on-reject: budget cap reached")
        rewrite_rounds = 0
    if rewrite_rounds > 0:
        # One meter across all rounds so the progress fraction never goes
        # backwards; the total grows as each round adds targets.
        rewrite_meter = _Meter(progress, 84.0, 90.0)
        rewrite_targets_seen = 0
        for round_index in range(rewrite_rounds):
            weak = [e for e in generated if e.verdict == "rejected" and e.scores]
            if not weak:
                break

            groups: dict[str, list[Example]] = {}
            for example in weak:
                groups.setdefault(example.source_url, []).append(example)
            targets = [(url, items) for url, items in groups.items() if url in by_url]
            rewrite_targets_seen += len(targets)
            rewrite_meter.set_total(rewrite_targets_seen)

            def _rewrite(item) -> list[Example]:
                url, items = item
                return rewrite_examples(
                    client, by_url[url], items, topic=topic, fmt=fmt, model=generator_model
                )

            replacements: list[Example] = []
            replace_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=min(worker, max(1, len(targets)))) as pool:
                future_map = {pool.submit(_rewrite, item): item[0] for item in targets}
                for future in as_completed(future_map):
                    url = future_map[future]
                    try:
                        produced = future.result()
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"rewrite failed for {url}: {exc}")
                        logger.error("Rewrite failed for %s: %s", url, exc)
                        produced = []
                    with replace_lock:
                        replacements.extend(produced)
                    rewrite_meter.tick(f"Rewrote examples from {outcomes[url].title[:40]}...")

            if not replacements:
                break

            regroups: dict[str, list[Example]] = {}
            for example in replacements:
                regroups.setdefault(example.source_url, []).append(example)
            with ThreadPoolExecutor(max_workers=min(worker, max(1, len(regroups)))) as pool:
                futures = [
                    pool.submit(grade_examples, client, by_url[url], items, grader_model)
                    for url, items in regroups.items()
                    if url in by_url
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"rewrite grading failed: {exc}")
                        logger.error("Rewrite grading failed: %s", exc)

            apply_threshold(replacements, keep_threshold, min_axis_score)
            generated.extend(replacements)
            stats.rewritten += len(replacements)
            logger.info(
                "Rewrite round %d: %d replacements generated", round_index + 1, len(replacements)
            )

    stats.graded = sum(1 for e in generated if e.scores)
    budget_hit = budget_hit or _over_budget()

    # --- 4. Dedupe ---------------------------------------------------------
    progress(90.0, "Removing near-duplicate examples...")
    candidates = [e for e in generated if e.verdict == "kept"]
    embeddings = None
    if use_embeddings and candidates:
        if budget_hit or _over_budget():
            budget_hit = True
            errors.append("skipped semantic de-duplication (embeddings): budget cap reached")
        else:
            embeddings = client.embed([f"{e.instruction}\n{e.response}" for e in candidates])
    kept = dedupe(candidates, threshold=config.DEFAULT_DEDUPE_THRESHOLD, embeddings=embeddings)
    stats.duplicates_removed = len(candidates) - len(kept)

    rejected = [e for e in generated if e.verdict != "kept"]

    # --- 5. Stats + report -------------------------------------------------
    progress(96.0, "Building quality report...")
    stats.kept = len(kept)
    stats.rejected = len(rejected)
    stats.recovered = sum(1 for e in kept if e.rewritten)
    if kept:
        n = len(kept)
        stats.avg_accuracy = round(sum(e.scores.get("accuracy", 0) for e in kept) / n, 2)
        stats.avg_helpfulness = round(sum(e.scores.get("helpfulness", 0) for e in kept) / n, 2)
        stats.avg_clarity = round(sum(e.scores.get("clarity", 0) for e in kept) / n, 2)

    progress(100.0, "Done.")
    return _finalize(
        kept,
        rejected,
        stats,
        sources,
        list(outcomes.values()),
        errors,
        client,
        started,
        started_iso,
        {
            "examples_per_source": examples_per_source,
            "max_sources": max_sources,
            "keep_threshold": keep_threshold,
            "min_axis_score": (
                config.DEFAULT_MIN_AXIS_SCORE if min_axis_score is None else min_axis_score
            ),
            "dedupe_threshold": config.DEFAULT_DEDUPE_THRESHOLD,
            "use_embeddings": use_embeddings,
            "max_concurrency": worker,
            "max_rewrite_rounds": rewrite_rounds,
            "max_usd": budget,
            "sample_sources": use_sample_sources,
        },
        budget_usd=budget,
        budget_hit=budget_hit,
    )


def _finalize(
    kept: list[Example],
    rejected: list[Example],
    stats: PipelineStats,
    sources: list[Source],
    outcomes: list[SourceOutcome],
    errors: list[str],
    client: TokenFactoryClient,
    started: float,
    started_iso: str,
    run_config: dict | None = None,
    budget_usd: float = 0.0,
    budget_hit: bool = False,
) -> PipelineResult:
    finished = time.time()
    manifest = RunManifest(
        version=__version__,
        started_at=started_iso,
        finished_at=_dt.datetime.now().isoformat(timespec="seconds"),
        duration_seconds=round(finished - started, 2),
        config=run_config or {},
        usage=client.usage.as_dict(),
        estimated_cost_usd=client.usage.total_cost_usd(),
        total_tokens=client.usage.total_tokens(),
        budget_usd=budget_usd,
        budget_hit=budget_hit,
        sources=[s.to_dict() for s in sources],
        errors=errors,
    )
    return PipelineResult(
        examples=kept,
        rejected=rejected,
        stats=stats,
        sources=sources,
        manifest=manifest,
        outcomes=outcomes,
    )
