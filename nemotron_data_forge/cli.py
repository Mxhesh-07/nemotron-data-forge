"""Command-line interface for Nemotron Data Forge."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from . import __version__, config
from .errors import ConfigError
from .export import (
    dataset_card,
    push_to_huggingface,
    quality_report,
    save_json,
    save_text,
    to_jsonl,
)
from .logging_utils import configure_logging, get_logger
from .pipeline import run_pipeline

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataforge",
        description="Generate, grade and filter a fine-tuning dataset from a topic using "
        "NVIDIA Nemotron on Nebius Token Factory.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--topic", default=config.DEFAULT_TOPIC, help="Topic to build a dataset for.")
    parser.add_argument(
        "--format", default=config.DEFAULT_FORMAT, choices=config.VALID_FORMATS, help="Example format."
    )
    parser.add_argument(
        "--examples-per-source", type=int, default=config.DEFAULT_EXAMPLES_PER_SOURCE
    )
    parser.add_argument("--max-sources", type=int, default=config.DEFAULT_MAX_SOURCES)
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_KEEP_THRESHOLD)
    parser.add_argument("--min-axis-score", type=float, default=config.DEFAULT_MIN_AXIS_SCORE)
    parser.add_argument(
        "--concurrency", type=int, default=config.MAX_CONCURRENCY, help="Parallel model requests."
    )
    parser.add_argument(
        "--rewrite-rounds",
        type=int,
        default=config.DEFAULT_REWRITE_ROUNDS,
        help="Rewrite-on-reject rounds: regenerate examples the grader rejected (0 disables).",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=config.DEFAULT_MAX_USD,
        help="Soft spend cap in USD for the run (0 = unlimited).",
    )
    parser.add_argument("--demo", action="store_true", help="Use bundled sample sources (no Tavily).")
    parser.add_argument(
        "--no-embeddings", action="store_true", help="Disable semantic de-duplication."
    )
    parser.add_argument("--out", default=None, help="Output directory (default: outputs/).")
    parser.add_argument("--push-hf", default=None, help="Hugging Face dataset repo id, e.g. user/name.")
    parser.add_argument(
        "--manifest-json", action="store_true", help="Print the run manifest as JSON to stdout."
    )
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging("WARNING" if args.quiet else args.log_level)

    if args.examples_per_source < 1 or args.max_sources < 1:
        logger.error("--examples-per-source and --max-sources must be >= 1")
        return 2
    if args.rewrite_rounds < 0:
        logger.error("--rewrite-rounds must be >= 0")
        return 2
    if args.max_usd < 0:
        logger.error("--max-usd must be >= 0")
        return 2

    out_dir = Path(args.out) if args.out else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    def progress(pct: float, message: str) -> None:
        if not args.quiet:
            print(f"[{pct:5.1f}%] {message}", file=sys.stderr)

    try:
        result = run_pipeline(
            topic=args.topic,
            fmt=args.format,
            examples_per_source=args.examples_per_source,
            max_sources=args.max_sources,
            keep_threshold=args.threshold,
            min_axis_score=args.min_axis_score,
            use_sample_sources=args.demo,
            use_embeddings=not args.no_embeddings,
            max_concurrency=args.concurrency,
            max_rewrite_rounds=args.rewrite_rounds,
            max_usd=args.max_usd,
            on_progress=progress,
        )
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        logger.error("Interrupted")
        return 130

    jsonl_path = to_jsonl(result.examples, out_dir / f"dataset-{stamp}.jsonl")
    report_path = save_text(
        quality_report(result.examples, result.rejected, result.stats, result.sources, result.manifest),
        out_dir / f"quality-report-{stamp}.md",
    )
    manifest_path = save_json(result.manifest.to_dict(), out_dir / f"manifest-{stamp}.json")
    card_path = save_text(
        dataset_card(result.stats.topic, result.stats, result.manifest),
        out_dir / f"dataset-card-{stamp}.md",
    )

    if not args.quiet:
        print("\n=== Summary ===")
        print(json.dumps(result.stats.to_dict(), indent=2))
        print(f"\nDataset:  {jsonl_path}")
        print(f"Report:   {report_path}")
        print(f"Manifest: {manifest_path}")
        print(f"Card:     {card_path}")
        if result.manifest.errors:
            print(f"\nWarnings ({len(result.manifest.errors)}):")
            for err in result.manifest.errors:
                print(f"  - {err}")

    if args.push_hf:
        try:
            url = push_to_huggingface(
                jsonl_path,
                args.push_hf,
                topic=result.stats.topic,
                card_markdown=dataset_card(result.stats.topic, result.stats, result.manifest),
            )
            print(f"Pushed to Hugging Face: {url}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Hugging Face push failed: %s", exc)

    if args.manifest_json:
        print(json.dumps(result.manifest.to_dict(), indent=2))

    if result.stats.kept == 0:
        logger.warning("No examples passed the filters. Try lowering --threshold.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
