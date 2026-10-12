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
    EXPORT_FORMATS,
    dataset_card,
    push_to_huggingface,
    quality_report,
    save_json,
    save_text,
    split_dataset,
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify keys, model access and connectivity with cheap live calls, then exit.",
    )
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
    parser.add_argument(
        "--export-format",
        default="alpaca",
        choices=EXPORT_FORMATS,
        help="Dataset record layout: alpaca, sharegpt, openai or raw.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction held out for a validation split, e.g. 0.1 (0 = no split).",
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


def _make_stdout_unicode_safe() -> None:
    """Avoid UnicodeEncodeError on legacy consoles (e.g. Windows cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - stream not reconfigurable (e.g. under test)
            pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _make_stdout_unicode_safe()
    configure_logging("WARNING" if args.quiet else args.log_level)

    if args.check:
        from .doctor import check_environment

        checks = check_environment(run_search=not args.demo, require_tavily=not args.demo)
        all_ok = True
        for check in checks:
            if not check.ok:
                all_ok = False
            mark = "PASS" if check.ok else "FAIL"
            detail = f" - {check.detail}" if check.detail else ""
            print(f"[{mark}] {check.name}{detail}")
        print("\nAll checks passed." if all_ok else "\nSome checks failed.")
        return 0 if all_ok else 1

    if args.examples_per_source < 1 or args.max_sources < 1:
        logger.error("--examples-per-source and --max-sources must be >= 1")
        return 2
    if args.rewrite_rounds < 0:
        logger.error("--rewrite-rounds must be >= 0")
        return 2
    if args.max_usd < 0:
        logger.error("--max-usd must be >= 0")
        return 2
    if not 0.0 <= args.val_ratio < 1.0:
        logger.error("--val-ratio must be in [0, 1)")
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

    train, val = split_dataset(result.examples, args.val_ratio)
    result.manifest.config["export_format"] = args.export_format
    result.manifest.config["val_ratio"] = args.val_ratio
    result.manifest.config["train_examples"] = len(train)
    result.manifest.config["val_examples"] = len(val)

    jsonl_path = to_jsonl(train, out_dir / f"dataset-{stamp}.jsonl", fmt=args.export_format)
    val_path = None
    if val:
        val_path = to_jsonl(val, out_dir / f"val-{stamp}.jsonl", fmt=args.export_format)
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
        print(f"\nDataset:  {jsonl_path}  ({args.export_format}, {len(train)} rows)")
        if val_path is not None:
            print(f"Val:      {val_path}  ({len(val)} rows)")
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
