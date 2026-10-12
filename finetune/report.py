"""Build the before/after comparison report (Markdown + JSON).

Pure and offline: it consumes already-generated predictions, so it can be unit
tested without loading any model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import scoring


@dataclass
class ComparisonRow:
    instruction: str
    reference: str
    before: str
    after: str
    fmt: str = ""
    source_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_METRIC_LABELS = {
    "token_f1": "Token F1 vs reference",
    "keyword_recall": "Keyword recall",
    "length_closeness": "Length closeness",
    "format_adherence": "Format adherence",
    "words": "Avg words",
}


def score_rows(rows: Sequence[ComparisonRow]) -> dict[str, Any]:
    """Per-metric aggregates for the before and after predictions."""
    before = [scoring.evaluate(r.before, r.reference, r.fmt) for r in rows]
    after = [scoring.evaluate(r.after, r.reference, r.fmt) for r in rows]
    before_agg = scoring.aggregate(before)
    after_agg = scoring.aggregate(after)
    return {
        "n": len(rows),
        "before": before_agg,
        "after": after_agg,
        "delta": {
            key: round(after_agg.get(key, 0.0) - before_agg.get(key, 0.0), 4)
            for key in before_agg
        },
    }


def render_markdown(
    rows: Sequence[ComparisonRow],
    metrics: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> str:
    """Render a human-readable before/after report."""
    meta = meta or {}
    lines: list[str] = ["# Before/After Fine-Tuning Report", ""]

    if meta:
        lines.append("## Run")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("| --- | --- |")
        for key in ("base_model", "dataset_path", "n_train", "n_eval", "epochs", "lora_r"):
            if key in meta:
                lines.append(f"| {key} | `{meta[key]}` |")
        lines.append("")

    lines.append("## Metrics (mean over eval examples)")
    lines.append("")
    lines.append("| Metric | Before | After | Delta |")
    lines.append("| --- | --- | --- | --- |")
    for key, label in _METRIC_LABELS.items():
        b = metrics["before"].get(key)
        a = metrics["after"].get(key)
        d = metrics["delta"].get(key)
        if b is None:
            continue
        arrow = "" if d is None else ("  ▲" if d > 0 else ("  ▼" if d < 0 else "  ="))
        lines.append(f"| {label} | {b} | {a} | {d}{arrow} |")
    lines.append("")

    lines.append("## Side by side")
    for i, row in enumerate(rows, 1):
        lines.append("")
        lines.append(f"### {i}. {row.instruction}")
        if row.source_url:
            lines.append("")
            lines.append(f"_Source: {row.source_url}_")
        lines.append("")
        lines.append("**Reference:**")
        lines.append("")
        lines.append(f"> {row.reference}")
        lines.append("")
        lines.append("**Before (base model):**")
        lines.append("")
        lines.append(f"> {row.before or '_(empty)_'}")
        lines.append("")
        lines.append("**After (fine-tuned):**")
        lines.append("")
        lines.append(f"> {row.after or '_(empty)_'}")
    lines.append("")

    return "\n".join(lines)


def save(
    rows: Sequence[ComparisonRow],
    metrics: dict[str, Any],
    out_dir: str | Path,
    meta: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``before_after.md`` and ``before_after.json`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "before_after.md"
    json_path = out / "before_after.json"

    md_path.write_text(render_markdown(rows, metrics, meta), encoding="utf-8")
    payload = {
        "meta": meta or {},
        "metrics": metrics,
        "rows": [row.to_dict() for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return md_path, json_path
