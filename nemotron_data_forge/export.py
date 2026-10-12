"""Export a dataset as JSONL plus Markdown reports, manifests and an HF card."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .schemas import Example, PipelineStats, RunManifest


def to_jsonl(examples: list[Example], path: str | Path) -> Path:
    """Write examples as newline-delimited JSON (fine-tuning ready)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            record = {
                "instruction": ex.instruction,
                "output": ex.response,
                "input": "",
                "source_url": ex.source_url,
                "source_title": ex.source_title,
                "topic": ex.topic,
                "format": ex.format,
                "scores": ex.scores,
                "average_score": ex.average_score,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def save_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def quality_report(
    examples: list[Example],
    rejected: list[Example],
    stats: PipelineStats,
    sources=None,
    manifest: RunManifest | None = None,
) -> str:
    """Render a Markdown quality report for the dataset."""
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# Dataset Quality Report\n")
    lines.append(f"- **Topic:** {stats.topic}")
    lines.append(f"- **Format:** {stats.format}")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Generator model:** `{stats.models.get('generator', '')}`")
    lines.append(f"- **Grader model:** `{stats.models.get('grader', '')}`")
    lines.append(f"- **Embeddings:** `{stats.models.get('embeddings', '')}`")
    lines.append("")
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Sources used | {stats.sources_used} |")
    lines.append(f"| Examples generated | {stats.generated} |")
    lines.append(f"| Examples graded | {stats.graded} |")
    lines.append(f"| Examples kept | {stats.kept} |")
    lines.append(f"| Rejected (not kept) | {stats.rejected} |")
    lines.append(f"| Duplicates removed | {stats.duplicates_removed} |")
    lines.append(f"| Rewritten (on reject) | {stats.rewritten} |")
    lines.append(f"| Recovered by rewrite | {stats.recovered} |")
    lines.append(f"| Avg accuracy (kept) | {stats.avg_accuracy} |")
    lines.append(f"| Avg helpfulness (kept) | {stats.avg_helpfulness} |")
    lines.append(f"| Avg clarity (kept) | {stats.avg_clarity} |")
    if manifest is not None:
        lines.append(f"| Estimated cost (USD) | {manifest.estimated_cost_usd} |")
        lines.append(f"| Total tokens | {manifest.total_tokens} |")
        lines.append(f"| Duration (s) | {manifest.duration_seconds} |")
    lines.append("")

    if manifest is not None and manifest.usage:
        lines.append("## Token usage\n")
        lines.append("| Model | Calls | Prompt | Completion | Total |")
        lines.append("| --- | --- | --- | --- | --- |")
        for model, usage in manifest.usage.items():
            lines.append(
                f"| `{model}` | {usage.get('calls')} | {usage.get('prompt_tokens')} | "
                f"{usage.get('completion_tokens')} | {usage.get('total_tokens')} |"
            )
        lines.append("")

    if sources:
        lines.append("## Sources\n")
        for s in sources:
            lines.append(f"- [{s.title}]({s.url})")
        lines.append("")

    lines.append("## Kept examples\n")
    for i, ex in enumerate(examples, 1):
        lines.append(f"### {i}. {ex.instruction}")
        lines.append("")
        lines.append(ex.response)
        lines.append("")
        lines.append(
            f"_accuracy {ex.scores.get('accuracy')} · helpfulness "
            f"{ex.scores.get('helpfulness')} · clarity {ex.scores.get('clarity')} · "
            f"[source]({ex.source_url})_"
        )
        lines.append("")

    if rejected:
        lines.append("## Rejected examples\n")
        lines.append("| # | Instruction | Avg score | Reason |")
        lines.append("| --- | --- | --- | --- |")
        for i, ex in enumerate(rejected, 1):
            instr = ex.instruction.replace("|", "\\|")[:80]
            reason = (ex.reason or "").replace("|", "\\|")[:60]
            lines.append(f"| {i} | {instr} | {ex.average_score} | {reason} |")
        lines.append("")

    return "\n".join(lines)


def save_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def dataset_card(topic: str, stats: PipelineStats, manifest: RunManifest) -> str:
    """Return a minimal Hugging Face dataset card (YAML front matter + body)."""
    safe_topic = topic.replace('"', "'")
    front = (
        "---\n"
        f"license: mit\n"
        f"task_categories:\n  - text-generation\n"
        f"language:\n  - en\n"
        f"tags:\n  - nvidia-nemotron\n  - nebius-token-factory\n  - synthetic\n"
        f"pretty_name: \"{safe_topic} synthetic instructions\"\n"
        "---\n"
    )
    body = (
        f"\n# {topic} — synthetic instruction dataset\n\n"
        f"Generated with NVIDIA Nemotron 3 Super (generation) and Nemotron 3 Ultra "
        f"(grading) on Nebius Token Factory, using web sources found by Tavily.\n\n"
        f"- Examples kept: {stats.kept}\n"
        f"- Avg accuracy / helpfulness / clarity: "
        f"{stats.avg_accuracy} / {stats.avg_helpfulness} / {stats.avg_clarity}\n"
        f"- Generator: `{stats.models.get('generator', '')}`\n"
        f"- Grader: `{stats.models.get('grader', '')}`\n"
    )
    return front + body


def push_to_huggingface(
    jsonl_path: str | Path,
    repo_id: str,
    topic: str = "",
    card_markdown: str | None = None,
    token: str | None = None,
) -> str:
    """Push the JSONL dataset (and a dataset card) to the Hugging Face Hub."""
    from huggingface_hub import HfApi  # lazy: huggingface_hub is optional

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo="dataset.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    if card_markdown:
        api.upload_file(
            path_or_fileobj=card_markdown.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    return f"https://huggingface.co/datasets/{repo_id}"
