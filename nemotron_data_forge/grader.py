"""Quality grading / filtering with NVIDIA Nemotron 3 Ultra on Token Factory."""

from __future__ import annotations

from . import config
from .logging_utils import get_logger
from .schemas import Example, Source
from .validation import normalize_scores

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a ruthless data-quality grader for a fine-tuning dataset. "
    "You judge each example strictly against the source text and reply with valid JSON only."
)


def _user_prompt(source: Source, batch: list[Example]) -> str:
    content = source.content[: config.MAX_SOURCE_CHARS]
    numbered = "\n".join(
        f"{i}. INSTRUCTION: {ex.instruction}\n   RESPONSE: {ex.response}"
        for i, ex in enumerate(batch)
    )
    return f"""Grade each example below on three axes, from 0 to 10:

- accuracy: is every claim supported by the source text? (10 = fully supported, 0 = invented)
- helpfulness: would this be genuinely useful in a fine-tuning dataset?
- clarity: is the response well-written, specific and unambiguous?

Source text:
\"\"\"
{content}
\"\"\"

Examples:
{numbered}

Return ONLY JSON in this exact shape, with one entry per example using 0-based indices:
{{"scores": [{{"index": 0, "accuracy": 8, "helpfulness": 9, "clarity": 8, "reason": "short note"}}]}}
"""


def grade_examples(
    client,
    source: Source,
    examples: list[Example],
    model: str | None = None,
    batch_size: int | None = None,
    temperature: float = 0.0,
) -> list[Example]:
    """Attach scores to each example in place and return the same list.

    Examples are graded in batches to keep prompts within token limits. Any
    example the grader fails to score is marked as rejected rather than kept.
    """
    if not examples:
        return examples

    model = model or config.GRADER_MODEL
    size = max(1, batch_size if batch_size is not None else config.GRADE_BATCH_SIZE)

    for start in range(0, len(examples), size):
        batch = examples[start : start + size]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(source, batch)},
        ]
        rows: list[dict] = []
        try:
            data = client.chat_json(
                model, messages, temperature=temperature, max_tokens=config.MAX_OUTPUT_TOKENS
            )
            if isinstance(data, dict):
                rows = data.get("scores", []) or []
            elif isinstance(data, list):
                rows = data
        except Exception as exc:  # noqa: BLE001 - mark batch ungraded, keep the run alive
            logger.warning("Grading failed for a batch of %s: %s", source.url, exc)

        by_index = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                by_index[int(row.get("index"))] = row
            except (TypeError, ValueError):
                continue

        for local_i, example in enumerate(batch):
            row = by_index.get(local_i)
            if row is None:
                example.scores = {}
                example.average_score = 0.0
                example.verdict = "rejected"
                example.reason = "grader returned no score"
                continue
            example.scores = normalize_scores(
                row.get("accuracy"), row.get("helpfulness"), row.get("clarity")
            )
            example.average_score = round(sum(example.scores.values()) / 3.0, 2)
            example.reason = str(row.get("reason", "")).strip()

    return examples


def apply_threshold(
    examples: list[Example],
    threshold: float,
    min_axis: float | None = None,
) -> None:
    """Mark examples as kept/rejected based on the average and per-axis floor.

    Examples already marked rejected (e.g. ungraded) are left untouched.
    """
    floor = config.DEFAULT_MIN_AXIS_SCORE if min_axis is None else min_axis
    for example in examples:
        if example.verdict == "rejected" and not example.scores:
            continue
        lowest = min(example.scores.values(), default=0.0)
        if example.average_score >= threshold and lowest >= floor:
            example.verdict = "kept"
        else:
            example.verdict = "rejected"
            if not example.reason:
                if lowest < floor:
                    example.reason = f"axis score {lowest} below floor {floor}"
                else:
                    example.reason = f"average {example.average_score} below threshold {threshold}"
