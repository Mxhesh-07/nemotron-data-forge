"""Instruction/response generation with NVIDIA Nemotron on Token Factory.

Two entry points:

- :func:`generate_examples` builds fresh examples from a source.
- :func:`rewrite_examples` regenerates examples the grader rejected, using the
  grader's reason and per-axis scores as guidance (rewrite-on-reject).
"""

from __future__ import annotations

from . import config
from .chunking import chunk_text
from .logging_utils import get_logger
from .schemas import Example, Source
from .validation import clean_fields, validate_example

logger = get_logger(__name__)

FORMAT_GUIDANCE = {
    "Q&A": "a single direct question and a concise, factual answer",
    "Chat": "a realistic user turn and a natural assistant reply",
    "Reasoning": (
        "a question that needs multi-step thinking and an answer that reasons step by "
        "step, one sub-step per line"
    ),
}

SYSTEM_PROMPT = (
    "You are a meticulous dataset-generation engine for fine-tuning language models. "
    "You only produce factual, well-written training examples that are fully grounded "
    "in the provided source text. You never invent facts. You always reply with valid JSON."
)

REWRITE_SYSTEM_PROMPT = (
    "You are a dataset repair engine. Earlier training examples were rejected by a quality "
    "grader. You produce improved replacements that are fully grounded in the given source "
    "text and that directly fix the stated problems. You always reply with valid JSON."
)


def _user_prompt(source: Source, content: str, count: int, fmt: str, topic: str) -> str:
    guidance = FORMAT_GUIDANCE.get(fmt, FORMAT_GUIDANCE["Q&A"])
    return f"""Create {count} high-quality training examples on the topic: "{topic}".

Each example must be {guidance}.

Grounding rules:
- Every factual claim in the response MUST be supported by the source text below.
- If the source does not cover something, do not invent it.
- Vary the questions; do not repeat the same intent.
- Write in clear English.

Source title: {source.title}
Source URL: {source.url}
Source text:
\"\"\"
{content}
\"\"\"

Return ONLY JSON in this exact shape:
{{"examples": [{{"instruction": "...", "response": "..."}}]}}
"""


def _rewrite_prompt(source: Source, rejected: list[Example], fmt: str, topic: str) -> str:
    guidance = FORMAT_GUIDANCE.get(fmt, FORMAT_GUIDANCE["Q&A"])
    problems = []
    for i, ex in enumerate(rejected):
        scores = ex.scores or {}
        score_text = ", ".join(f"{k}={v}" for k, v in scores.items()) or "unscored"
        problems.append(
            f"{i + 1}. INSTRUCTION: {ex.instruction}\n"
            f"   WEAK RESPONSE: {ex.response}\n"
            f"   SCORES: {score_text}\n"
            f"   GRADER NOTE: {ex.reason or 'below quality bar'}"
        )
    joined = "\n".join(problems)
    return f"""Improve these rejected training examples on the topic: "{topic}".

Each improved example must be {guidance}.

Fix the problems the grader reported (low accuracy, helpfulness or clarity).
Keep the same underlying intent where it is sound, but correct or rewrite the
answer so that every claim is supported by the source text below. Do not invent
facts. Produce exactly {len(rejected)} improved examples.

Source title: {source.title}
Source URL: {source.url}
Source text:
\"\"\"
{source.content[: config.MAX_SOURCE_CHARS]}
\"\"\"

Rejected examples:
{joined}

Return ONLY JSON in this exact shape:
{{"examples": [{{"instruction": "...", "response": "..."}}]}}
"""


def _collect(
    raw_items,
    source: Source,
    topic: str,
    fmt: str,
    limit: int,
    seen: set[str],
    rewritten: bool = False,
) -> list[Example]:
    """Validate and materialise model-provided examples, skipping bad ones."""
    examples: list[Example] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        instruction, response = clean_fields(
            str(item.get("instruction", "")), str(item.get("response", ""))
        )
        reason = validate_example(instruction, response)
        if reason:
            logger.debug("Dropped example from %s: %s", source.url, reason)
            continue
        key = instruction.lower()
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            Example(
                instruction=instruction,
                response=response,
                source_url=source.url,
                source_title=source.title,
                topic=topic,
                format=fmt,
                rewritten=rewritten,
            )
        )
        if len(examples) >= limit:
            break
    return examples


def _extract_items(data) -> list:
    if isinstance(data, dict):
        return data.get("examples", []) or []
    return data or []


def generate_examples(
    client,
    source: Source,
    topic: str,
    fmt: str = "Q&A",
    count: int = 3,
    model: str | None = None,
    temperature: float = 0.4,
) -> list[Example]:
    """Generate up to ``count`` validated examples from a single source.

    Long sources are chunked so the model sees focused context. Any pair that
    fails validation is dropped rather than exported.
    """
    if count <= 0:
        return []

    model = model or config.GENERATOR_MODEL
    # Cap how much of a long source is used to MAX_SOURCE_CHARS of text. This is a
    # number of *chunks*, not a raw list slice by character count (a previous
    # `[: config.MAX_SOURCE_CHARS]` capped the chunk list at ~20000 entries, i.e.
    # did nothing).
    max_chunks = max(1, config.MAX_SOURCE_CHARS // max(1, config.SOURCE_CHUNK_CHARS))
    chunks = chunk_text(source.content, config.SOURCE_CHUNK_CHARS)[:max_chunks] or [
        source.content[: config.MAX_SOURCE_CHARS]
    ]
    per_chunk = max(1, -(-count // len(chunks)))  # ceil division

    examples: list[Example] = []
    seen: set[str] = set()

    for content in chunks:
        if len(examples) >= count:
            break
        remaining = count - len(examples)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _user_prompt(source, content, min(per_chunk, remaining), fmt, topic),
            },
        ]
        try:
            data = client.chat_json(
                model, messages, temperature=temperature, max_tokens=config.MAX_OUTPUT_TOKENS
            )
        except Exception as exc:  # noqa: BLE001 - one chunk failing should not kill the source
            logger.warning("Generation failed for %s: %s", source.url, exc)
            continue

        examples.extend(
            _collect(_extract_items(data), source, topic, fmt, count - len(examples), seen)
        )

    return examples[:count]


def rewrite_examples(
    client,
    source: Source,
    rejected: list[Example],
    topic: str,
    fmt: str = "Q&A",
    model: str | None = None,
    temperature: float = 0.4,
) -> list[Example]:
    """Regenerate rejected examples in a single grounded repair pass.

    Returns validated replacements tagged with ``rewritten=True``. The original
    rejected examples are not modified.
    """
    if not rejected:
        return []

    model = model or config.GENERATOR_MODEL
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _rewrite_prompt(source, rejected, fmt, topic),
        },
    ]
    try:
        data = client.chat_json(
            model, messages, temperature=temperature, max_tokens=config.MAX_OUTPUT_TOKENS
        )
    except Exception as exc:  # noqa: BLE001 - a failed repair pass is not fatal
        logger.warning("Rewrite failed for %s: %s", source.url, exc)
        return []

    return _collect(
        _extract_items(data), source, topic, fmt, len(rejected), set(), rewritten=True
    )
