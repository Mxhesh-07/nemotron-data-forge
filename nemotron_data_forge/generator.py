"""Instruction/response generation with NVIDIA Nemotron on Token Factory."""

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
    chunks = chunk_text(source.content, config.SOURCE_CHUNK_CHARS)[: config.MAX_SOURCE_CHARS] or [
        source.content[: config.MAX_SOURCE_CHARS]
    ]
    per_chunk = max(1, -(-count // len(chunks)))  # ceil division

    examples: list[Example] = []
    seen_instructions: set = set()

    for content in chunks:
        if len(examples) >= count:
            break
        remaining = count - len(examples)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(source, content, min(per_chunk, remaining), fmt, topic)},
        ]
        try:
            data = client.chat_json(model, messages, temperature=temperature, max_tokens=2048)
        except Exception as exc:  # noqa: BLE001 - one chunk failing should not kill the source
            logger.warning("Generation failed for %s: %s", source.url, exc)
            continue

        raw = data.get("examples", []) if isinstance(data, dict) else (data or [])
        for item in raw:
            if not isinstance(item, dict):
                continue
            instruction, response = clean_fields(
                str(item.get("instruction", "")), str(item.get("response", ""))
            )
            reason = validate_example(instruction, response)
            if reason:
                logger.debug("Dropped generated example from %s: %s", source.url, reason)
                continue
            key = instruction.lower()
            if key in seen_instructions:
                continue
            seen_instructions.add(key)
            examples.append(
                Example(
                    instruction=instruction,
                    response=response,
                    source_url=source.url,
                    source_title=source.title,
                    topic=topic,
                    format=fmt,
                )
            )
            if len(examples) >= count:
                break

    return examples[:count]
