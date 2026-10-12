"""Validation and cleanup of generated examples and grader scores."""

from __future__ import annotations

from . import config


def clean_fields(instruction: str, response: str) -> tuple[str, str]:
    """Strip whitespace and collapse excess blank lines."""
    instruction = " ".join((instruction or "").split())
    response = "\n".join(line.rstrip() for line in (response or "").strip().splitlines())
    return instruction.strip(), response.strip()


def validate_example(instruction: str, response: str) -> str | None:
    """Return a rejection reason if the pair is invalid, else None."""
    if not instruction or not response:
        return "empty instruction or response"
    if len(instruction) < config.MIN_INSTRUCTION_CHARS:
        return f"instruction shorter than {config.MIN_INSTRUCTION_CHARS} chars"
    if len(response) < config.MIN_RESPONSE_CHARS:
        return f"response shorter than {config.MIN_RESPONSE_CHARS} chars"
    if len(instruction) > config.MAX_INSTRUCTION_CHARS:
        return f"instruction longer than {config.MAX_INSTRUCTION_CHARS} chars"
    if len(response) > config.MAX_RESPONSE_CHARS:
        return f"response longer than {config.MAX_RESPONSE_CHARS} chars"
    if instruction.lower() == response.lower():
        return "instruction and response are identical"
    return None


def clamp_score(value, low: float = 0.0, high: float = 10.0) -> float:
    """Coerce a model-provided score into [low, high]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return round(max(low, min(high, number)), 2)


def normalize_scores(accuracy, helpfulness, clarity) -> dict[str, float]:
    return {
        "accuracy": clamp_score(accuracy),
        "helpfulness": clamp_score(helpfulness),
        "clarity": clamp_score(clarity),
    }
