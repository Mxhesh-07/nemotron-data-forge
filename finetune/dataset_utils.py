"""Loading, splitting and formatting datasets for before/after fine-tuning.

Pure, dependency-free helpers so they can be unit-tested without torch.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REQUIRED_KEYS = ("instruction", "output")

_OPTIONAL_STRING_KEYS = ("input", "source_url", "source_title", "topic", "format")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a dataset.jsonl file produced by the pipeline (or a fixture).

    Accepts both the pipeline's schema (``instruction``/``output``) and a
    plain ``instruction``/``response`` shape. Raises ``ValueError`` with the
    offending line number on malformed input.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    examples: list[dict[str, Any]] = []
    for line_number, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{line_number}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{p}:{line_number}: expected a JSON object")

        # Normalise the response field name.
        if "output" not in obj and "response" in obj:
            obj["output"] = obj["response"]

        missing = [k for k in REQUIRED_KEYS if not str(obj.get(k, "")).strip()]
        if missing:
            raise ValueError(f"{p}:{line_number}: missing or empty field(s): {', '.join(missing)}")

        record: dict[str, Any] = {
            "instruction": str(obj["instruction"]).strip(),
            "output": str(obj["output"]).strip(),
        }
        for key in _OPTIONAL_STRING_KEYS:
            record[key] = str(obj.get(key, "") or "").strip()
        examples.append(record)

    if not examples:
        raise ValueError(f"{p}: no examples found")
    return examples


def split(
    examples: Sequence[dict[str, Any]], eval_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle and split into (train, eval).

    Guarantees a non-empty train set and, when there are at least two examples,
    a non-empty eval set.
    """
    if not examples:
        raise ValueError("Cannot split an empty dataset")
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1)")

    data = list(examples)
    random.Random(seed).shuffle(data)

    if len(data) == 1:
        return data, []

    n_eval = round(len(data) * eval_ratio)
    n_eval = max(1, min(n_eval, len(data) - 1))
    return data[n_eval:], data[:n_eval]


def build_prompt(instruction: str, input_text: str = "") -> str:
    """The instruction prompt (no answer) used at inference time."""
    text = instruction.strip()
    if input_text.strip():
        text = f"{text}\n{input_text.strip()}"
    return f"### Instruction:\n{text}\n\n### Response:\n"


def build_training_text(example: dict[str, Any], eos_token: str = "") -> str:
    """Supervised fine-tuning text for one example (prompt + answer)."""
    prompt = build_prompt(example["instruction"], example.get("input", ""))
    return f"{prompt}{example['output'].strip()}{eos_token}"


def format_counts(examples: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count examples per ``format`` value, useful in run summaries."""
    return dict(Counter((e.get("format") or "unknown").strip() or "unknown" for e in examples))
