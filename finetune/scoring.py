"""Offline quality metrics for before/after comparison.

These are deterministic, reference-based heuristics. They are *not* a
replacement for the pipeline's Nemotron grader; they exist so a fine-tune can
be compared to its base model without needing any API keys.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> list[str]:
    """Lower-case word tokens."""
    return _WORD_RE.findall((text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a prediction and the reference answer."""
    pred, ref = normalize(prediction), normalize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(prediction: str, reference: str, top_k: int = 8) -> float:
    """Fraction of the reference's salient keywords that appear in the prediction."""
    words = [w for w in normalize(reference) if len(w) > 3]
    if not words:
        return 0.0
    keywords = [word for word, _ in Counter(words).most_common(top_k)]
    present = set(normalize(prediction))
    return sum(1 for word in keywords if word in present) / len(keywords)


def length_closeness(prediction: str, reference: str) -> float:
    """1.0 when the prediction length matches the reference, dropping to 0 far away."""
    ref_len = max(1, len(normalize(reference)))
    ratio = len(normalize(prediction)) / ref_len
    return max(0.0, 1.0 - abs(ratio - 1.0))


def format_adherence(prediction: str, fmt: str = "") -> float:
    """A light check that a response respects the requested format.

    ``Reasoning`` expects multiple non-empty lines; other formats just require a
    non-empty answer. Returns 1.0 or 0.0.
    """
    text = (prediction or "").strip()
    if not text:
        return 0.0
    if fmt == "Reasoning":
        non_empty_lines = [line for line in text.splitlines() if line.strip()]
        return 1.0 if len(non_empty_lines) >= 2 else 0.0
    return 1.0


def evaluate(prediction: str, reference: str, fmt: str = "") -> dict[str, float]:
    """All metrics for one prediction."""
    return {
        "token_f1": round(token_f1(prediction, reference), 4),
        "keyword_recall": round(keyword_recall(prediction, reference), 4),
        "length_closeness": round(length_closeness(prediction, reference), 4),
        "format_adherence": round(format_adherence(prediction, fmt), 4),
        "words": float(len(normalize(prediction))),
    }


def aggregate(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across a list of per-example score dicts."""
    if not scores:
        return {}
    keys = scores[0].keys()
    n = len(scores)
    return {key: round(sum(s.get(key, 0.0) for s in scores) / n, 4) for key in keys}
