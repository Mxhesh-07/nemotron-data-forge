"""Near-duplicate removal using token similarity or semantic embeddings."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from .schemas import Example

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _token_set(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity of the word sets of two strings."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def dedupe(
    examples: list[Example],
    threshold: float = 0.82,
    embeddings: list[list[float] | None] | None = None,
) -> list[Example]:
    """Return examples with near-duplicates removed.

    Similarity is measured as cosine of embeddings when supplied (same order as
    ``examples``), otherwise as token-set Jaccard. Marked duplicates get
    ``verdict="duplicate"`` and an explanatory ``reason``.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")

    kept: list[Example] = []
    kept_text: list[str] = []
    kept_vecs: list[list[float] | None] = []

    for i, example in enumerate(examples):
        text = _normalize(f"{example.instruction} {example.response}")
        vec = embeddings[i] if embeddings and i < len(embeddings) else None

        duplicate_of: int | None = None
        best_sim = 0.0
        for j in range(len(kept)):
            if vec is not None and kept_vecs[j] is not None:
                sim = _cosine(vec, kept_vecs[j])  # type: ignore[arg-type]
            else:
                sim = jaccard(text, kept_text[j])
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                duplicate_of = j

        if duplicate_of is None:
            kept.append(example)
            kept_text.append(text)
            kept_vecs.append(vec)
        else:
            example.verdict = "duplicate"
            example.reason = f"near-duplicate of kept example #{duplicate_of + 1} (sim={best_sim:.2f})"

    return kept
