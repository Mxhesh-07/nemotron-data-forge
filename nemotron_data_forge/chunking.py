"""Text chunking for long source pages."""

from __future__ import annotations


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most ``max_chars``, preferring paragraph breaks.

    Falls back to hard slicing for a single paragraph that is too long.
    """
    text = (text or "").strip()
    if not text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    paragraphs = [p.strip() for p in _split_paragraphs(text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # A single paragraph longer than the limit: slice it.
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[i : i + max_chars])
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    # Normalise newlines, then split on blank lines; if there are none, split on
    # single newlines so we still get sensible boundaries.
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n\n" in normalised:
        return normalised.split("\n\n")
    return normalised.split("\n")
