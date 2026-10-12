"""Tests for chunking and validation helpers."""

import pytest

from nemotron_data_forge.chunking import chunk_text
from nemotron_data_forge.validation import (
    clamp_score,
    clean_fields,
    normalize_scores,
    validate_example,
)


def test_chunk_text_empty():
    assert chunk_text("", 100) == []
    assert chunk_text("   ", 100) == []


def test_chunk_text_short():
    assert chunk_text("hello world", 100) == ["hello world"]


def test_chunk_text_splits_paragraphs():
    text = "para one\n\npara two\n\npara three"
    chunks = chunk_text(text, 15)
    assert all(len(c) <= 15 for c in chunks)
    assert "".join(chunks).replace("\n\n", "").replace(" ", "") == text.replace("\n\n", "").replace(" ", "")


def test_chunk_text_hard_split_long_paragraph():
    text = "x" * 250
    chunks = chunk_text(text, 100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunk_text_rejects_bad_limit():
    with pytest.raises(ValueError):
        chunk_text("abc", 0)


def test_clean_fields():
    assert clean_fields("  a   b ", "  line1  \n\nline2  ") == ("a b", "line1\n\nline2")


def test_validate_example_ok():
    assert validate_example("What is GST?", "It is a tax on goods and services.") is None


def test_validate_example_rejects_empty_and_short():
    assert validate_example("", "answer") == "empty instruction or response"
    assert validate_example("short", "answer") is not None


def test_validate_example_rejects_identical():
    text = "this is the same text"
    assert validate_example(text, text) == "instruction and response are identical"


def test_validate_example_rejects_too_long():
    assert validate_example("q" * 3000, "a" * 50) is not None


def test_clamp_score():
    assert clamp_score(7.126) == 7.13
    assert clamp_score(-5) == 0.0
    assert clamp_score(99) == 10.0
    assert clamp_score("not a number") == 0.0
    assert clamp_score(None) == 0.0


def test_normalize_scores():
    assert normalize_scores(11, -1, "5") == {"accuracy": 10.0, "helpfulness": 0.0, "clarity": 5.0}
