"""Tests for de-duplication."""

import pytest

from nemotron_data_forge.dedupe import dedupe, jaccard


def test_jaccard_identical_and_disjoint():
    assert jaccard("hello world", "hello world") == 1.0
    assert jaccard("apple banana", "car truck") == 0.0
    assert jaccard("", "anything") == 0.0


def test_dedupe_by_tokens(make_example):
    a = make_example("What is GST in India?", "GST is a tax on goods and services.")
    b = make_example("What is GST in India?", "GST is a tax on goods and services.")
    c = make_example("How do I pay advance tax?", "Pay it in quarterly instalments.")
    kept = dedupe([a, b, c], threshold=0.82)
    assert len(kept) == 2
    assert b.verdict == "duplicate"
    assert "near-duplicate" in b.reason


def test_dedupe_by_embeddings(make_example):
    a = make_example("A?", "one")
    b = make_example("B?", "two")
    c = make_example("C?", "three")
    kept = dedupe([a, b, c], threshold=0.9, embeddings=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert len(kept) == 2
    assert b.verdict == "duplicate"


def test_dedupe_threshold_validation(make_example):
    with pytest.raises(ValueError):
        dedupe([make_example()], threshold=2.0)


def test_dedupe_keeps_first_of_group(make_example):
    a = make_example("Same?", "Same answer.")
    b = make_example("Same?", "Same answer.")
    kept = dedupe([a, b], threshold=0.8)
    assert kept == [a]
