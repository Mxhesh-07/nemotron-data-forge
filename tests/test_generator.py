"""Tests for the generator."""

from fakes import FakeClient

from nemotron_data_forge.generator import generate_examples
from nemotron_data_forge.schemas import Source


def _source(content="Some real content about taxes. " * 10):
    return Source(title="T", url="https://example.com/t", content=content, score=0.9)


def test_generates_requested_count():
    client = FakeClient()
    examples = generate_examples(client, _source(), topic="tax", count=3)
    assert len(examples) == 3
    assert all(e.source_url == "https://example.com/t" for e in examples)
    assert all(e.topic == "tax" for e in examples)


def test_prompt_contains_source_and_guidance():
    client = FakeClient()
    generate_examples(client, _source("UNIQUE_SOURCE_TEXT"), topic="tax", fmt="Reasoning", count=1)
    _, prompt = client.calls[0]
    assert "UNIQUE_SOURCE_TEXT" in prompt
    assert "step by step" in prompt


def test_invalid_examples_are_dropped():
    client = FakeClient(examples=lambda _t: [{"instruction": "x", "response": "y"}])
    examples = generate_examples(client, _source(), topic="tax", count=3)
    assert examples == []


def test_duplicate_instructions_deduped_within_source():
    client = FakeClient(
        examples=lambda _t: [
            {"instruction": "Same question here?", "response": "Answer one is fine."},
            {"instruction": "Same question here?", "response": "Answer two is fine."},
        ]
    )
    examples = generate_examples(client, _source(), topic="tax", count=5)
    assert len(examples) == 1


def test_zero_count_returns_empty():
    client = FakeClient()
    assert generate_examples(client, _source(), topic="tax", count=0) == []


def test_generation_failure_is_contained():
    client = FakeClient(fail_marker="Some real content")
    examples = generate_examples(client, _source(), topic="tax", count=2)
    assert examples == []


def test_long_source_is_capped_to_max_source_chars(monkeypatch):
    """Only MAX_SOURCE_CHARS // SOURCE_CHUNK_CHARS chunks may be sent."""
    from nemotron_data_forge import config

    monkeypatch.setattr(config, "MAX_SOURCE_CHARS", 3000)
    monkeypatch.setattr(config, "SOURCE_CHUNK_CHARS", 1000)
    content = ("x" * 999 + " ") * 10  # ten 1000-char chunks
    client = FakeClient()
    examples = generate_examples(client, _source(content), topic="tax", count=99)
    assert len(client.calls) == 3  # 3000 // 1000
    assert len(examples) > 0

