"""Tests for the rewrite-on-reject step."""

from fakes import FakeClient, fake_search

from nemotron_data_forge.generator import rewrite_examples
from nemotron_data_forge.pipeline import run_pipeline
from nemotron_data_forge.schemas import Example, Source


def _source(content="Some real content about taxes. " * 10):
    return Source(title="T", url="https://example.com/t", content=content, score=0.9)


def _rejected(i=1):
    return Example(
        instruction=f"Weak question {i}?",
        response=f"Weak answer {i}.",
        source_url="https://example.com/t",
        source_title="T",
        topic="tax",
        format="Q&A",
        scores={"accuracy": 3.0, "helpfulness": 4.0, "clarity": 3.0},
        average_score=3.33,
        verdict="rejected",
        reason="low accuracy",
    )


def test_rewrite_examples_returns_tagged_replacements():
    out = rewrite_examples(FakeClient(), _source(), [_rejected(1), _rejected(2)], topic="tax")
    assert len(out) == 2
    assert all(e.rewritten for e in out)
    assert all(e.verdict == "pending" for e in out)


def test_rewrite_examples_no_input_returns_empty():
    assert rewrite_examples(FakeClient(), _source(), [], topic="tax") == []


def test_rewrite_failure_is_contained():
    client = FakeClient(fail_marker="Some real content")
    assert rewrite_examples(client, _source(), [_rejected(1)], topic="tax") == []


def test_pipeline_recovers_rejected_examples():
    client = FakeClient(scores_sequence=[4.0, 9.0])  # initial grade low, rewrite grade high
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=1,
        keep_threshold=7.0,
        client=client,
        search_fn=fake_search(1),
        use_embeddings=False,
        max_concurrency=1,
        max_rewrite_rounds=1,
    )
    assert result.stats.rewritten == 1
    assert result.stats.recovered == 1
    assert result.stats.kept == 1
    assert result.manifest.config["max_rewrite_rounds"] == 1


def test_pipeline_rewrite_disabled():
    client = FakeClient(score=4.0)
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=1,
        keep_threshold=7.0,
        client=client,
        search_fn=fake_search(1),
        use_embeddings=False,
        max_rewrite_rounds=0,
    )
    assert result.stats.rewritten == 0
    assert result.stats.recovered == 0
    assert result.stats.kept == 0


def test_pipeline_rewrite_only_helps_when_scores_improve():
    # Rewrite produces replacements but they are still graded low -> not recovered.
    client = FakeClient(score=4.0)
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=1,
        keep_threshold=7.0,
        client=client,
        search_fn=fake_search(1),
        use_embeddings=False,
        max_rewrite_rounds=1,
    )
    assert result.stats.rewritten == 1
    assert result.stats.recovered == 0
    assert result.stats.kept == 0
