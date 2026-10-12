"""Tests for the soft USD budget cap."""

from fakes import FakeClient, fake_search

from nemotron_data_forge.config import DEFAULT_MAX_USD
from nemotron_data_forge.pipeline import run_pipeline

TINY = 1e-6  # one generation call already costs ~2.1e-5 USD


def test_unlimited_budget_is_the_default():
    assert DEFAULT_MAX_USD == 0.0


def test_budget_stops_new_sources_and_skips_rewrite():
    client = FakeClient(score=4.0)  # everything is rejected -> rewrite would normally run
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=2,
        keep_threshold=7.0,
        client=client,
        search_fn=fake_search(2),
        use_embeddings=False,
        max_concurrency=1,
        max_rewrite_rounds=1,
        max_usd=TINY,
    )
    assert result.manifest.budget_hit is True
    assert result.manifest.budget_usd == TINY
    # Only the first source was generated before the cap cancelled the rest.
    assert result.stats.generated == 1
    # Rewrite is an optional phase and must be skipped when over budget.
    assert result.stats.rewritten == 0
    assert any("rewrite-on-reject" in e for e in result.manifest.errors)


def test_budget_skips_embeddings_but_keeps_graded_examples():
    client = FakeClient(score=9.0)
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=3,
        client=client,
        search_fn=fake_search(3),
        use_embeddings=True,
        max_concurrency=1,
        max_usd=TINY,
    )
    assert result.manifest.budget_hit is True
    assert result.stats.kept >= 1
    assert "embeddings" not in result.manifest.usage
    assert any("de-duplication" in e for e in result.manifest.errors)


def test_no_cap_runs_everything():
    client = FakeClient(score=9.0)
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=2,
        client=client,
        search_fn=fake_search(2),
        use_embeddings=True,
        max_concurrency=1,
        max_usd=0.0,
    )
    assert result.manifest.budget_hit is False
    assert result.manifest.budget_usd == 0.0
    assert result.stats.generated == 2
    assert "embeddings" in result.manifest.usage


def test_generous_cap_is_not_hit():
    client = FakeClient(score=9.0)
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=2,
        client=client,
        search_fn=fake_search(2),
        use_embeddings=False,
        max_concurrency=1,
        max_usd=1000.0,
    )
    assert result.manifest.budget_hit is False
    assert result.stats.generated == 2
    assert result.manifest.errors == []
