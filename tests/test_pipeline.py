"""End-to-end pipeline tests using fakes (no network)."""

from fakes import FakeClient, fake_search

from nemotron_data_forge.pipeline import run_pipeline


def test_pipeline_end_to_end():
    result = run_pipeline(
        topic="tax",
        examples_per_source=2,
        max_sources=3,
        keep_threshold=7.0,
        client=FakeClient(score=9.0),
        search_fn=fake_search(3),
        use_embeddings=False,
        max_concurrency=2,
    )
    assert result.stats.sources_used == 3
    assert result.stats.generated == 6
    assert result.stats.kept == 6
    assert result.stats.rejected == 0
    assert result.stats.avg_accuracy == 9.0
    assert result.manifest.total_tokens > 0
    assert result.manifest.estimated_cost_usd >= 0.0
    assert result.manifest.duration_seconds >= 0.0
    assert result.manifest.config["examples_per_source"] == 2


def test_pipeline_rejects_low_scores():
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=2,
        keep_threshold=8.0,
        client=FakeClient(score=5.0),
        search_fn=fake_search(2),
        use_embeddings=False,
        max_rewrite_rounds=0,
    )
    assert result.stats.kept == 0
    assert result.stats.rejected == 2
    assert result.stats.generated == 2


def test_pipeline_dedupes_identical_examples():
    client = FakeClient(examples=lambda _t: [{"instruction": "Same Q here?", "response": "Same A here."}])
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=2,
        keep_threshold=7.0,
        client=client,
        search_fn=fake_search(2),
        use_embeddings=False,
    )
    assert result.stats.kept == 1
    assert result.stats.duplicates_removed == 1


def test_pipeline_sample_mode():
    result = run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=5,
        client=FakeClient(score=9.0),
        use_sample_sources=True,
        use_embeddings=False,
    )
    assert result.stats.sources_used == 3  # only three bundled samples exist
    assert result.stats.kept == 3


def test_pipeline_records_search_failure():
    def boom(*a, **k):
        raise RuntimeError("search down")

    result = run_pipeline(
        topic="tax",
        client=FakeClient(score=9.0),
        search_fn=boom,
        use_embeddings=False,
    )
    assert result.stats.sources_used == 0
    assert any("search failed" in e for e in result.manifest.errors)


def test_pipeline_reports_progress_to_100():
    seen = []
    run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=2,
        client=FakeClient(score=9.0),
        search_fn=fake_search(2),
        use_embeddings=False,
        on_progress=lambda pct, msg: seen.append(pct),
    )
    assert seen[0] <= 100 and seen[-1] == 100.0
    assert seen == sorted(seen)


def test_pipeline_progress_is_monotonic_through_rewrite():
    # First grade rejects, then the rewrite is graded high -> the rewrite phase runs.
    seen = []
    run_pipeline(
        topic="tax",
        examples_per_source=1,
        max_sources=1,
        keep_threshold=7.0,
        client=FakeClient(scores_sequence=[4.0, 9.0]),
        search_fn=fake_search(1),
        use_embeddings=False,
        max_concurrency=1,
        max_rewrite_rounds=1,
        on_progress=lambda pct, _msg: seen.append(pct),
    )
    assert seen == sorted(seen)
    assert seen[-1] == 100.0

