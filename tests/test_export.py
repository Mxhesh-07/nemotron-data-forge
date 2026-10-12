"""Tests for export helpers."""

import json

from nemotron_data_forge.export import (
    dataset_card,
    quality_report,
    save_json,
    save_text,
    to_jsonl,
)
from nemotron_data_forge.schemas import PipelineStats, RunManifest


def _example(make_example, i=1):
    ex = make_example(f"Question {i}?", f"Answer {i}.")
    ex.scores = {"accuracy": 9.0, "helpfulness": 9.0, "clarity": 9.0}
    ex.average_score = 9.0
    ex.verdict = "kept"
    return ex


def test_to_jsonl_roundtrip(make_example, tmp_path):
    examples = [_example(make_example, 1), _example(make_example, 2)]
    path = to_jsonl(examples, tmp_path / "dataset.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["instruction"] == "Question 1?"
    assert record["output"] == "Answer 1."
    assert record["source_url"] == "https://example.com/x"
    assert record["scores"]["accuracy"] == 9.0


def test_quality_report_contains_key_sections(make_example):
    stats = PipelineStats(topic="tax", format="Q&A", kept=1, generated=2, rejected=1, sources_used=1)
    manifest = RunManifest(
        estimated_cost_usd=0.01,
        total_tokens=123,
        duration_seconds=4.2,
        usage={
            "nvidia/Nemotron-3-Super-120b-a12b": {
                "calls": 1,
                "prompt_tokens": 60,
                "completion_tokens": 63,
                "total_tokens": 123,
            }
        },
    )
    report = quality_report(
        [_example(make_example, 1)], [_example(make_example, 2)], stats, [], manifest
    )
    assert "# Dataset Quality Report" in report
    assert "Token usage" in report
    assert "## Sources" not in report  # no sources supplied
    assert "Rejected" in report


def test_dataset_card_has_front_matter():
    stats = PipelineStats(topic="tax", kept=3)
    card = dataset_card("Indian tax law", stats, RunManifest())
    assert card.startswith("---")
    assert "license: mit" in card
    assert "Indian tax law" in card


def test_save_text_and_json(tmp_path):
    p1 = save_text("hello", tmp_path / "sub" / "a.txt")
    p2 = save_json({"a": 1}, tmp_path / "sub" / "b.json")
    assert p1.read_text(encoding="utf-8") == "hello"
    assert json.loads(p2.read_text(encoding="utf-8")) == {"a": 1}
