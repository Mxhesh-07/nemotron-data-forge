"""Tests for export helpers."""

import json

import pytest

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


def _examples(make_example, n):
    return [_example(make_example, i) for i in range(1, n + 1)]


def test_export_formats_have_expected_shapes(make_example):
    from nemotron_data_forge.export import EXPORT_FORMATS, records_jsonl

    assert EXPORT_FORMATS == ["alpaca", "sharegpt", "openai", "raw"]
    ex = [_example(make_example, 1)]

    alpaca = json.loads(records_jsonl(ex, "alpaca"))
    assert alpaca["instruction"] == "Question 1?" and alpaca["output"] == "Answer 1."
    assert alpaca["format"] == "Q&A" and alpaca["scores"]["accuracy"] == 9.0

    share = json.loads(records_jsonl(ex, "sharegpt"))
    assert share["conversations"][0]["from"] == "human"
    assert share["conversations"][1]["value"] == "Answer 1."

    oai = json.loads(records_jsonl(ex, "openai"))
    assert oai["messages"][0]["role"] == "user"
    assert oai["messages"][1]["content"] == "Answer 1."

    raw = json.loads(records_jsonl(ex, "raw"))
    assert raw["response"] == "Answer 1." and "output" not in raw


def test_records_jsonl_rejects_unknown_format(make_example):
    from nemotron_data_forge.export import records_jsonl

    with pytest.raises(ValueError):
        records_jsonl([_example(make_example)], "csv")


def test_to_jsonl_honours_format(make_example, tmp_path):
    path = to_jsonl(_examples(make_example, 2), tmp_path / "d.jsonl", fmt="openai")
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "messages" in first


def test_split_dataset(make_example):
    from nemotron_data_forge.export import split_dataset

    ex = _examples(make_example, 10)
    train, val = split_dataset(ex, 0.2, seed=1)
    assert len(val) == 2 and len(train) == 8
    assert {id(e) for e in train}.isdisjoint({id(e) for e in val})
    # Deterministic for a given seed.
    assert [e.instruction for e in val] == [e.instruction for e in split_dataset(ex, 0.2, 1)[1]]


def test_split_dataset_edge_cases(make_example):
    from nemotron_data_forge.export import split_dataset

    assert split_dataset([], 0.2) == ([], [])
    one = _examples(make_example, 1)
    assert split_dataset(one, 0.5) == (one, [])
    ten = _examples(make_example, 10)
    assert split_dataset(ten, 0.0)[1] == []
    # Never leave train empty, even for a huge ratio.
    train, val = split_dataset(_examples(make_example, 5), 0.99)
    assert len(train) >= 1 and len(val) == 4
