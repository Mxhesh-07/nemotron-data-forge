"""Tests for the offline fine-tune helpers (no torch required)."""

import pytest

from finetune import dataset_utils, scoring
from finetune.dataset_utils import (
    build_prompt,
    build_training_text,
    format_counts,
    load_jsonl,
    split,
)

SAMPLE = "finetune/sample_dataset.jsonl"


# --- dataset_utils ---------------------------------------------------------
def test_load_sample_dataset():
    examples = load_jsonl(SAMPLE)
    assert len(examples) == 20
    assert all(ex["instruction"] and ex["output"] for ex in examples)


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_jsonl("does/not/exist.jsonl")


def test_load_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"instruction": "q", "output": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_jsonl(path)


def test_load_missing_field(tmp_path):
    path = tmp_path / "missing.jsonl"
    path.write_text('{"instruction": "q"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing or empty"):
        load_jsonl(path)


def test_load_accepts_response_alias(tmp_path):
    path = tmp_path / "alias.jsonl"
    path.write_text('{"instruction": "q", "response": "a"}\n', encoding="utf-8")
    examples = load_jsonl(path)
    assert examples[0]["output"] == "a"


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no examples"):
        load_jsonl(path)


def test_split_sizes_and_disjoint():
    examples = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(10)]
    train, eval_set = split(examples, eval_ratio=0.2, seed=1)
    assert len(eval_set) == 2
    assert len(train) == 8
    assert not ({id(x) for x in train} & {id(x) for x in eval_set})


def test_split_single_example():
    train, eval_set = split([{"instruction": "q", "output": "a"}])
    assert len(train) == 1
    assert eval_set == []


def test_split_bad_ratio():
    with pytest.raises(ValueError):
        split([{"instruction": "q", "output": "a"}], eval_ratio=1.5)


def test_build_prompt_and_training_text():
    example = {"instruction": "What is GST?", "input": "", "output": "A tax."}
    prompt = build_prompt("What is GST?")
    assert prompt.startswith("### Instruction:\n")
    assert prompt.endswith("### Response:\n")
    text = build_training_text(example, eos_token="<|end|>")
    assert text == prompt + "A tax.<|end|>"


def test_build_prompt_with_input():
    prompt = build_prompt("Summarise", "some context")
    assert "Summarise\nsome context" in prompt


def test_format_counts():
    examples = [{"format": "Q&A"}, {"format": "Q&A"}, {"format": ""}]
    assert format_counts(examples) == {"Q&A": 2, "unknown": 1}


# --- scoring ---------------------------------------------------------------
def test_normalize():
    assert scoring.normalize("Hello, World! 42") == ["hello", "world", "42"]


def test_token_f1():
    assert scoring.token_f1("the quick brown fox", "the quick brown fox") == 1.0
    assert scoring.token_f1("apple banana", "car truck") == 0.0
    assert scoring.token_f1("", "") == 1.0
    assert scoring.token_f1("something", "") == 0.0


def test_keyword_recall():
    ref = "Freelancers must pay advance tax in quarterly instalments"
    assert scoring.keyword_recall(ref, ref) == 1.0
    assert scoring.keyword_recall("nothing relevant", ref) == 0.0


def test_length_closeness():
    assert scoring.length_closeness("one two three", "one two three") == 1.0
    assert scoring.length_closeness("x " * 50, "one two three") < 0.2


def test_format_adherence():
    assert scoring.format_adherence("", "Q&A") == 0.0
    assert scoring.format_adherence("An answer.", "Q&A") == 1.0
    assert scoring.format_adherence("just one line", "Reasoning") == 0.0
    assert scoring.format_adherence("step one\nstep two", "Reasoning") == 1.0


def test_evaluate_and_aggregate():
    scores = scoring.evaluate("the answer", "the answer")
    assert scores["token_f1"] == 1.0
    assert scores["words"] == 2.0
    agg = scoring.aggregate([scores, scores])
    assert agg["token_f1"] == 1.0
    assert scoring.aggregate([]) == {}


def test_dataset_utils_and_scoring_import_without_torch():
    # If this module imported torch, the test session would already have failed.
    assert hasattr(dataset_utils, "load_jsonl")
