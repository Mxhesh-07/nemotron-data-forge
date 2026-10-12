"""Tests for the before/after report and the fine-tune CLI --check path."""

import json

from finetune import train_lora
from finetune.report import ComparisonRow, render_markdown, save, score_rows


def _rows():
    return [
        ComparisonRow(
            instruction="What is GST?",
            reference="GST is a tax on goods and services.",
            before="I'm not sure, it could be many things maybe.",
            after="GST is a tax on goods and services.",
            fmt="Q&A",
            source_url="https://example.com/gst",
        ),
        ComparisonRow(
            instruction="How is advance tax paid?",
            reference="Advance tax is paid in quarterly instalments.",
            before="Advance tax is paid in quarterly instalments.",
            after="Advance tax is paid in quarterly instalments.",
            fmt="Q&A",
        ),
    ]


def test_score_rows_detects_improvement():
    metrics = score_rows(_rows())
    assert metrics["n"] == 2
    assert metrics["after"]["token_f1"] > metrics["before"]["token_f1"]
    assert metrics["delta"]["token_f1"] > 0


def test_render_markdown_contains_sections():
    rows = _rows()
    md = render_markdown(rows, score_rows(rows), {"base_model": "tiny", "epochs": 3})
    assert "# Before/After Fine-Tuning Report" in md
    assert "Token F1 vs reference" in md
    assert "**Before (base model):**" in md
    assert "**After (fine-tuned):**" in md
    assert "https://example.com/gst" in md
    assert "`tiny`" in md


def test_save_writes_both_files(tmp_path):
    rows = _rows()
    metrics = score_rows(rows)
    md_path, json_path = save(rows, metrics, tmp_path, {"base_model": "tiny"})
    assert md_path.exists() and json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["meta"]["base_model"] == "tiny"
    assert len(payload["rows"]) == 2


def test_resolve_dataset_glob(tmp_path):
    (tmp_path / "dataset-1.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "dataset-2.jsonl").write_text("{}\n", encoding="utf-8")
    resolved = train_lora.resolve_dataset(str(tmp_path / "dataset-*.jsonl"))
    assert resolved.endswith("dataset-2.jsonl")


def test_resolve_dataset_glob_no_match():
    import pytest

    with pytest.raises(FileNotFoundError):
        train_lora.resolve_dataset("nope-*.jsonl")


def test_check_mode_runs_without_torch():
    # The --check path must not import torch/transformers.
    code = train_lora.main(["--dataset", "finetune/sample_dataset.jsonl", "--check"])
    assert code == 0


def test_check_mode_missing_dataset_returns_2(capsys):
    code = train_lora.main(["--dataset", "no/such/file.jsonl", "--check"])
    assert code == 2
    assert "error:" in capsys.readouterr().err
