"""Tests for the CLI, monkeypatching the client so nothing hits the network."""

import json
from pathlib import Path

import pytest
from fakes import FakeClient

from nemotron_data_forge import pipeline as pipeline_module
from nemotron_data_forge.cli import build_parser, main


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    monkeypatch.setattr(pipeline_module, "TokenFactoryClient", lambda **kwargs: FakeClient(score=9.0))


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.format in ("Q&A", "Chat", "Reasoning")
    assert args.examples_per_source >= 1


def test_cli_demo_run_writes_outputs(tmp_path):
    code = main(
        [
            "--demo",
            "--topic",
            "tax",
            "--examples-per-source",
            "1",
            "--max-sources",
            "2",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 0
    names = {p.name for p in Path(tmp_path).iterdir()}
    assert any(n.startswith("dataset-") and n.endswith(".jsonl") for n in names)
    assert any(n.startswith("quality-report-") and n.endswith(".md") for n in names)
    assert any(n.startswith("manifest-") and n.endswith(".json") for n in names)
    assert any(n.startswith("dataset-card-") and n.endswith(".md") for n in names)


def test_cli_invalid_counts_return_2(tmp_path):
    assert main(["--demo", "--examples-per-source", "0", "--out", str(tmp_path), "--quiet"]) == 2
    assert main(["--demo", "--max-sources", "0", "--out", str(tmp_path), "--quiet"]) == 2


def test_cli_export_format_and_split(tmp_path):
    code = main(
        [
            "--demo",
            "--topic",
            "tax",
            "--examples-per-source",
            "1",
            "--max-sources",
            "4",
            "--out",
            str(tmp_path),
            "--quiet",
            "--export-format",
            "openai",
            "--val-ratio",
            "0.25",
            "--no-embeddings",
        ]
    )
    assert code == 0
    dataset = next(tmp_path.glob("dataset-*.jsonl"))
    val = next(tmp_path.glob("val-*.jsonl"))
    assert "messages" in json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
    assert "messages" in json.loads(val.read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads(next(tmp_path.glob("manifest-*.json")).read_text(encoding="utf-8"))
    assert manifest["config"]["export_format"] == "openai"
    assert manifest["config"]["val_ratio"] == 0.25


def test_cli_rejects_bad_val_ratio(tmp_path):
    assert main(["--demo", "--val-ratio", "1.0", "--out", str(tmp_path), "--quiet"]) == 2
    assert main(["--demo", "--val-ratio=-0.1", "--out", str(tmp_path), "--quiet"]) == 2


def test_cli_manifest_json_flag(tmp_path, capsys):
    code = main(
        [
            "--demo",
            "--topic",
            "tax",
            "--examples-per-source",
            "1",
            "--max-sources",
            "1",
            "--out",
            str(tmp_path),
            "--quiet",
            "--manifest-json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    assert payload["app"] == "nemotron-data-forge"
    assert "usage" in payload
