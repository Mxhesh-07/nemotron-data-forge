"""Tests for the `dataforge --check` doctor."""

from fakes import FakeClient, fake_search

from nemotron_data_forge import config
from nemotron_data_forge.cli import main
from nemotron_data_forge.doctor import check_environment


def _by_name(checks):
    return {c.name: c for c in checks}


def test_missing_nebius_key_stops_early(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "")
    checks = check_environment(client=None)
    assert len(checks) == 1
    assert checks[0].ok is False
    assert "NEBIUS_API_KEY" in checks[0].name


def test_all_checks_pass_with_fakes(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "x")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "y")
    checks = check_environment(client=FakeClient(), search_fn=fake_search(1))
    failed = [(c.name, c.detail) for c in checks if not c.ok]
    assert failed == []
    names = {c.name for c in checks}
    assert {"Chat round-trip", "Embeddings round-trip", "Tavily search round-trip"} <= names


def test_unknown_model_is_reported(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "x")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    client = FakeClient()
    client.list_models = lambda: ["some/other-model"]  # type: ignore[method-assign]
    checks = _by_name(check_environment(client=client, run_search=False, require_tavily=False))
    assert checks["generator model available"].ok is False
    assert checks["Chat round-trip"].ok is True


def test_chat_failure_is_reported(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "x")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    client = FakeClient(fail_marker="Reply with the single word")
    checks = _by_name(check_environment(client=client, run_search=False, require_tavily=False))
    assert checks["Chat round-trip"].ok is False


def test_demo_mode_does_not_require_tavily(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "x")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    checks = _by_name(check_environment(client=FakeClient(), run_search=False, require_tavily=False))
    assert checks["TAVILY_API_KEY present"].ok is True


def test_cli_check_without_key_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "")
    code = main(["--check"])
    assert code == 1
    out = capsys.readouterr().out
    assert "NEBIUS_API_KEY" in out
