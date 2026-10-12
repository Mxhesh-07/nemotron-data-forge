"""Tests for the Tavily search wrapper."""

import pytest

from nemotron_data_forge import config
from nemotron_data_forge import search as search_module
from nemotron_data_forge.errors import ConfigError, SearchError


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    with pytest.raises(ConfigError):
        search_module.tavily_search("tax", api_key="")


def test_empty_query_raises():
    with pytest.raises(SearchError):
        search_module.tavily_search("", api_key="key")


def test_parse_results(monkeypatch):
    payload = {
        "results": [
            {"title": "A", "url": "https://a.com", "content": "aaa", "score": 0.9},
            {"title": "", "url": "https://b.com", "content": "bbb", "score": 0.8},
            {"title": "no content", "url": "https://c.com", "content": "", "score": 0.7},
        ]
    }
    monkeypatch.setattr(search_module.requests, "post", lambda *a, **k: _Resp(200, payload))
    sources = search_module.tavily_search("tax", api_key="key")
    assert len(sources) == 2  # the empty-content result is dropped
    assert sources[0].title == "A"
    assert sources[1].title == "https://b.com"  # falls back to url


def test_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(429, text="rate limited")
        return _Resp(200, {"results": [{"title": "A", "url": "https://a.com", "content": "x"}]})

    monkeypatch.setattr(search_module.requests, "post", post)
    sources = search_module.tavily_search("tax", api_key="key", sleep=lambda _s: None)
    assert calls["n"] == 2
    assert len(sources) == 1


def test_permanent_error_raises(monkeypatch):
    monkeypatch.setattr(search_module.requests, "post", lambda *a, **k: _Resp(401, text="unauthorized"))
    with pytest.raises(SearchError):
        search_module.tavily_search("tax", api_key="key", sleep=lambda _s: None)


def test_exhausted_retries_raise(monkeypatch):
    monkeypatch.setattr(search_module.requests, "post", lambda *a, **k: _Resp(503, text="down"))
    with pytest.raises(SearchError):
        search_module.tavily_search("tax", api_key="key", max_retries=2, sleep=lambda _s: None)


def test_invalid_json_raises(monkeypatch):
    def bad_json():
        raise ValueError("no json")

    resp = _Resp(200)
    resp.json = bad_json  # type: ignore[assignment]
    monkeypatch.setattr(search_module.requests, "post", lambda *a, **k: resp)
    with pytest.raises(SearchError):
        search_module.tavily_search("tax", api_key="key", sleep=lambda _s: None)


def test_non_numeric_score_defaults_to_zero(monkeypatch):
    payload = {
        "results": [
            {"title": "A", "url": "https://a.com", "content": "x", "score": "high"},
            {"title": "B", "url": "https://b.com", "content": "y"},
        ]
    }
    monkeypatch.setattr(search_module.requests, "post", lambda *a, **k: _Resp(200, payload))
    sources = search_module.tavily_search("tax", api_key="key", sleep=lambda _s: None)
    assert [s.score for s in sources] == [0.0, 0.0]
