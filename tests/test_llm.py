"""Tests for JSON extraction, retry classification and the retry loop."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from nemotron_data_forge.errors import ConfigError, LLMResponseError
from nemotron_data_forge.llm import TokenFactoryClient, extract_json, is_retryable


# --- extract_json ----------------------------------------------------------
def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"examples": []}\n```') == {"examples": []}


def test_extract_json_with_prose():
    assert extract_json('Here: {"scores": [{"index": 0}]} done')["scores"][0]["index"] == 0


def test_extract_json_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_braces_inside_strings():
    payload = '{"text": "use { and } safely"}'
    assert extract_json(payload)["text"] == "use { and } safely"


def test_extract_json_empty_raises():
    with pytest.raises(LLMResponseError):
        extract_json("   ")


def test_extract_json_garbage_raises():
    with pytest.raises(LLMResponseError):
        extract_json("no json here at all")


# --- is_retryable ----------------------------------------------------------
def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.example/v1"))


def test_is_retryable_rate_limit():
    exc = openai.RateLimitError("slow down", response=_response(429), body=None)
    assert is_retryable(exc) is True


def test_is_retryable_server_error():
    exc = openai.InternalServerError("boom", response=_response(503), body=None)
    assert is_retryable(exc) is True


def test_is_retryable_client_error():
    exc = openai.BadRequestError("bad", response=_response(400), body=None)
    assert is_retryable(exc) is False


def test_is_retryable_value_error():
    assert is_retryable(ValueError("nope")) is False


# --- client behavior -------------------------------------------------------
class _FakeCompletions:
    def __init__(self, script):
        self._script = script
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        action = self._script.pop(0) if self._script else "ok"
        if isinstance(action, Exception):
            raise action
        return SimpleNamespace(
            model=kwargs.get("model", "m"),
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
            choices=[SimpleNamespace(message=SimpleNamespace(content=action))],
        )


def _client_with(script):
    client = TokenFactoryClient(api_key="test-key", max_retries=3, sleep=lambda _s: None)
    completions = _FakeCompletions(list(script))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_missing_key_raises():
    with pytest.raises(ConfigError):
        TokenFactoryClient(api_key="")


def test_chat_retries_then_succeeds():
    client, completions = _client_with(
        [openai.InternalServerError("boom", response=_response(503), body=None), '{"ok": true}']
    )
    assert client.chat("m", [{"role": "user", "content": "hi"}]) == '{"ok": true}'
    assert completions.calls == 2


def test_chat_gives_up_after_max_retries():
    client, completions = _client_with(
        [
            openai.InternalServerError("boom", response=_response(503), body=None),
            openai.InternalServerError("boom", response=_response(503), body=None),
            openai.InternalServerError("boom", response=_response(503), body=None),
        ]
    )
    with pytest.raises(openai.InternalServerError):
        client.chat("m", [{"role": "user", "content": "hi"}])
    assert completions.calls == 3


def test_chat_does_not_retry_bad_request():
    client, completions = _client_with([openai.BadRequestError("bad", response=_response(400), body=None)])
    with pytest.raises(openai.BadRequestError):
        client.chat("m", [{"role": "user", "content": "hi"}])
    assert completions.calls == 1


def test_usage_recorded():
    client, _ = _client_with(['{"ok": true}'])
    client.chat("m", [{"role": "user", "content": "hi"}])
    assert client.usage.total_tokens() == 7


def test_chat_json_repairs_non_json():
    client, completions = _client_with(["not json", "still not json", '{"fixed": true}'])
    assert client.chat_json("m", [{"role": "user", "content": "hi"}]) == {"fixed": True}
    assert completions.calls == 3


def test_embed_returns_none_on_failure():
    client = TokenFactoryClient(api_key="test-key", max_retries=1, sleep=lambda _s: None)

    def boom(**kwargs):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.example/v1"))

    client._client = SimpleNamespace(embeddings=SimpleNamespace(create=boom))
    assert client.embed(["a", "b"]) is None


def test_malformed_json_from_model_raises_response_error():
    from nemotron_data_forge.llm import extract_json as ej

    with pytest.raises(LLMResponseError):
        ej("totally not json")
