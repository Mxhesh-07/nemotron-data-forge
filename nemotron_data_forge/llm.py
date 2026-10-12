"""Robust client for Nebius Token Factory (OpenAI-compatible API).

Features:
- exponential backoff with jitter for connection errors, timeouts, 429 and 5xx
- JSON mode with graceful fallback and a single repair attempt
- thread-safe token usage / cost accounting
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from . import config
from .errors import LLMError, LLMResponseError
from .logging_utils import get_logger
from .usage import UsageTracker

logger = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def extract_json(text: str) -> Any:
    """Extract a JSON value from a model response.

    Handles ```json fences and prose surrounding a JSON object/array.
    Raises LLMResponseError when nothing parseable is found.
    """
    if text is None or not str(text).strip():
        raise LLMResponseError("Empty model response")

    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise LLMResponseError(f"Could not parse JSON from model output: {cleaned[:300]!r}")


def is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that are worth retrying."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) in _RETRYABLE_STATUS
    return False


class TokenFactoryClient:
    """OpenAI-compatible client for Nebius Token Factory with retries."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        usage: UsageTracker | None = None,
        sleep=time.sleep,
    ) -> None:
        key = config.require_nebius_key(api_key)
        self.base_url = base_url or config.NEBIUS_BASE_URL
        self.max_retries = max(1, max_retries if max_retries is not None else config.MAX_RETRIES)
        self.timeout = timeout if timeout is not None else config.REQUEST_TIMEOUT
        self.usage = usage if usage is not None else UsageTracker()
        self._sleep = sleep
        # We implement retries ourselves, so disable the SDK's.
        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout, max_retries=0)

    # -- internals ----------------------------------------------------------
    def _backoff(self, attempt: int) -> float:
        delay = min(config.RETRY_BACKOFF_BASE ** attempt, config.RETRY_BACKOFF_CAP)
        return delay * (0.5 + random.random() * 0.5)

    def _run_with_retries(self, fn, description: str):
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                if not is_retryable(exc) or attempt == self.max_retries - 1:
                    raise
                delay = self._backoff(attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                    description,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                self._sleep(delay)
        raise LLMError(f"{description} failed after {self.max_retries} attempts: {last_exc}")

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        model = getattr(response, "model", None) or "unknown"
        if usage is None:
            return
        self.usage.add(
            model,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    # -- public API ---------------------------------------------------------
    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """Single chat completion (with retries). Returns the text content."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if config.LLM_DISABLE_THINKING:
            # NVIDIA Nemotron 3 chat template supports turning the reasoning
            # trace off; keep responses short and JSON-safe.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        def call():
            try:
                return self._client.chat.completions.create(**kwargs)
            except BadRequestError:
                # Endpoints differ in what they accept: some reject
                # response_format, others reject the chat_template_kwargs
                # extension. Drop the optional parameters and retry once.
                dropped = [key for key in ("response_format", "extra_body") if key in kwargs]
                if not dropped:
                    raise
                for key in dropped:
                    kwargs.pop(key, None)
                logger.warning(
                    "Extended options %s rejected by %s; retrying with base parameters",
                    dropped,
                    model,
                )
                return self._client.chat.completions.create(**kwargs)

        response = self._run_with_retries(call, f"chat({model})")
        self._record_usage(response)
        if not response.choices:
            return ""
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning(
                "Response from %s was truncated at max_tokens=%s; raise MAX_OUTPUT_TOKENS "
                "or disable reasoning if JSON keeps getting cut off",
                model,
                max_tokens,
            )
        return choice.message.content or ""

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        required_key: str | None = None,
    ) -> Any:
        """Chat completion that must return JSON.

        Tries JSON mode, then plain mode, and finally one repair turn before
        giving up. When ``required_key`` is set, the parsed value must be a JSON
        object containing that key.
        """
        def _validate(parsed):
            if required_key and (not isinstance(parsed, dict) or required_key not in parsed):
                raise LLMResponseError(f"expected a JSON object with key {required_key!r}")
            return parsed

        content = self.chat(model, messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        try:
            return _validate(extract_json(content))
        except LLMResponseError:
            logger.warning("Model %s returned non-JSON; retrying without json_mode", model)

        content = self.chat(model, messages, temperature=temperature, max_tokens=max_tokens, json_mode=False)
        try:
            return _validate(extract_json(content))
        except LLMResponseError:
            logger.warning("Repairing malformed JSON from %s", model)

        repair_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "That was not valid JSON. Reply with ONLY the valid JSON, no prose or code fences.",
            },
        ]
        repaired = self.chat(model, repair_messages, temperature=0.0, max_tokens=max_tokens, json_mode=False)
        return _validate(extract_json(repaired))

    def list_models(self) -> list[str]:
        """Return the model IDs the Token Factory account can access."""
        response = self._run_with_retries(lambda: self._client.models.list(), "models.list")
        items = getattr(response, "data", response) or []
        return [mid for item in items if (mid := getattr(item, "id", None))]

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]] | None:
        """Return embeddings, or None if the endpoint is unavailable."""
        if not texts:
            return []
        target = model or config.EMBEDDING_MODEL

        def call():
            return self._client.embeddings.create(model=target, input=texts)

        try:
            response = self._run_with_retries(call, f"embed({target})")
        except Exception as exc:  # noqa: BLE001 - embeddings are optional
            logger.warning("Embeddings unavailable (%s); falling back to token dedupe", exc)
            return None

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.add(target, getattr(usage, "prompt_tokens", 0) or 0, 0)
        return [item.embedding for item in response.data]
