"""Test doubles so the whole pipeline can be exercised without any network."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable

from nemotron_data_forge.schemas import Source
from nemotron_data_forge.usage import UsageTracker


class FakeClient:
    """Stands in for TokenFactoryClient.

    Detects whether a prompt is a generation or grading request and returns
    well-formed JSON, tracking usage like the real client. Set ``fail_marker``
    to make any request whose prompt contains that marker raise.
    """

    def __init__(
        self,
        score: float = 9.0,
        fail_marker: str | None = None,
        examples: Callable[[str], list[dict]] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.usage = UsageTracker()
        self.calls: list[tuple] = []
        self._counter = 0
        self._lock = threading.Lock()
        self.score = score
        self.fail_marker = fail_marker
        self._examples = examples
        self._embeddings = embeddings

    # -- helpers ------------------------------------------------------------
    def _next(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _generate(self, text: str) -> dict:
        match = re.search(r"Create (\d+) high", text)
        count = int(match.group(1)) if match else 1
        if self._examples is not None:
            return {"examples": self._examples(text)}
        out = []
        for _ in range(count):
            n = self._next()
            out.append(
                {"instruction": f"Question number {n} about the topic?",
                 "response": f"Grounded answer number {n}."}
            )
        return {"examples": out}

    def _grade(self, text: str) -> dict:
        n = text.count("INSTRUCTION:")
        rows = [
            {
                "index": i,
                "accuracy": self.score,
                "helpfulness": self.score,
                "clarity": self.score,
                "reason": "looks good",
            }
            for i in range(n)
        ]
        return {"scores": rows}

    # -- public API ---------------------------------------------------------
    def chat_json(self, model: str, messages, temperature: float = 0.0, max_tokens: int = 0, **kwargs):
        text = "\n".join(m.get("content", "") for m in messages)
        with self._lock:
            self.calls.append((model, text))
        if self.fail_marker and self.fail_marker in text:
            raise RuntimeError(f"fake failure for marker {self.fail_marker!r}")
        self.usage.add(model, prompt_tokens=10, completion_tokens=20)
        if "Grade each example" in text:
            return self._grade(text)
        return self._generate(text)

    def chat(self, model: str, messages, **kwargs) -> str:  # pragma: no cover - unused
        return ""

    def embed(self, texts, model: str | None = None):
        self.usage.add(model or "embeddings", prompt_tokens=len(texts), completion_tokens=0)
        if self._embeddings is not None:
            return self._embeddings
        return [[float(len(t) % 97), float((sum(ord(c) for c in t)) % 89), 1.0] for t in texts]


def fake_search(count: int = 3):
    """Return a search function that yields ``count`` deterministic sources."""

    def _search(query: str, max_results: int = 4, api_key=None):
        sources = [
            Source(
                title=f"Source {i}",
                url=f"https://example.com/s{i}",
                content=f"Content for source {i}. " * 20,
                score=0.9 - i * 0.05,
            )
            for i in range(count)
        ]
        return sources[: max(1, max_results)]

    return _search
