"""Thread-safe token-usage and cost accounting for a run."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .pricing import estimate_cost


@dataclass
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class UsageTracker:
    """Accumulates usage per model across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, ModelUsage] = {}

    def add(self, model: str, prompt_tokens: int, completion_tokens: int, calls: int = 1) -> None:
        with self._lock:
            usage = self._by_model.setdefault(model, ModelUsage())
            usage.calls += calls
            usage.prompt_tokens += max(0, int(prompt_tokens or 0))
            usage.completion_tokens += max(0, int(completion_tokens or 0))

    def total_cost_usd(self) -> float:
        with self._lock:
            return round(
                sum(
                    estimate_cost(m, u.prompt_tokens, u.completion_tokens)
                    for m, u in self._by_model.items()
                ),
                6,
            )

    def total_tokens(self) -> int:
        with self._lock:
            return sum(u.total_tokens for u in self._by_model.values())

    def as_dict(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {model: usage.to_dict() for model, usage in self._by_model.items()}

    def reset(self) -> None:
        with self._lock:
            self._by_model.clear()
