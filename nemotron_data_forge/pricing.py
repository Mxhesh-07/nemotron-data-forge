"""Token pricing for the models used, so runs can report an estimated cost.

Prices are USD per 1,000,000 tokens, taken from the public Token Factory
catalog. They are estimates only and can change; unknown models cost 0.
"""

from __future__ import annotations

# model_id -> (input_usd_per_1m, output_usd_per_1m)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "nvidia/Nemotron-3-Super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    # NVIDIA NIM development-fallback fast model (same family as the Nano model).
    "nvidia/nemotron-3.5-lightning-30b-a3b": (0.06, 0.24),
    "Qwen/Qwen3-Embedding-8B": (0.01, 0.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the estimated USD cost for a single model's token usage."""
    price_in, price_out = MODEL_PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens / 1_000_000.0) * price_in + (completion_tokens / 1_000_000.0) * price_out
