"""Token pricing for the models used, so runs can report an estimated cost.

Prices are USD per 1,000,000 tokens. The Token Factory figures come from its
public catalog; the NVIDIA NIM development-fallback ids reuse the equivalent
Token Factory prices (plus a nominal embedding price). They are estimates only
and can change; genuinely unknown models cost 0.
"""

from __future__ import annotations

# model_id -> (input_usd_per_1m, output_usd_per_1m)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "nvidia/Nemotron-3-Super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "Qwen/Qwen3-Embedding-8B": (0.01, 0.00),
    # NVIDIA NIM development-fallback models. NIM returns lowercase ids; its
    # Super/Ultra ids resolve case-insensitively to the Token Factory entries
    # above (same underlying models), while the fast and embedding models below
    # are NIM-specific.
    "nvidia/nemotron-3.5-lightning-30b-a3b": (0.06, 0.24),
    "nvidia/nemotron-3-embed-1b": (0.01, 0.00),
}

# Providers spell the same model differently (Token Factory uses CamelCase ids,
# NVIDIA NIM returns lowercase ones). Match case-insensitively so the estimated
# cost and the soft budget cap stay correct whichever provider is configured.
_PRICES_BY_CASEFOLD: dict[str, tuple[float, float]] = {
    model.casefold(): price for model, price in MODEL_PRICES.items()
}


def has_price(model: str) -> bool:
    """Return True when a price is known for the model (case-insensitive)."""
    return model.casefold() in _PRICES_BY_CASEFOLD


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the estimated USD cost for a single model's token usage."""
    price_in, price_out = _PRICES_BY_CASEFOLD.get(model.casefold(), (0.0, 0.0))
    return (prompt_tokens / 1_000_000.0) * price_in + (completion_tokens / 1_000_000.0) * price_out
