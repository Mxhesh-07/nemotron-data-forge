"""Tests for the cost estimator used by the budget cap and manifest."""

from nemotron_data_forge.pricing import estimate_cost, has_price


def test_has_price_distinguishes_known_models():
    assert has_price("nvidia/Nemotron-3-Super-120b-a12b") is True
    assert has_price("some/unknown-model") is False


def test_nebius_models_are_priced():
    assert estimate_cost("nvidia/Nemotron-3-Super-120b-a12b", 1000, 1000) > 0
    assert estimate_cost("nvidia/Nemotron-3-Ultra-550b-a55b", 1000, 1000) > 0
    assert estimate_cost("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", 1000, 1000) > 0


def test_nvidia_nim_fallback_models_are_priced():
    # Every model the NIM preset uses must be priced, or budget accounting
    # silently reads 0 for the whole run.
    assert estimate_cost("nvidia/nemotron-3-super-120b-a12b", 1000, 1000) > 0
    assert estimate_cost("nvidia/nemotron-3-ultra-550b-a55b", 1000, 1000) > 0
    assert estimate_cost("nvidia/nemotron-3.5-lightning-30b-a3b", 1000, 1000) > 0
    assert estimate_cost("nvidia/nemotron-3-embed-1b", 1000, 0) > 0


def test_openrouter_fallback_models_reuse_price_table():
    # The OpenRouter preset uses the same Nemotron ids as NIM, so cost accounting
    # must resolve them (case-insensitively) instead of silently reading $0.
    assert estimate_cost("nvidia/nemotron-3-super-120b-a12b", 1000, 1000) > 0
    assert estimate_cost("nvidia/nemotron-3-ultra-550b-a55b", 1000, 1000) > 0
    assert estimate_cost("nvidia/nemotron-3-embed-1b", 1000, 0) > 0


def test_pricing_is_case_insensitive():
    # Token Factory returns CamelCase ids, NIM lowercase ones; both must price.
    assert has_price("nvidia/nemotron-3-super-120b-a12b") is True
    assert has_price("NVIDIA/NEMOTRON-3-SUPER-120B-A12B") is True
    assert estimate_cost("nvidia/nemotron-3-super-120b-a12b", 1000, 1000) == estimate_cost(
        "nvidia/Nemotron-3-Super-120b-a12b", 1000, 1000
    )


def test_unknown_model_is_free_not_a_crash():
    assert estimate_cost("some/unknown-model", 1000, 1000) == 0.0


def test_cost_scales_with_tokens():
    cheap = estimate_cost("nvidia/Nemotron-3-Super-120b-a12b", 1000, 0)
    pricey = estimate_cost("nvidia/Nemotron-3-Super-120b-a12b", 2000, 0)
    assert pricey > cheap
