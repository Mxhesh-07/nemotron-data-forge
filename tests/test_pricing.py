"""Tests for the cost estimator used by the budget cap and manifest."""

from nemotron_data_forge.pricing import estimate_cost


def test_nebius_models_are_priced():
    assert estimate_cost("nvidia/Nemotron-3-Super-120b-a12b", 1000, 1000) > 0
    assert estimate_cost("nvidia/Nemotron-3-Ultra-550b-a55b", 1000, 1000) > 0
    assert estimate_cost("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", 1000, 1000) > 0


def test_nvidia_nim_fallback_models_are_priced():
    # The NIM fast model must be priced too, or budget accounting silently reads 0.
    assert estimate_cost("nvidia/nemotron-3.5-lightning-30b-a3b", 1000, 1000) > 0


def test_unknown_model_is_free_not_a_crash():
    assert estimate_cost("some/unknown-model", 1000, 1000) == 0.0


def test_cost_scales_with_tokens():
    cheap = estimate_cost("nvidia/Nemotron-3-Super-120b-a12b", 1000, 0)
    pricey = estimate_cost("nvidia/Nemotron-3-Super-120b-a12b", 2000, 0)
    assert pricey > cheap
