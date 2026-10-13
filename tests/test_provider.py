"""Tests for the OpenAI-compatible provider switch (Nebius default, NVIDIA/OpenRouter dev fallbacks)."""

import pytest

from nemotron_data_forge import config
from nemotron_data_forge.config import provider_settings
from nemotron_data_forge.errors import ConfigError


def test_nebius_preset_is_the_hackathon_runtime():
    preset = provider_settings("nebius")
    assert preset["base_url"] == "https://api.tokenfactory.nebius.com/v1/"
    assert preset["key_env"] == "NEBIUS_API_KEY"
    assert preset["generator"] == "nvidia/Nemotron-3-Super-120b-a12b"
    assert preset["grader"] == "nvidia/Nemotron-3-Ultra-550b-a55b"


def test_nvidia_preset_points_at_nim():
    preset = provider_settings("nvidia")
    assert preset["base_url"] == "https://integrate.api.nvidia.com/v1/"
    assert preset["key_env"] == "NVIDIA_API_KEY"
    assert preset["generator"].startswith("nvidia/nemotron-3-")


def test_openrouter_preset_points_at_openrouter():
    preset = provider_settings("openrouter")
    assert preset["base_url"] == "https://openrouter.ai/api/v1/"
    assert preset["key_env"] == "OPENROUTER_API_KEY"
    assert preset["generator"].startswith("nvidia/nemotron-3-")
    assert preset["embedding"] == "nvidia/nemotron-3-embed-1b"


def test_unknown_provider_raises():
    with pytest.raises(ConfigError):
        provider_settings("openai")


def test_provider_lookup_is_case_insensitive():
    assert provider_settings("NEBIUS")["key_env"] == "NEBIUS_API_KEY"


def test_active_provider_matches_module_state():
    active = provider_settings(config.LLM_PROVIDER)
    assert config.LLM_KEY_ENV == active["key_env"]
    assert config.LLM_PROVIDER_LABEL == active["label"]
