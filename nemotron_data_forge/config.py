"""Central configuration. Every value can be overridden by an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import ConfigError

try:  # dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv missing
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} must be a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Provider --------------------------------------------------------------
# "nebius" is the default and the runtime required for the hackathon
# (Nebius Token Factory). "nvidia" points the same OpenAI-compatible client at
# NVIDIA's free NIM trial endpoint, so the pipeline can be exercised card-free
# during development. It is NOT a substitute for the Token Factory run in the
# final submission.
LLM_PROVIDER = _env("LLM_PROVIDER", "nebius").strip().lower()

_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "nebius": {
        "label": "Nebius Token Factory",
        "base_url": "https://api.tokenfactory.nebius.com/v1/",
        "key_env": "NEBIUS_API_KEY",
        "key_url": "https://tokenfactory.nebius.com",
        "generator": "nvidia/Nemotron-3-Super-120b-a12b",
        "grader": "nvidia/Nemotron-3-Ultra-550b-a55b",
        "fast": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "embedding": "Qwen/Qwen3-Embedding-8B",
    },
    "nvidia": {
        "label": "NVIDIA NIM (development fallback)",
        "base_url": "https://integrate.api.nvidia.com/v1/",
        "key_env": "NVIDIA_API_KEY",
        "key_url": "https://build.nvidia.com",
        "generator": "nvidia/nemotron-3-super-120b-a12b",
        "grader": "nvidia/nemotron-3-ultra-550b-a55b",
        "fast": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "embedding": "nvidia/nemotron-3-embed-1b",
    },
}


def provider_settings(provider: str | None = None) -> dict[str, str]:
    """Return the resolved settings preset for a provider name."""
    name = (provider if provider is not None else LLM_PROVIDER).strip().lower()
    if name not in _PROVIDER_PRESETS:
        raise ConfigError(
            f"Unknown LLM_PROVIDER {name!r}. Choose one of {sorted(_PROVIDER_PRESETS)}."
        )
    return dict(_PROVIDER_PRESETS[name])


LLM_PROVIDER = _env("LLM_PROVIDER", "nebius").strip().lower()
_PRESET = provider_settings(LLM_PROVIDER)
LLM_KEY_ENV = _PRESET["key_env"]
LLM_PROVIDER_LABEL = _PRESET["label"]

# NVIDIA Nemotron 3 models "think" by default, which can spend the whole token
# budget on a reasoning trace before the final JSON answer. Disable it by
# default when using the NVIDIA fallback; override with LLM_DISABLE_THINKING.
LLM_DISABLE_THINKING = _env_bool("LLM_DISABLE_THINKING", LLM_PROVIDER == "nvidia")

# --- OpenAI-compatible endpoint (Nebius Token Factory by default) -----------
NEBIUS_BASE_URL = _env("NEBIUS_BASE_URL", _PRESET["base_url"])
# The provider-specific key wins; the other is accepted as a fallback so that
# switching providers is a one-line change to LLM_PROVIDER.
NEBIUS_API_KEY = _env(_PRESET["key_env"]) or _env("NEBIUS_API_KEY") or _env("NVIDIA_API_KEY")

# --- Tavily ----------------------------------------------------------------
TAVILY_API_KEY = _env("TAVILY_API_KEY")
TAVILY_URL = _env("TAVILY_URL", "https://api.tavily.com/search")

# --- Models ----------------------------------------------------------------
GENERATOR_MODEL = _env("GENERATOR_MODEL", _PRESET["generator"])
GRADER_MODEL = _env("GRADER_MODEL", _PRESET["grader"])
FAST_MODEL = _env("FAST_MODEL", _PRESET["fast"])
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", _PRESET["embedding"])

# --- Pipeline defaults -----------------------------------------------------
DEFAULT_TOPIC = _env("DEFAULT_TOPIC", "Indian tax law for freelancers")
DEFAULT_FORMAT = _env("DEFAULT_FORMAT", "Q&A")
DEFAULT_EXAMPLES_PER_SOURCE = _env_int("DEFAULT_EXAMPLES_PER_SOURCE", 3)
DEFAULT_MAX_SOURCES = _env_int("DEFAULT_MAX_SOURCES", 4)
DEFAULT_KEEP_THRESHOLD = _env_float("DEFAULT_KEEP_THRESHOLD", 7.0)
DEFAULT_MIN_AXIS_SCORE = _env_float("DEFAULT_MIN_AXIS_SCORE", 5.0)
DEFAULT_DEDUPE_THRESHOLD = _env_float("DEFAULT_DEDUPE_THRESHOLD", 0.82)
DEFAULT_REWRITE_ROUNDS = _env_int("DEFAULT_REWRITE_ROUNDS", 1)
# Soft spend cap in USD for a single run. 0 means unlimited.
DEFAULT_MAX_USD = _env_float("DEFAULT_MAX_USD", 0.0)

# --- Limits ----------------------------------------------------------------
REQUEST_TIMEOUT = _env_float("REQUEST_TIMEOUT", 120.0)
MAX_RETRIES = _env_int("MAX_RETRIES", 4)
RETRY_BACKOFF_BASE = _env_float("RETRY_BACKOFF_BASE", 1.6)
RETRY_BACKOFF_CAP = _env_float("RETRY_BACKOFF_CAP", 20.0)
MAX_CONCURRENCY = _env_int("MAX_CONCURRENCY", 4)
GRADE_BATCH_SIZE = _env_int("GRADE_BATCH_SIZE", 8)
# Max completion tokens for generation/grading. Reasoning-capable models count
# their thinking tokens toward this budget, so leave headroom.
MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 4096)

MAX_SOURCE_CHARS = _env_int("MAX_SOURCE_CHARS", 6000)
SOURCE_CHUNK_CHARS = _env_int("SOURCE_CHUNK_CHARS", 3000)

MIN_INSTRUCTION_CHARS = _env_int("MIN_INSTRUCTION_CHARS", 8)
MIN_RESPONSE_CHARS = _env_int("MIN_RESPONSE_CHARS", 8)
MAX_INSTRUCTION_CHARS = _env_int("MAX_INSTRUCTION_CHARS", 2000)
MAX_RESPONSE_CHARS = _env_int("MAX_RESPONSE_CHARS", 8000)

VALID_FORMATS: list[str] = ["Q&A", "Chat", "Reasoning"]

# --- Output ----------------------------------------------------------------
OUTPUT_DIR = Path(_env("OUTPUT_DIR", "outputs"))
LOG_LEVEL = _env("LOG_LEVEL", "INFO")

APP_NAME = "nemotron-data-forge"


def require_nebius_key(api_key: str | None = None) -> str:
    """Return a usable provider API key or raise ConfigError."""
    key = api_key or NEBIUS_API_KEY
    if not key or not key.strip():
        raise ConfigError(
            f"Missing {LLM_KEY_ENV} (provider: {LLM_PROVIDER_LABEL}). Get one at "
            f"{_PRESET['key_url']} and set it in your environment, a .env file, "
            "or the app sidebar."
        )
    return key.strip()


def require_tavily_key(api_key: str | None = None) -> str:
    """Return a usable Tavily key or raise ConfigError."""
    key = api_key or TAVILY_API_KEY
    if not key or not key.strip():
        raise ConfigError(
            "Missing TAVILY_API_KEY. Get a free key at https://tavily.com (the Nebius "
            "Builder Program includes $25 of Tavily credit)."
        )
    return key.strip()


def validate_format(fmt: str) -> str:
    if fmt not in VALID_FORMATS:
        raise ConfigError(f"Unknown format {fmt!r}. Choose one of {VALID_FORMATS}.")
    return fmt
