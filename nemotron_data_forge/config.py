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


# --- Nebius Token Factory --------------------------------------------------
NEBIUS_BASE_URL = _env("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
NEBIUS_API_KEY = _env("NEBIUS_API_KEY")

# --- Tavily ----------------------------------------------------------------
TAVILY_API_KEY = _env("TAVILY_API_KEY")
TAVILY_URL = _env("TAVILY_URL", "https://api.tavily.com/search")

# --- Models ----------------------------------------------------------------
GENERATOR_MODEL = _env("GENERATOR_MODEL", "nvidia/Nemotron-3-Super-120b-a12b")
GRADER_MODEL = _env("GRADER_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")
FAST_MODEL = _env("FAST_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

# --- Pipeline defaults -----------------------------------------------------
DEFAULT_TOPIC = _env("DEFAULT_TOPIC", "Indian tax law for freelancers")
DEFAULT_FORMAT = _env("DEFAULT_FORMAT", "Q&A")
DEFAULT_EXAMPLES_PER_SOURCE = _env_int("DEFAULT_EXAMPLES_PER_SOURCE", 3)
DEFAULT_MAX_SOURCES = _env_int("DEFAULT_MAX_SOURCES", 4)
DEFAULT_KEEP_THRESHOLD = _env_float("DEFAULT_KEEP_THRESHOLD", 7.0)
DEFAULT_MIN_AXIS_SCORE = _env_float("DEFAULT_MIN_AXIS_SCORE", 5.0)
DEFAULT_DEDUPE_THRESHOLD = _env_float("DEFAULT_DEDUPE_THRESHOLD", 0.82)

# --- Limits ----------------------------------------------------------------
REQUEST_TIMEOUT = _env_float("REQUEST_TIMEOUT", 120.0)
MAX_RETRIES = _env_int("MAX_RETRIES", 4)
RETRY_BACKOFF_BASE = _env_float("RETRY_BACKOFF_BASE", 1.6)
RETRY_BACKOFF_CAP = _env_float("RETRY_BACKOFF_CAP", 20.0)
MAX_CONCURRENCY = _env_int("MAX_CONCURRENCY", 4)
GRADE_BATCH_SIZE = _env_int("GRADE_BATCH_SIZE", 8)

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
    """Return a usable Nebius key or raise ConfigError."""
    key = api_key or NEBIUS_API_KEY
    if not key or not key.strip():
        raise ConfigError(
            "Missing NEBIUS_API_KEY. Get one at https://tokenfactory.nebius.com and set it "
            "in your environment, a .env file, or the app sidebar."
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
