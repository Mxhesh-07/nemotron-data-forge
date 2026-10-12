"""Typed exceptions for Nemotron Data Forge."""

from __future__ import annotations


class DataForgeError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(DataForgeError):
    """Invalid or missing configuration (e.g. absent API key)."""


class LLMError(DataForgeError):
    """A call to Nebius Token Factory failed after retries."""


class LLMResponseError(LLMError):
    """The model returned a response that could not be parsed as valid JSON."""


class SearchError(DataForgeError):
    """A Tavily search request failed."""


class ValidationError(DataForgeError):
    """An example failed validation before it could be exported."""
