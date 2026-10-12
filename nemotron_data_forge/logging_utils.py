"""Logging setup shared by the CLI, pipeline and app."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once, writing to stderr."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
