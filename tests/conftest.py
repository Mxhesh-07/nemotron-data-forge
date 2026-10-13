"""Shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Keep the test suite hermetic: never read the developer's real .env and never
# hit live APIs. Setting these (even to empty) before the package is imported
# stops python-dotenv from filling them in from .env.
os.environ["LLM_PROVIDER"] = "nebius"
os.environ["NEBIUS_API_KEY"] = ""
os.environ["NVIDIA_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""
os.environ["LLM_DISABLE_THINKING"] = "0"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nemotron_data_forge.schemas import Example  # noqa: E402


@pytest.fixture
def make_example():
    def _make(instruction="Q?", response="A.", **kwargs):
        base = {
            "instruction": instruction,
            "response": response,
            "source_url": "https://example.com/x",
            "source_title": "X",
            "topic": "t",
            "format": "Q&A",
        }
        base.update(kwargs)
        return Example(**base)

    return _make
