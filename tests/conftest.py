"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
