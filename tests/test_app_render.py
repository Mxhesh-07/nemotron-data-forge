"""Render test for the Streamlit app (no API calls on first render)."""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_renders_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Nemotron Data Forge" in t.value for t in at.title)


def test_app_has_expected_controls():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    labels = [b.label for b in at.button]
    assert any("Generate dataset" in label for label in labels)
    text_inputs = [i.label for i in at.text_input]
    assert "NEBIUS_API_KEY" in text_inputs
    assert "TAVILY_API_KEY" in text_inputs
