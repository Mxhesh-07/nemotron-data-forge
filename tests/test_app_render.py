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


def test_app_generates_a_dataset_with_a_fake_client(monkeypatch):
    """Click through the UI end to end, with the model client faked."""
    from fakes import FakeClient

    from nemotron_data_forge import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "TokenFactoryClient", lambda **kwargs: FakeClient(score=9.0)
    )

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    next(i for i in at.text_input if i.label == "NEBIUS_API_KEY").set_value("test-key")
    next(c for c in at.checkbox if c.label.startswith("Demo mode")).set_value(True)
    at.button[0].click()
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert any("kept" in s.value for s in at.success)
    metric_labels = [m.label for m in at.metric]
    assert "Kept" in metric_labels
    assert "Not kept" in metric_labels
