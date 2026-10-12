"""Tests for the grader and threshold logic."""

from fakes import FakeClient

from nemotron_data_forge.grader import apply_threshold, grade_examples
from nemotron_data_forge.schemas import Example, Source


def _source():
    return Source(title="T", url="https://example.com/t", content="content", score=0.9)


def _example(i=1):
    return Example(
        instruction=f"Question {i}?",
        response=f"Answer {i}.",
        source_url="https://example.com/t",
        source_title="T",
        topic="tax",
        format="Q&A",
    )


def test_grade_examples_attaches_scores():
    examples = [_example(i) for i in range(3)]
    grade_examples(FakeClient(score=8.0), _source(), examples)
    assert all(e.average_score == 8.0 for e in examples)
    assert all(e.scores["accuracy"] == 8.0 for e in examples)


def test_grading_batches_across_multiple_calls():
    client = FakeClient(score=9.0)
    examples = [_example(i) for i in range(20)]
    grade_examples(client, _source(), examples, batch_size=8)
    assert all(e.average_score == 9.0 for e in examples)
    grade_calls = [c for c in client.calls if "Grade each example" in c[1]]
    assert len(grade_calls) == 3  # 8 + 8 + 4


def test_ungraded_example_is_rejected():
    client = FakeClient(examples=None)

    # Return a grading payload with a missing index for one batch.
    client.chat_json = lambda model, messages, **kw: {  # type: ignore[assignment]
        "scores": [{"index": 0, "accuracy": 9, "helpfulness": 9, "clarity": 9}]
    }
    examples = [_example(1), _example(2)]
    grade_examples(client, _source(), examples)
    assert examples[0].verdict != "rejected" or examples[0].average_score == 9.0
    assert examples[1].verdict == "rejected"
    assert examples[1].reason == "grader returned no score"


def test_grading_failure_marks_batch_rejected():
    examples = [_example(1)]
    grade_examples(FakeClient(fail_marker="Grade each example"), _source(), examples)
    assert examples[0].verdict == "rejected"
    assert examples[0].scores == {}


def test_apply_threshold_keeps_and_rejects():
    good = _example(1)
    good.scores = {"accuracy": 9, "helpfulness": 9, "clarity": 9}
    good.average_score = 9.0
    low_avg = _example(2)
    low_avg.scores = {"accuracy": 6, "helpfulness": 6, "clarity": 6}
    low_avg.average_score = 6.0
    low_axis = _example(3)
    low_axis.scores = {"accuracy": 9, "helpfulness": 9, "clarity": 3}
    low_axis.average_score = 7.0

    examples = [good, low_avg, low_axis]
    apply_threshold(examples, threshold=7.0, min_axis=5.0)

    assert good.verdict == "kept"
    assert low_avg.verdict == "rejected"
    assert "average" in low_avg.reason
    assert low_axis.verdict == "rejected"
    assert "axis" in low_axis.reason


def test_apply_threshold_leaves_ungraded_rejected():
    ex = _example(1)
    ex.scores = {}
    ex.verdict = "rejected"
    ex.reason = "grader returned no score"
    apply_threshold([ex], threshold=7.0)
    assert ex.verdict == "rejected"
    assert ex.reason == "grader returned no score"
