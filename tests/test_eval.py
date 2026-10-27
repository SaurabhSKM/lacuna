"""
score() is pure - given a benchmark and a list of already-computed
results, it just aggregates. That means the scoring logic itself is
fully unit-testable without ever calling an LLM or running the graph;
only run_single() (which does invoke the real graph) needs live access.
These tests hand-build result dicts with the same shape app.invoke()
returns, so the aggregation logic can be verified in isolation.
"""
from eval.run_eval import score
from src.schemas import CompletenessVerdict


def make_result(final_status, action=None):
    verdict = None
    if action:
        verdict = CompletenessVerdict(action=action, sufficiency_score=0.5, reasoning="test")
    return {"final_status": final_status, "completeness_verdict": verdict}


def test_answerable_correctly_accepted():
    benchmark = [{"question": "q1", "category": "answerable"}]
    results = [make_result("accepted", "accept")]
    report = score(benchmark, results)
    assert report["accuracy"] == 1.0
    assert report["false_negative_rate"] == 0.0


def test_answerable_wrongly_not_accepted():
    benchmark = [{"question": "q1", "category": "answerable"}]
    results = [make_result("needs_enrichment_capped", "enrich")]
    report = score(benchmark, results)
    assert report["accuracy"] == 0.0


def test_unanswerable_wrongly_accepted_is_a_false_negative():
    benchmark = [{"question": "q1", "category": "unanswerable"}]
    results = [make_result("accepted", "accept")]
    report = score(benchmark, results)
    assert report["accuracy"] == 0.0
    assert report["false_negative_rate"] == 1.0
    assert "q1" in report["false_negatives"]


def test_unanswerable_correctly_not_accepted():
    benchmark = [{"question": "q1", "category": "unanswerable"}]
    results = [make_result("needs_enrichment_capped", "enrich")]
    report = score(benchmark, results)
    assert report["accuracy"] == 1.0
    assert report["false_negative_rate"] == 0.0
    assert report["false_negatives"] == []


def test_ambiguous_correctly_routed_to_ask_user():
    benchmark = [{"question": "q1", "category": "ambiguous"}]
    results = [make_result("needs_clarification", "ask_user")]
    report = score(benchmark, results)
    assert report["accuracy"] == 1.0


def test_ambiguous_wrongly_accepted():
    benchmark = [{"question": "q1", "category": "ambiguous"}]
    results = [make_result("accepted", "accept")]
    report = score(benchmark, results)
    assert report["accuracy"] == 0.0


def test_mixed_batch_accuracy_and_false_negative_rate():
    benchmark = [
        {"question": "a1", "category": "answerable"},
        {"question": "u1", "category": "unanswerable"},
        {"question": "u2", "category": "unanswerable"},
    ]
    results = [
        make_result("accepted", "accept"),                  # correct
        make_result("accepted", "accept"),                  # WRONG - false negative
        make_result("needs_enrichment_capped", "enrich"),   # correct
    ]
    report = score(benchmark, results)
    assert report["accuracy"] == 2 / 3
    assert report["false_negative_rate"] == 1 / 2  # 1 of 2 unanswerable wrongly accepted
    assert report["false_negatives"] == ["u1"]


def test_false_negative_rate_is_zero_when_no_unanswerable_questions():
    benchmark = [{"question": "q1", "category": "answerable"}]
    results = [make_result("accepted", "accept")]
    report = score(benchmark, results)
    assert report["false_negative_rate"] == 0.0


def test_empty_benchmark_does_not_crash():
    report = score([], [])
    assert report["accuracy"] == 0.0
    assert report["false_negative_rate"] == 0.0
    assert report["rows"] == []
