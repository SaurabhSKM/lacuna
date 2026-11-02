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
        verdict = CompletenessVerdict(
            action=action,
            # "accept" is only valid when the answer actually resolved
            # the question - a supported refusal is not an acceptance.
            answer_resolves_question=(action == "accept"),
            sufficiency_score=0.5,
            # Required by the schema for anything other than "accept" -
            # the critic must always name the gap it wants closed.
            gap_description=None if action == "accept" else "a specific missing figure",
            reasoning="test",
        )
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
    assert report["by_category"] == {}


def test_per_category_accuracy_is_broken_out():
    # A single accuracy number hides which category regressed, and the
    # three fail in completely different ways.
    benchmark = [
        {"question": "a1", "category": "answerable"},
        {"question": "a2", "category": "answerable"},
        {"question": "u1", "category": "unanswerable"},
    ]
    results = [
        make_result("accepted", "accept"),
        make_result("needs_enrichment_capped", "enrich"),  # wrong
        make_result("needs_enrichment_capped", "enrich"),
    ]
    report = score(benchmark, results)
    assert report["by_category"]["answerable"]["accuracy"] == 0.5
    assert report["by_category"]["unanswerable"]["accuracy"] == 1.0


def test_an_error_is_never_correct_even_when_not_accepted():
    # The harness bug this pins: correctness for "unanswerable" was
    # defined as `final_status != "accepted"`, so a crashed pipeline
    # scored as having correctly refused to fabricate. A provider outage
    # mid-run would have quietly improved the headline number.
    benchmark = [{"question": "u1", "category": "unanswerable"}]
    results = [{"final_status": "error", "completeness_verdict": None, "error": "boom"}]
    report = score(benchmark, results)
    assert report["accuracy"] == 0.0
    assert report["errors"] == ["u1"]
    assert report["error_rate"] == 1.0


def test_an_error_is_not_counted_as_a_false_negative():
    # It's a failure to produce a verdict, not a fabrication - the two
    # need different remedies, so they shouldn't share a number.
    benchmark = [{"question": "u1", "category": "unanswerable"}]
    results = [{"final_status": "error", "completeness_verdict": None, "error": "boom"}]
    report = score(benchmark, results)
    assert report["false_negatives"] == []
    assert report["false_negative_rate"] == 0.0


def test_an_error_on_an_answerable_question_is_also_incorrect():
    benchmark = [{"question": "a1", "category": "answerable"}]
    results = [{"final_status": "error", "completeness_verdict": None, "error": "boom"}]
    assert score(benchmark, results)["accuracy"] == 0.0


def test_error_message_is_recorded_for_diagnosis():
    benchmark = [{"question": "q1", "category": "answerable"}]
    results = [{"final_status": "error", "completeness_verdict": None, "error": "429 rate limit"}]
    assert score(benchmark, results)["rows"][0]["error"] == "429 rate limit"


def test_a_clean_run_reports_no_errors():
    benchmark = [{"question": "q1", "category": "answerable"}]
    report = score(benchmark, [make_result("accepted", "accept")])
    assert report["errors"] == []
    assert report["error_rate"] == 0.0


def test_enrichment_rounds_are_totalled():
    benchmark = [{"question": "q1", "category": "answerable"}]
    results = [{**make_result("accepted", "accept"), "iteration_count": 2}]
    assert score(benchmark, results)["enrichment_rounds"] == 2
