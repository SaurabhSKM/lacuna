"""
The critic's schema constraints and its history formatting.

Both are pure, and both are load-bearing. The gap_description
constraint is what turns "enrich with no search query" into a
self-correcting re-prompt rather than a wasted enrichment round, and the
history block is what stops the critic asking for the same search twice.
"""
import pytest
from pydantic import ValidationError

from src.completeness import format_history
from src.schemas import CompletenessVerdict


def test_accept_needs_no_gap_description():
    verdict = CompletenessVerdict(
        action="accept", answer_resolves_question=True,
        sufficiency_score=0.9, reasoning="context covers it"
    )
    assert verdict.gap_description is None


def test_enrich_without_a_gap_description_is_rejected():
    # The damaging case: with no gap to search for, the enrichment node
    # falls back to re-searching the raw question - the search that
    # already failed to help. Rejecting it here makes
    # invoke_structured() re-prompt with the error attached.
    with pytest.raises(ValidationError, match="gap_description is required"):
        CompletenessVerdict(
            action="enrich", answer_resolves_question=False,
            sufficiency_score=0.4, reasoning="thin",
        )


def test_ask_user_without_a_gap_description_is_rejected():
    with pytest.raises(ValidationError, match="gap_description is required"):
        CompletenessVerdict(
            action="ask_user", answer_resolves_question=False,
            sufficiency_score=0.2, reasoning="unclear",
        )


def test_a_whitespace_only_gap_description_does_not_count():
    with pytest.raises(ValidationError):
        CompletenessVerdict(
            action="enrich", answer_resolves_question=False,
            sufficiency_score=0.4, gap_description="   ", reasoning="x",
        )


def test_enrich_with_a_gap_description_is_accepted():
    verdict = CompletenessVerdict(
        action="enrich",
        answer_resolves_question=False,
        sufficiency_score=0.4,
        gap_description="Enterprise plan pricing tiers",
        reasoning="price not stated",
    )
    assert verdict.gap_description == "Enterprise plan pricing tiers"


def test_sufficiency_score_is_bounded():
    with pytest.raises(ValidationError):
        CompletenessVerdict(
            action="accept", answer_resolves_question=True,
            sufficiency_score=1.4, reasoning="x",
        )


def test_an_unknown_action_is_rejected():
    with pytest.raises(ValidationError):
        CompletenessVerdict(
            action="give_up", answer_resolves_question=True,
            sufficiency_score=0.5, reasoning="x",
        )


def test_accept_is_rejected_when_the_answer_did_not_resolve_the_question():
    # The defect the end-to-end benchmark exposed: every unanswerable
    # question came back "accepted" at sufficiency 1.0. Nothing was
    # fabricating - the answer honestly said the context lacked the fact,
    # and the critic correctly judged that claim fully supported. The two
    # claims "well-supported" and "answered" were both riding on
    # `accept`, and only the first was ever checked, so the caller got a
    # refusal labelled as a finished answer.
    with pytest.raises(ValidationError, match="answer_resolves_question=True"):
        CompletenessVerdict(
            action="accept",
            answer_resolves_question=False,
            sufficiency_score=1.0,
            reasoning="the context genuinely does not state it",
        )


def test_a_supported_refusal_routes_to_enrichment():
    # Which is the whole point: a named missing fact is exactly what the
    # web search exists to go and find. Before this, the gap was never
    # routed anywhere and the enrichment loop sat idle.
    verdict = CompletenessVerdict(
        action="enrich",
        answer_resolves_question=False,
        sufficiency_score=1.0,
        gap_description="Aria Cloud Storage's current CTO",
        reasoning="context does not name a CTO",
    )
    assert verdict.action == "enrich"


def test_accept_with_a_resolved_question_is_still_fine():
    verdict = CompletenessVerdict(
        action="accept",
        answer_resolves_question=True,
        sufficiency_score=0.95,
        reasoning="context states the refund window directly",
    )
    assert verdict.action == "accept"


# --- history formatting ---

def test_no_history_renders_as_nothing():
    assert format_history([]) == ""


def test_history_lists_each_search_and_its_outcome():
    text = format_history([
        {"iteration": 1, "query": "Enterprise pricing tiers", "num_results": 3},
        {"iteration": 2, "query": "Aria revenue 2024", "num_results": 0},
    ])
    assert "Enterprise pricing tiers" in text
    assert "3 result(s)" in text
    assert "0 result(s)" in text


def test_history_reports_a_failed_search_as_a_failure():
    text = format_history([{"iteration": 1, "query": "x", "error": "rate limited"}])
    assert "search failed (rate limited)" in text


def test_history_tells_the_critic_not_to_repeat_itself():
    text = format_history([{"iteration": 1, "query": "x", "num_results": 1}])
    assert "Do not request any of these again" in text


def test_history_tolerates_a_partial_log_entry():
    assert "?" in format_history([{"query": "x"}])
