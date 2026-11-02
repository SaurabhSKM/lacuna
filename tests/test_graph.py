"""
Tests for the LangGraph wiring.

The routing functions are pure state-in/string-out and graph.compile()
only validates structure, so both need no LLM. What's new here is that
the graph is also run end to end, using LLM_PROVIDER=stub (see
src/llm.py) and a monkeypatched web search. That matters: the cycle,
the iteration cap, the empty-retrieval short-circuit and citation
resolution on every terminal path are all *behaviour*, and testing the
router in isolation never showed whether the wiring around it agreed
with it. The cap in particular was only ever verified as a routing
decision, never as "the graph actually stops".

The stub is not a model and its answers mean nothing. It is a
deterministic driver for the state machine.
"""
import uuid

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src import websearch
from src.graph import (
    build_graph,
    initial_state,
    make_accept_node,
    route_after_completeness,
    route_after_retrieval,
    run_question,
)
from src.retrieval import HybridRetriever
from src.schemas import CompletenessVerdict, RAGResponse


def make_verdict(action, sufficiency=0.5, gap=None, resolves=None):
    return CompletenessVerdict(
        action=action,
        # "accept" is only valid when the answer resolved the question;
        # anything else implies it didn't.
        answer_resolves_question=(action == "accept") if resolves is None else resolves,
        sufficiency_score=sufficiency,
        gap_description=gap or (None if action == "accept" else "a specific missing figure"),
        reasoning="test reasoning",
    )


# --- routing after the critic ---

def test_route_accept():
    state = {"completeness_verdict": make_verdict("accept"), "error": None}
    assert route_after_completeness(state) == "accept"


def test_route_enrich():
    state = {
        "completeness_verdict": make_verdict("enrich", gap="pricing tiers missing"),
        "error": None,
        "iteration_count": 0,
    }
    assert route_after_completeness(state) == "enrich"


def test_route_enrich_under_cap_continues_looping():
    from src.config import config

    state = {
        "completeness_verdict": make_verdict("enrich", gap="x"),
        "error": None,
        "iteration_count": config.MAX_ENRICHMENT_ITERATIONS - 1,
    }
    assert route_after_completeness(state) == "enrich"


def test_route_enrich_at_cap_stops_looping():
    from src.config import config

    state = {
        "completeness_verdict": make_verdict("enrich", gap="x"),
        "error": None,
        "iteration_count": config.MAX_ENRICHMENT_ITERATIONS,
    }
    assert route_after_completeness(state) == "max_iterations"


def test_route_enrich_missing_iteration_count_defaults_to_zero():
    state = {"completeness_verdict": make_verdict("enrich", gap="x"), "error": None}
    assert route_after_completeness(state) == "enrich"


def test_route_ask_user():
    state = {
        "completeness_verdict": make_verdict("ask_user", gap="which plan is meant"),
        "error": None,
    }
    assert route_after_completeness(state) == "ask_user"


def test_route_error_takes_priority_over_verdict():
    state = {"completeness_verdict": make_verdict("accept"), "error": "boom"}
    assert route_after_completeness(state) == "error"


def test_route_missing_verdict_is_error():
    state = {"completeness_verdict": None, "error": None}
    assert route_after_completeness(state) == "error"


# --- routing after retrieval ---

def test_route_after_retrieval_goes_to_answer_when_context_exists():
    state = {"retrieved_docs": [Document(page_content="x")]}
    assert route_after_retrieval(state) == "answer"


def test_route_after_retrieval_skips_to_enrichment_when_empty():
    # Answering from a knowingly-empty context and then grading that
    # answer is two LLM calls spent to learn something already certain.
    assert route_after_retrieval({"retrieved_docs": []}) == "enrich"


# --- compilation ---

def test_graph_compiles_without_a_real_vectorstore():
    app = build_graph(object(), retriever=object())
    assert app is not None


def test_graph_compiles_with_persist_enrichment_disabled():
    app = build_graph(object(), persist_enrichment=False, retriever=object())
    assert app is not None


# --- the accept node ---

def make_store(docs=None):
    store = Chroma(
        collection_name=f"test-{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=384),
    )
    if docs:
        store.add_documents(docs)
    return store


WEB_RESULT = {"title": "T", "href": "https://example.com", "body": "B"}


def accept_state(used_source_ids, web_sources=None):
    docs = [Document(page_content="x", metadata={"source": "doc.txt"})]
    from src.rag import build_source_refs

    web_sources = [] if web_sources is None else web_sources
    return {
        "question": "What is the refund policy?",
        "retrieved_docs": docs,
        "source_refs": build_source_refs(docs, web_sources),
        "structured_answer": RAGResponse(
            answer="14 days.", confidence=0.9, used_source_ids=used_source_ids
        ),
        "web_sources": web_sources,
    }


def test_accept_node_persists_a_cited_web_source():
    node = make_accept_node(make_store(), persist_enrichment=True)
    # id 2 is the web result (id 1 is the document chunk)
    result = node(accept_state([1, 2], [WEB_RESULT]))
    assert result["final_status"] == "accepted"
    assert result["persisted_web_sources"] == 1


def test_accept_node_does_not_persist_an_uncited_web_source():
    # "The search returned it" is not "the answer used it". Persisting
    # uncited results puts unvetted web text into the store, where it
    # then competes with the user's own documents forever.
    node = make_accept_node(make_store(), persist_enrichment=True)
    result = node(accept_state([1], [WEB_RESULT]))
    assert result["persisted_web_sources"] == 0


def test_accept_node_does_not_persist_a_web_only_answer():
    # The incident this prevents: "What is the price?" was enriched, the
    # search returned Wikipedia and a dictionary on the *concept* of
    # price, the critic accepted an answer built only from those, and all
    # three were written into the knowledge base - after which the same
    # question retrieved them, looked supported, and was accepted
    # instantly. The contamination made the bad answer more likely each
    # time rather than less.
    node = make_accept_node(make_store(), persist_enrichment=True)
    # id 2 is the web result; id 1 (the document) is deliberately uncited.
    result = node(accept_state([2], [WEB_RESULT]))
    assert result["final_status"] == "accepted"
    assert [c.kind for c in result["citations"]] == ["web"]
    assert result["persisted_web_sources"] == 0


def test_web_only_answer_still_persists_when_the_anchor_is_disabled(monkeypatch):
    monkeypatch.setattr("src.config.config.PERSIST_REQUIRES_DOC_ANCHOR", False)
    node = make_accept_node(make_store(), persist_enrichment=True)
    assert node(accept_state([2], [WEB_RESULT]))["persisted_web_sources"] == 1


def test_accept_node_skips_persistence_when_disabled():
    node = make_accept_node(make_store(), persist_enrichment=False)
    assert node(accept_state([1, 2], [WEB_RESULT]))["persisted_web_sources"] == 0


def test_accept_node_handles_no_web_sources():
    node = make_accept_node(make_store(), persist_enrichment=True)
    assert node(accept_state([1]))["persisted_web_sources"] == 0


def test_accept_node_resolves_both_kinds_of_citation():
    node = make_accept_node(make_store(), persist_enrichment=False)
    result = node(accept_state([1, 2], [WEB_RESULT]))
    assert [c.kind for c in result["citations"]] == ["document", "web"]
    assert result["sources"] == ["doc.txt", "https://example.com"]


# --- end to end, with the stub provider ---

@pytest.fixture
def stub_provider(monkeypatch):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "stub")


@pytest.fixture
def no_web(monkeypatch):
    """A search that finds nothing: the enrichment path still has to run."""
    monkeypatch.setattr(websearch, "web_search", lambda query, max_results=3: [])


def seeded_store():
    return make_store([
        Document(
            page_content=(
                "Aria offers a 14-day refund window on monthly plans, provided "
                "less than 5 GB of storage has been used."
            ),
            metadata={"source": "billing.txt"},
        )
    ])


def build_app(store, **kwargs):
    # min_relevance=0 because fake embeddings can't produce a usable
    # relevance score; the gate's own behaviour is tested in
    # tests/test_retrieval.py.
    return build_graph(
        store, retriever=HybridRetriever(store, min_relevance=0.0), **kwargs
    )


def test_end_to_end_accepts_a_covered_question(stub_provider, no_web):
    store = seeded_store()
    result = run_question(build_app(store), "What is the refund window on monthly plans?")
    assert result["final_status"] == "accepted"
    assert result["iteration_count"] == 0
    assert result["citations"]


def test_end_to_end_routes_an_ambiguous_question_to_clarification(stub_provider, no_web):
    store = seeded_store()
    result = run_question(build_app(store), "What is the price?")
    assert result["final_status"] == "needs_clarification"
    assert result["completeness_verdict"].gap_description


def test_end_to_end_enriches_then_stops_at_the_cap(stub_provider, no_web):
    from src.config import config

    store = seeded_store()
    result = run_question(
        build_app(store), "What were the quarterly earnings and headcount in Tokyo?"
    )
    # The critic keeps asking to enrich; the cap is what ends the run.
    assert result["final_status"] == "needs_enrichment_capped"
    assert result["iteration_count"] == config.MAX_ENRICHMENT_ITERATIONS
    assert len(result["enrichment_log"]) == config.MAX_ENRICHMENT_ITERATIONS


def test_capped_run_still_reports_an_answer_and_citations(stub_provider, no_web):
    store = seeded_store()
    result = run_question(build_app(store), "What were the quarterly earnings in Tokyo?")
    assert result["structured_answer"] is not None
    # A flagged, best-effort answer needs its provenance more than a
    # clean one, not less - every terminal node resolves citations.
    assert "citations" in result


def test_empty_retrieval_skips_the_first_answer_round(stub_provider, no_web):
    store = make_store()  # nothing in it at all
    result = run_question(build_app(store), "What is the refund policy?")
    assert result["iteration_count"] >= 1
    # Enrichment ran before any answer was drafted.
    assert result["enrichment_log"][0]["iteration"] == 1


def test_enrichment_deduplicates_repeated_web_results(stub_provider, monkeypatch):
    # Both rounds return the same page. It must appear once, or the model
    # sees one source twice and reads it as two sources agreeing.
    monkeypatch.setattr(
        websearch, "web_search", lambda query, max_results=3: [dict(WEB_RESULT)]
    )
    store = seeded_store()
    result = run_question(build_app(store), "What were the quarterly earnings in Tokyo?")
    assert result["iteration_count"] == 2
    assert len(result["web_sources"]) == 1


def test_enrichment_broadens_a_repeated_query(stub_provider, no_web):
    # The stub critic returns the same gap_description every round, which
    # is exactly the behaviour the guard exists for.
    store = seeded_store()
    result = run_question(build_app(store), "What were the quarterly earnings in Tokyo?")
    assert result["enrichment_log"][1].get("broadened") is True
    assert len({e["query"] for e in result["enrichment_log"]}) == 2


def test_search_failure_still_advances_the_iteration_count(stub_provider, monkeypatch):
    def boom(query, max_results=3):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(websearch, "web_search", boom)
    store = seeded_store()
    result = run_question(build_app(store), "What were the quarterly earnings in Tokyo?")
    # A broken search must not become an infinite loop.
    assert result["final_status"] == "needs_enrichment_capped"
    assert all("error" in entry for entry in result["enrichment_log"])


def test_missing_api_key_ends_at_the_error_node(monkeypatch, no_web):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "groq")
    monkeypatch.setattr("src.config.config.GROQ_API_KEY", "")
    store = seeded_store()
    result = run_question(build_app(store), "What is the refund policy?")
    assert result["final_status"] == "error"
    assert result["error"]


def test_initial_state_covers_every_graph_state_field():
    # Three entry points used to hand-write this dict; a field added to
    # GraphState and missed in one of them failed only at runtime, only
    # on that one entry point.
    from src.graph import GraphState

    assert set(initial_state("q")) == set(GraphState.__annotations__)
