"""
Tests for the LangGraph wiring. Both the routing function and the graph
compilation step need no LLM call and no network access - route_after_
completeness() is pure state-in/string-out, and graph.compile() only
validates the node/edge structure without executing any node.
"""
import uuid

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.graph import build_graph, route_after_completeness, make_accept_node
from src.schemas import RAGResponse, CompletenessVerdict


def make_verdict(action, sufficiency=0.5, gap=None):
    return CompletenessVerdict(
        action=action,
        sufficiency_score=sufficiency,
        gap_description=gap,
        reasoning="test reasoning",
    )


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
    # iteration_count absent entirely (e.g. very first pass) should
    # behave the same as iteration_count=0, not crash or misroute
    state = {"completeness_verdict": make_verdict("enrich", gap="x"), "error": None}
    assert route_after_completeness(state) == "enrich"


def test_route_ask_user():
    state = {
        "completeness_verdict": make_verdict("ask_user", gap="which plan is meant"),
        "error": None,
    }
    assert route_after_completeness(state) == "ask_user"


def test_route_error_takes_priority_over_verdict():
    # even if a verdict somehow exists, an upstream error should win
    state = {"completeness_verdict": make_verdict("accept"), "error": "boom"}
    assert route_after_completeness(state) == "error"


def test_route_missing_verdict_is_error():
    state = {"completeness_verdict": None, "error": None}
    assert route_after_completeness(state) == "error"


def test_graph_compiles_without_a_real_vectorstore():
    # build_graph only needs *something* to close over for the retrieve
    # node - it's never called during compile(), so a dummy object is
    # fine here. This test is really just "does the graph structure
    # itself (nodes + edges + conditional routing) assemble correctly."
    dummy_vectorstore = object()
    app = build_graph(dummy_vectorstore)
    assert app is not None


def test_graph_compiles_with_persist_enrichment_disabled():
    dummy_vectorstore = object()
    app = build_graph(dummy_vectorstore, persist_enrichment=False)
    assert app is not None


def make_vectorstore():
    # Unique collection per call to keep tests isolated from each other.
    return Chroma(
        collection_name=f"test-{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=384),
    )


def make_answer():
    return RAGResponse(answer="14 days.", confidence=0.9, used_source_ids=[1])


def test_accept_node_persists_web_sources_when_enabled():
    vectorstore = make_vectorstore()
    node = make_accept_node(vectorstore, persist_enrichment=True)

    state = {
        "question": "What is the refund policy?",
        "retrieved_docs": [Document(page_content="x", metadata={"source": "doc.txt"})],
        "structured_answer": make_answer(),
        "web_sources": [{"title": "T", "href": "https://example.com", "body": "B"}],
    }
    result = node(state)

    assert result["final_status"] == "accepted"
    assert result["persisted_web_sources"] == 1


def test_accept_node_skips_persistence_when_disabled():
    vectorstore = make_vectorstore()
    node = make_accept_node(vectorstore, persist_enrichment=False)

    state = {
        "question": "What is the refund policy?",
        "retrieved_docs": [Document(page_content="x", metadata={"source": "doc.txt"})],
        "structured_answer": make_answer(),
        "web_sources": [{"title": "T", "href": "https://example.com", "body": "B"}],
    }
    result = node(state)

    assert result["persisted_web_sources"] == 0


def test_accept_node_handles_no_web_sources():
    vectorstore = make_vectorstore()
    node = make_accept_node(vectorstore, persist_enrichment=True)

    state = {
        "question": "What is the refund policy?",
        "retrieved_docs": [Document(page_content="x", metadata={"source": "doc.txt"})],
        "structured_answer": make_answer(),
        "web_sources": [],
    }
    result = node(state)

    assert result["persisted_web_sources"] == 0
