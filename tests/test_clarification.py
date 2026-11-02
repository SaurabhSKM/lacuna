"""
The clarification loop: `ask_user` as a resumable outcome rather than a
dead end.

The design claim being tested is that resuming re-runs the pipeline with
the clarification attached, instead of restoring a checkpoint. The
justification is measurable and is asserted here on a small corpus: an
ambiguous question retrieves nothing, and the same question plus the
user's own disambiguating words retrieves the right document. There is
no retrieval state worth preserving across the turn, so preserving it
would be ceremony - and would reinstate an empty result.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import src.api as api_module
from src import rag, websearch
from src.api import PendingClarifications
from src.config import config
from src.graph import build_graph, initial_state, route_after_completeness, run_question
from src.retrieval import HybridRetriever
from src.schemas import CompletenessVerdict


# --- composing the clarified question ---

def turns(*pairs):
    return [{"question_asked": q, "user_answer": a} for q, a in pairs]


def test_retrieval_query_is_the_question_when_unclarified():
    assert rag.retrieval_query("What is the price?", []) == "What is the price?"


def test_retrieval_query_appends_the_users_answer():
    # The user's reply carries the disambiguating terms; the question we
    # asked them is our own wording and would only add lexical noise.
    query = rag.retrieval_query(
        "What is the price?", turns(("Which plan do you mean?", "the Team plan"))
    )
    assert query == "What is the price? the Team plan"
    assert "Which plan" not in query


def test_retrieval_query_accumulates_multiple_turns():
    query = rag.retrieval_query("What is the price?", turns(("a?", "Team"), ("b?", "monthly")))
    assert query == "What is the price? Team monthly"


def test_clarifications_render_into_the_prompt():
    block = rag.format_clarifications(turns(("Which plan?", "Team")))
    assert "Which plan?" in block
    assert "Team" in block
    assert "Do not ask for the same clarification again" in block


def test_no_clarifications_renders_as_nothing():
    assert rag.format_clarifications([]) == ""


# --- the cap ---

def ask_user_state(n_clarifications):
    return {
        "completeness_verdict": CompletenessVerdict(
            action="ask_user",
            answer_resolves_question=False,
            sufficiency_score=0.2,
            gap_description="which plan is meant",
            reasoning="ambiguous",
        ),
        "error": None,
        "clarifications": turns(*[(f"q{i}", f"a{i}") for i in range(n_clarifications)]),
    }


def test_ask_user_is_allowed_under_the_cap():
    assert route_after_completeness(ask_user_state(0)) == "ask_user"
    assert route_after_completeness(ask_user_state(config.MAX_CLARIFICATIONS - 1)) == "ask_user"


def test_ask_user_at_the_cap_stops_asking():
    # A critic that can request a clarification can request one forever.
    # Past the cap, answer as best we can and flag it rather than
    # bouncing the question back again.
    assert route_after_completeness(ask_user_state(config.MAX_CLARIFICATIONS)) == "max_iterations"


def test_initial_state_carries_clarifications():
    state = initial_state("q", turns(("a?", "b")))
    assert state["clarifications"] == turns(("a?", "b"))
    assert state["clarification_request"] is None


def test_initial_state_defaults_to_no_clarifications():
    assert initial_state("q")["clarifications"] == []


# --- end to end through the graph ---

def make_store(docs):
    store = Chroma(
        collection_name=f"test-{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=384),
    )
    if docs:
        store.add_documents(docs)
    return store


def pricing_store():
    return make_store([
        Document(
            page_content="Annual billing gives a 15% discount on the Team plan.",
            metadata={"source": "billing.txt"},
        )
    ])


def build_app(store):
    return build_graph(store, retriever=HybridRetriever(store, min_relevance=0.0))


@pytest.fixture
def stub_provider(monkeypatch):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "stub")


@pytest.fixture
def no_web(monkeypatch):
    monkeypatch.setattr(websearch, "web_search", lambda query, max_results=3: [])


def test_ambiguous_question_ends_resumable(stub_provider, no_web):
    result = run_question(build_app(pricing_store()), "What is the price?")
    assert result["final_status"] == "needs_clarification"
    # The request is the whole input a caller needs to resume.
    assert result["clarification_request"]


def test_resuming_changes_the_outcome(stub_provider, no_web):
    # The point of the feature: the same question, plus the user's
    # answer, no longer dead-ends.
    app = build_app(pricing_store())
    first = run_question(app, "What is the price?")
    assert first["final_status"] == "needs_clarification"

    resumed = run_question(
        app,
        "What is the price?",
        turns((first["clarification_request"], "the Team plan annual discount")),
    )
    assert resumed["final_status"] != "needs_clarification"
    assert resumed["clarifications"]


def test_clarification_reaches_the_retriever(stub_provider, no_web):
    # Regression guard for the actual mechanism: if retrieval ignored the
    # clarification, resuming would retrieve exactly what the ambiguous
    # turn did and nothing would improve.
    captured = {}
    store = pricing_store()
    retriever = HybridRetriever(store, min_relevance=0.0)
    original = retriever.retrieve

    def spy(question, k=None):
        captured["query"] = question
        return original(question, k=k)

    retriever.retrieve = spy
    app = build_graph(store, retriever=retriever)
    run_question(app, "What is the price?", turns(("Which plan?", "Team plan")))
    assert "Team plan" in captured["query"]


def test_clarification_cap_terminates_the_conversation(stub_provider, no_web):
    app = build_app(pricing_store())
    many = turns(*[(f"q{i}?", "still vague") for i in range(config.MAX_CLARIFICATIONS)])
    result = run_question(app, "What is the price?", many)
    assert result["final_status"] != "needs_clarification"


# --- the pending-request store ---

def test_pending_store_round_trips_a_request():
    store = PendingClarifications()
    token = store.open("What is the price?", [], "which plan?")
    entry = store.take(token)
    assert entry["question"] == "What is the price?"
    assert entry["request"] == "which plan?"


def test_pending_token_is_single_use():
    # Answering one request twice would otherwise fork the conversation
    # off a single id, with the second answer silently replacing the first.
    store = PendingClarifications()
    token = store.open("q", [], "r")
    assert store.take(token) is not None
    assert store.take(token) is None


def test_pending_unknown_token_returns_none():
    assert PendingClarifications().take("never-issued") is None


def test_pending_entries_expire():
    store = PendingClarifications(ttl_seconds=0)
    token = store.open("q", [], "r")
    assert store.take(token) is None


def test_pending_store_is_bounded():
    # An abandoned conversation must not accumulate into a slow leak.
    store = PendingClarifications(max_entries=3)
    tokens = [store.open(f"q{i}", [], "r") for i in range(10)]
    assert len(store) <= 3
    assert store.take(tokens[0]) is None   # oldest evicted
    assert store.take(tokens[-1]) is not None


# --- over HTTP ---

@pytest.fixture
def client():
    store = pricing_store()
    retriever = HybridRetriever(store, min_relevance=0.0)
    api_module._state["vectorstore"] = store
    api_module._state["retriever"] = retriever
    api_module._state["graph_app"] = build_graph(store, retriever=retriever)
    yield TestClient(api_module.app)
    api_module._state["vectorstore"] = None
    api_module._state["retriever"] = None
    api_module._state["graph_app"] = None
    api_module._pending.clear()


def test_query_returns_a_clarification_token(client, stub_provider, no_web):
    body = client.post("/query", json={"question": "What is the price?"}).json()
    assert body["status"] == "needs_clarification"
    assert body["clarification_request"]
    assert body["clarification_id"]


def test_clarify_endpoint_resumes_the_conversation(client, stub_provider, no_web):
    opened = client.post("/query", json={"question": "What is the price?"}).json()
    resumed = client.post(
        "/clarify",
        json={"clarification_id": opened["clarification_id"], "answer": "the Team plan"},
    )
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] != "needs_clarification"
    assert body["clarifications"][0]["user_answer"] == "the Team plan"


def test_clarify_rejects_an_unknown_token(client, stub_provider, no_web):
    resp = client.post("/clarify", json={"clarification_id": "bogus", "answer": "x"})
    assert resp.status_code == 404
    assert "expired" in resp.json()["detail"]


def test_clarify_rejects_a_reused_token(client, stub_provider, no_web):
    opened = client.post("/query", json={"question": "What is the price?"}).json()
    payload = {"clarification_id": opened["clarification_id"], "answer": "the Team plan"}
    assert client.post("/clarify", json=payload).status_code == 200
    assert client.post("/clarify", json=payload).status_code == 404


def test_clarify_rejects_a_blank_answer(client, stub_provider, no_web):
    opened = client.post("/query", json={"question": "What is the price?"}).json()
    resp = client.post(
        "/clarify", json={"clarification_id": opened["clarification_id"], "answer": ""}
    )
    assert resp.status_code == 422


def test_query_accepts_inline_clarifications_for_stateless_clients(client, stub_provider, no_web):
    # Lets a caller behind a load balancer carry the exchange itself
    # rather than depend on a process-local token.
    body = client.post(
        "/query",
        json={
            "question": "What is the price?",
            "clarifications": [
                {"question_asked": "Which plan?", "user_answer": "the Team plan"}
            ],
        },
    ).json()
    assert body["status"] != "needs_clarification"
