"""
These tests use DeterministicFakeEmbedding (a real, offline LangChain
testing utility - not a hack) so the entire request -> graph ->
vectorstore path can be exercised without downloading a real embedding
model or calling a real LLM.

Two of them are genuine integration checks rather than unit tests. The
"no API key" case proves the graph degrades to a clean "error" status
through the full HTTP stack instead of a 500 with a stack trace leaking
to the client. The stub-provider case walks a real question all the way
to a JSON body, which is the only way to catch a response field that the
graph stopped populating.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import src.api as api_module
from src.graph import build_graph
from src.retrieval import HybridRetriever


@pytest.fixture
def client():
    seed_docs = [
        Document(
            page_content="Aria offers a 14-day refund window on monthly plans.",
            metadata={"source": "seed.txt"},
        )
    ]
    vectorstore = Chroma.from_documents(
        documents=seed_docs,
        embedding=DeterministicFakeEmbedding(size=384),
        collection_name=f"test-{uuid.uuid4().hex}",
    )
    # min_relevance=0: fake embeddings can't produce a usable relevance
    # score, and the gate is covered in tests/test_retrieval.py.
    retriever = HybridRetriever(vectorstore, min_relevance=0.0)

    api_module._state["vectorstore"] = vectorstore
    api_module._state["retriever"] = retriever
    api_module._state["graph_app"] = build_graph(vectorstore, retriever=retriever)

    yield TestClient(api_module.app)

    # Reset so state doesn't leak between tests
    api_module._state["vectorstore"] = None
    api_module._state["retriever"] = None
    api_module._state["graph_app"] = None


@pytest.fixture
def no_web(monkeypatch):
    monkeypatch.setattr("src.websearch.web_search", lambda query, max_results=3: [])


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_stats_reports_the_seeded_chunk(client):
    body = client.get("/stats").json()
    assert body["chunks"] == 1
    assert body["auto_ingested_chunks"] == 0
    assert body["sources"] == 1


def test_upload_accepts_txt_file(client, tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("Enterprise plans include SSO via SAML.")

    with open(file_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("note.txt", f, "text/plain")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "note.txt"
    assert body["chunks_added"] >= 1


def test_uploaded_document_is_immediately_searchable(client, tmp_path):
    file_path = tmp_path / "sso.txt"
    file_path.write_text("Enterprise plans include SSO via SAML.")
    with open(file_path, "rb") as f:
        client.post("/upload", files={"file": ("sso.txt", f, "text/plain")})

    # Same live vectorstore object, no reopen from disk.
    assert client.get("/stats").json()["chunks"] == 2


def test_upload_rejects_unsupported_extension(client, tmp_path):
    file_path = tmp_path / "note.exe"
    file_path.write_bytes(b"not a real document")

    with open(file_path, "rb") as f:
        resp = client.post(
            "/upload", files={"file": ("note.exe", f, "application/octet-stream")}
        )

    assert resp.status_code == 400


def test_upload_rejects_a_file_with_no_extractable_text(client, tmp_path):
    file_path = tmp_path / "blank.txt"
    file_path.write_text("   \n  \n")
    with open(file_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("blank.txt", f, "text/plain")})
    assert resp.status_code == 400


def test_upload_rejects_a_file_over_the_size_limit(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 10)
    file_path = tmp_path / "big.txt"
    file_path.write_text("x" * 100)
    with open(file_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("big.txt", f, "text/plain")})
    assert resp.status_code == 413


def test_feedback_endpoint_records_entry(client, tmp_path, monkeypatch):
    log_path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(api_module, "FEEDBACK_LOG_PATH", str(log_path))

    resp = client.post(
        "/feedback",
        json={
            "question": "What is the refund policy?",
            "answer": "14 days for monthly plans.",
            "rating": "up",
            "comment": "accurate and concise",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "recorded"}

    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1

    import json
    entry = json.loads(lines[0])
    assert entry["question"] == "What is the refund policy?"
    assert entry["rating"] == "up"
    assert "timestamp" in entry


def test_feedback_rejects_invalid_rating(client):
    resp = client.post(
        "/feedback",
        json={"question": "q", "answer": "a", "rating": "sideways"},
    )
    assert resp.status_code == 422  # Pydantic validation error, not a 500


def test_query_rejects_an_empty_question(client):
    # Contract-level rejection, rather than embedding an empty string and
    # then answering a question nobody asked.
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_without_api_key_fails_gracefully(client, monkeypatch, no_web):
    # Force no key regardless of the local environment, to deterministically
    # exercise the failure path: the graph should terminate at handle_error
    # and the API should return a clean 200 with status="error", not an
    # unhandled exception.
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "groq")
    monkeypatch.setattr("src.config.config.GROQ_API_KEY", "")
    monkeypatch.setattr("src.config.config.GOOGLE_API_KEY", "")

    resp = client.post("/query", json={"question": "What is the refund policy?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"] is not None


def test_query_returns_a_full_result_with_the_stub_provider(client, monkeypatch, no_web):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "stub")

    resp = client.post(
        "/query", json={"question": "What is the refund window on monthly plans?"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["answer"]
    assert body["completeness_action"] == "accept"
    assert body["completeness_reasoning"]
    assert body["citations"]
    assert body["citations"][0]["kind"] == "document"
    assert body["retrieval_debug"]


def test_query_reports_the_enrichment_log(client, monkeypatch, no_web):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "stub")

    resp = client.post(
        "/query", json={"question": "What were the quarterly earnings in Tokyo?"}
    )

    body = resp.json()
    assert body["status"] == "needs_enrichment_capped"
    assert body["iteration_count"] == 2
    assert len(body["enrichment_log"]) == 2
