"""
These tests use DeterministicFakeEmbedding (a real, offline LangChain
testing utility - not a hack) so the entire request -> graph -> vectorstore
path can be exercised without downloading a real embedding model or
calling a real LLM. The "no API key" test in particular is a genuine
integration check: it proves the graph degrades to a clean "error" status
through the full HTTP stack rather than the API raising an unhandled
500 with a stack trace leaking to the client.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document
from langchain_chroma import Chroma

import src.api as api_module
from src.graph import build_graph


@pytest.fixture
def client():
    fake_embeddings = DeterministicFakeEmbedding(size=384)
    seed_docs = [
        Document(
            page_content="Aria offers a 14-day refund window on monthly plans.",
            metadata={"source": "seed.txt"},
        )
    ]
    vectorstore = Chroma.from_documents(
        documents=seed_docs,
        embedding=fake_embeddings,
        collection_name=f"test-{uuid.uuid4().hex}",
    )

    api_module._state["vectorstore"] = vectorstore
    api_module._state["graph_app"] = build_graph(vectorstore)

    yield TestClient(api_module.app)

    # Reset so state doesn't leak between tests
    api_module._state["vectorstore"] = None
    api_module._state["graph_app"] = None


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_accepts_txt_file(client, tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("Enterprise plans include SSO via SAML.")

    with open(file_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("note.txt", f, "text/plain")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "note.txt"
    assert body["chunks_added"] >= 1


def test_upload_rejects_unsupported_extension(client, tmp_path):
    file_path = tmp_path / "note.exe"
    file_path.write_bytes(b"not a real document")

    with open(file_path, "rb") as f:
        resp = client.post(
            "/upload", files={"file": ("note.exe", f, "application/octet-stream")}
        )

    assert resp.status_code == 400


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


def test_query_without_api_key_fails_gracefully(client, monkeypatch):
    # Force no key regardless of the local environment, to deterministically
    # exercise the failure path: the graph should terminate at handle_error
    # and the API should return a clean 200 with status="error", not an
    # unhandled exception.
    monkeypatch.setattr("src.config.config.GROQ_API_KEY", "")
    monkeypatch.setattr("src.config.config.GOOGLE_API_KEY", "")

    resp = client.post("/query", json={"question": "What is the refund policy?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"] is not None
