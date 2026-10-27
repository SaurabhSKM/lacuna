"""
format_web_results() and persist_web_sources() have no network call
inside them, so they're fully testable with canned results - this
sandbox can't reach duckduckgo.com to test the live web_search() call
itself, same limitation as fastembed and Groq earlier; that part is
verified on your machine instead.
"""
import uuid

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_chroma import Chroma

from src.websearch import format_web_results, persist_web_sources


def make_results():
    return [
        {"title": "Aria pricing FAQ", "href": "https://example.com/pricing", "body": "Enterprise pricing is custom."},
        {"title": "Aria security page", "href": "https://example.com/security", "body": "SOC 2 Type II certified."},
    ]


def test_format_continues_numbering_after_doc_sources():
    text = format_web_results(make_results(), start_index=5)
    assert "[Source 5]" in text
    assert "[Source 6]" in text
    assert "[Source 4]" not in text


def test_format_includes_title_url_and_body():
    text = format_web_results(make_results(), start_index=1)
    assert "Aria pricing FAQ" in text
    assert "https://example.com/pricing" in text
    assert "Enterprise pricing is custom." in text


def test_format_empty_results_returns_empty_string():
    assert format_web_results([], start_index=1) == ""


def test_format_handles_missing_fields_gracefully():
    # a malformed/partial search result shouldn't crash formatting
    text = format_web_results([{"title": "Only a title"}], start_index=1)
    assert "Only a title" in text
    assert "[Source 1]" in text


def make_empty_vectorstore():
    # Unique collection per call: Chroma's in-process client shares the
    # default collection across instances, which leaks state between tests.
    return Chroma(
        collection_name=f"test-{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=384),
    )


def test_persist_web_sources_adds_documents():
    vectorstore = make_empty_vectorstore()
    added = persist_web_sources(vectorstore, make_results(), origin_question="What is the pricing?")
    assert added == 2
    found = vectorstore.similarity_search("pricing", k=5)
    assert len(found) == 2


def test_persist_web_sources_tags_metadata_correctly():
    vectorstore = make_empty_vectorstore()
    persist_web_sources(vectorstore, make_results(), origin_question="What is the pricing?")
    found = vectorstore.similarity_search("pricing", k=5)
    for doc in found:
        assert doc.metadata["auto_ingested"] is True
        assert doc.metadata["source"].startswith("web:")
        assert doc.metadata["origin_question"] == "What is the pricing?"


def test_persist_web_sources_skips_empty_body():
    vectorstore = make_empty_vectorstore()
    results = [{"title": "No body here", "href": "https://example.com", "body": ""}]
    added = persist_web_sources(vectorstore, results, origin_question="q")
    assert added == 0


def test_persist_web_sources_empty_list_returns_zero():
    vectorstore = make_empty_vectorstore()
    assert persist_web_sources(vectorstore, [], origin_question="q") == 0
