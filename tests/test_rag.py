"""
These tests need no API key and no network access - that's by design.
resolve_sources() and RAGResponse are pure/local logic, deliberately kept
separate from the LLM call so they can be verified in isolation. This is
also what a CI pipeline (GitHub Actions) would run on every push.

Run with:
    pytest tests/
"""
import pytest
from pydantic import ValidationError
from langchain_core.documents import Document

from src.rag import resolve_sources
from src.schemas import RAGResponse


def make_docs():
    return [
        Document(page_content="chunk A", metadata={"source": "aria_overview.txt"}),
        Document(page_content="chunk B", metadata={"source": "aria_billing_policy.txt"}),
        Document(page_content="chunk C", metadata={"source": "aria_security.txt"}),
    ]


def test_resolve_sources_maps_ids_to_filenames():
    docs = make_docs()
    result = resolve_sources(docs, used_source_ids=[1, 3])
    assert result == ["aria_overview.txt", "aria_security.txt"]


def test_resolve_sources_deduplicates():
    docs = make_docs()
    # if two chunks came from the same file, LLM might cite both -
    # the caller should only see the filename once
    docs_with_dupe = docs + [
        Document(page_content="chunk A2", metadata={"source": "aria_overview.txt"})
    ]
    result = resolve_sources(docs_with_dupe, used_source_ids=[1, 4])
    assert result == ["aria_overview.txt"]


def test_resolve_sources_ignores_out_of_range_ids():
    docs = make_docs()
    # a hallucinated or off-by-one source ID shouldn't crash the pipeline
    result = resolve_sources(docs, used_source_ids=[1, 99, 0, -1])
    assert result == ["aria_overview.txt"]


def test_resolve_sources_empty_input():
    docs = make_docs()
    assert resolve_sources(docs, used_source_ids=[]) == []


def test_ragresponse_accepts_valid_data():
    r = RAGResponse(
        answer="The refund window is 14 days.",
        confidence=0.9,
        missing_info=[],
        used_source_ids=[2],
    )
    assert r.confidence == 0.9


def test_ragresponse_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        RAGResponse(
            answer="...",
            confidence=1.5,  # invalid: must be <= 1.0
            missing_info=[],
            used_source_ids=[],
        )


def test_ragresponse_defaults_are_empty_lists():
    r = RAGResponse(answer="...", confidence=0.5)
    assert r.missing_info == []
    assert r.used_source_ids == []
