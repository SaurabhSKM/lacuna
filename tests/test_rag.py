"""
These tests need no API key and no network access - that's by design.
The source-list logic and the response schemas are pure/local, kept
deliberately separate from the LLM call so they can be verified in
isolation. This is also what CI (GitHub Actions) runs on every push.

Run with:
    pytest tests/
"""
import pytest
from pydantic import ValidationError
from langchain_core.documents import Document

from src.rag import (
    build_source_refs,
    format_context,
    resolve_citations,
    resolve_sources,
    NO_CONTEXT_MARKER,
)
from src.schemas import RAGResponse


def make_docs():
    return [
        Document(page_content="chunk A", metadata={"source": "docs/aria_overview.txt"}),
        Document(page_content="chunk B", metadata={"source": "docs/aria_billing_policy.txt"}),
        Document(page_content="chunk C", metadata={"source": "docs/aria_security.txt"}),
    ]


def make_web():
    return [
        {"title": "SOC 2 explained", "href": "https://example.com/soc2", "body": "A SOC 2 report is..."},
        {"title": "FIDO2 overview", "href": "https://example.com/fido2", "body": "FIDO2 is..."},
    ]


# --- numbering ---

def test_documents_are_numbered_from_one():
    refs = build_source_refs(make_docs())
    assert [r.source_id for r in refs] == [1, 2, 3]
    assert all(r.kind == "document" for r in refs)


def test_web_results_continue_the_document_numbering():
    refs = build_source_refs(make_docs(), make_web())
    assert [r.source_id for r in refs] == [1, 2, 3, 4, 5]
    assert [r.kind for r in refs] == ["document"] * 3 + ["web"] * 2


def test_document_numbering_is_stable_as_web_results_accumulate():
    # The loop re-renders context every enrichment round. A document id
    # must keep pointing at the same chunk, or citations made in an
    # earlier round silently become wrong.
    first = build_source_refs(make_docs(), make_web()[:1])
    second = build_source_refs(make_docs(), make_web())
    assert [r.label for r in first[:3]] == [r.label for r in second[:3]]
    assert first[3].label == second[3].label


def test_document_label_strips_directories():
    refs = build_source_refs(make_docs())
    assert refs[0].label == "aria_overview.txt"


def test_document_label_includes_pdf_page():
    doc = Document(page_content="x", metadata={"source": "a/manual.pdf", "page": 7})
    assert build_source_refs([doc])[0].label == "manual.pdf p.7"


# --- context formatting ---

def test_format_context_labels_provenance():
    text = format_context(build_source_refs(make_docs(), make_web()))
    assert "[Source 1] (document: aria_overview.txt)" in text
    assert "[Source 4] (web: SOC 2 explained - https://example.com/soc2)" in text


def test_format_context_empty_returns_explicit_marker():
    # A blank context reads to the model as "no instruction about
    # context" and invites answering from training data. An explicit
    # sentence is a fact it can report instead.
    assert format_context([]) == NO_CONTEXT_MARKER
    assert format_context([]).strip() != ""


# --- citation resolution ---

def test_resolve_citations_maps_ids_to_filenames():
    refs = build_source_refs(make_docs())
    cites = resolve_citations(refs, [1, 3])
    assert [c.label for c in cites] == ["aria_overview.txt", "aria_security.txt"]
    assert all(c.kind == "document" for c in cites)


def test_resolve_citations_resolves_web_sources_too():
    # This is the regression that motivated the unified source list: a
    # web id used to fall outside the document range and be dropped, so
    # an answer could cite the web and report no source for it.
    refs = build_source_refs(make_docs(), make_web())
    cites = resolve_citations(refs, [1, 4])
    assert [c.kind for c in cites] == ["document", "web"]
    assert cites[1].url == "https://example.com/soc2"


def test_resolve_citations_deduplicates_by_provenance():
    docs = make_docs() + [
        Document(page_content="chunk A2", metadata={"source": "docs/aria_overview.txt"})
    ]
    cites = resolve_citations(build_source_refs(docs), [1, 4])
    assert [c.label for c in cites] == ["aria_overview.txt"]


def test_resolve_citations_ignores_out_of_range_ids():
    refs = build_source_refs(make_docs())
    cites = resolve_citations(refs, [1, 99, 0, -1])
    assert [c.label for c in cites] == ["aria_overview.txt"]


def test_resolve_citations_empty_input():
    assert resolve_citations(build_source_refs(make_docs()), []) == []


def test_resolve_sources_prefers_url_for_web_results():
    refs = build_source_refs(make_docs(), make_web())
    assert resolve_sources(refs, [2, 5]) == [
        "aria_billing_policy.txt",
        "https://example.com/fido2",
    ]


# --- schema ---

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
        RAGResponse(answer="...", confidence=1.5, missing_info=[], used_source_ids=[])


def test_ragresponse_defaults_are_empty_lists():
    r = RAGResponse(answer="...", confidence=0.5)
    assert r.missing_info == []
    assert r.used_source_ids == []
