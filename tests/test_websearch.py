"""
The decision logic in websearch.py - dedup, persistence filtering - has
no network call and no vector store inside it, so it's fully testable
with canned results. The live search call itself is a thin wrapper around
someone else's client and is the one thing here not covered offline.
"""
import uuid

from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.websearch import (
    dedupe_results,
    format_web_results,
    persist_web_sources,
    select_persistable,
)


def make_results():
    return [
        {"title": "Aria pricing FAQ", "href": "https://example.com/pricing",
         "body": "Enterprise pricing is custom."},
        {"title": "Aria security page", "href": "https://example.com/security",
         "body": "SOC 2 Type II certified."},
    ]


# --- formatting ---

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
    text = format_web_results([{"title": "Only a title"}], start_index=1)
    assert "Only a title" in text
    assert "[Source 1]" in text


# --- dedup across enrichment rounds ---

def test_dedupe_drops_results_already_held():
    existing = make_results()[:1]
    new = dedupe_results(existing, make_results())
    assert [r["href"] for r in new] == ["https://example.com/security"]


def test_dedupe_drops_duplicates_within_one_batch():
    duplicated = make_results() + make_results()
    assert len(dedupe_results([], duplicated)) == 2


def test_dedupe_keeps_everything_when_nothing_overlaps():
    assert len(dedupe_results([], make_results())) == 2


def test_dedupe_drops_a_result_with_neither_url_nor_body():
    assert dedupe_results([], [{"title": "empty"}]) == []


def test_dedupe_keeps_a_urlless_result_that_has_text():
    assert len(dedupe_results([], [{"title": "t", "body": "real text"}])) == 1


# --- which results are worth keeping ---

def test_select_persistable_keeps_only_cited_results():
    cited = {"https://example.com/security"}
    kept = select_persistable(make_results(), cited, only_cited=True)
    assert [r["href"] for r in kept] == ["https://example.com/security"]


def test_select_persistable_can_keep_everything():
    assert len(select_persistable(make_results(), set(), only_cited=False)) == 2


def test_select_persistable_always_drops_empty_bodies():
    results = [{"title": "t", "href": "https://example.com/x", "body": ""}]
    assert select_persistable(results, {"https://example.com/x"}, only_cited=True) == []
    assert select_persistable(results, set(), only_cited=False) == []


# --- persistence ---

def make_empty_vectorstore():
    # Unique collection per call: Chroma's in-process client shares the
    # default collection across instances, which leaks state between tests.
    return Chroma(
        collection_name=f"test-{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=384),
    )


def all_cited():
    return {r["href"] for r in make_results()}


def test_persist_web_sources_adds_documents():
    store = make_empty_vectorstore()
    added = persist_web_sources(
        store, make_results(), origin_question="What is the pricing?", cited_urls=all_cited()
    )
    assert added == 2
    assert len(store.similarity_search("pricing", k=5)) == 2


def test_persist_web_sources_tags_metadata_correctly():
    store = make_empty_vectorstore()
    persist_web_sources(
        store, make_results(), origin_question="What is the pricing?", cited_urls=all_cited()
    )
    for doc in store.similarity_search("pricing", k=5):
        assert doc.metadata["auto_ingested"] is True
        assert doc.metadata["source"].startswith("web:")
        assert doc.metadata["origin_question"] == "What is the pricing?"


def test_persist_web_sources_skips_uncited_results():
    store = make_empty_vectorstore()
    added = persist_web_sources(
        store, make_results(), origin_question="q",
        cited_urls={"https://example.com/pricing"},
    )
    assert added == 1


def test_persist_web_sources_is_idempotent_for_the_same_url():
    # Accepting several answers on the same topic used to append a fresh
    # near-duplicate chunk every time, so the store grew without
    # learning anything. A URL already stored is skipped.
    store = make_empty_vectorstore()
    first = persist_web_sources(store, make_results(), "q", cited_urls=all_cited())
    second = persist_web_sources(store, make_results(), "q again", cited_urls=all_cited())
    assert (first, second) == (2, 0)
    assert len(store.get(include=[])["ids"]) == 2


def test_persist_web_sources_skips_empty_body():
    store = make_empty_vectorstore()
    results = [{"title": "No body here", "href": "https://example.com", "body": ""}]
    assert persist_web_sources(store, results, "q", cited_urls={"https://example.com"}) == 0


def test_persist_web_sources_empty_list_returns_zero():
    assert persist_web_sources(make_empty_vectorstore(), [], "q", cited_urls=set()) == 0


def test_persist_web_sources_without_citation_info_keeps_everything():
    # cited_urls=None means "no citation information available" - fall
    # back to the old behaviour rather than silently persisting nothing.
    store = make_empty_vectorstore()
    assert persist_web_sources(store, make_results(), "q", cited_urls=None) == 2
