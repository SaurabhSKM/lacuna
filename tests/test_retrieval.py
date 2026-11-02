"""
Retrieval tests. The fusion and scoring logic is pure, so it's tested
directly; the retriever itself is tested against a real Chroma
collection built with DeterministicFakeEmbedding, which is offline.

Note what fake embeddings can and cannot show. They give a real vector
store with real nearest-neighbour behaviour, so wiring, gating and
fusion are genuinely exercised. They carry no semantics, so these tests
can't show that hybrid retrieval finds better documents - that claim is
measured separately against real embeddings and labelled questions by
eval/retrieval_eval.py, which is where the numbers in the README come
from.
"""
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.retrieval import BM25Index, HybridRetriever, reciprocal_rank_fusion, tokenize
from eval.retrieval_eval import score_retrieval


# --- pure helpers ---

def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("X-RateLimit-Reset: 429!") == ["x", "ratelimit", "reset", "429"]


def test_rrf_rewards_agreement_between_rankings():
    # "b" is second in both rankings; "a" is first in one and absent from
    # the other. Two mid rankings should beat one top ranking plus a miss.
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["d", "b", "e"]], rrf_k=1)
    assert scores["b"] > scores["a"]


def test_rrf_is_order_sensitive():
    scores = reciprocal_rank_fusion([["a", "b"]], rrf_k=1)
    assert scores["a"] > scores["b"]


def test_rrf_empty_rankings_give_empty_scores():
    assert reciprocal_rank_fusion([[], []]) == {}


# --- metric definitions ---

def test_score_retrieval_counts_a_hit_at_any_rank():
    report = score_retrieval([{"gold.txt"}], [["other.txt", "gold.txt"]], [])
    assert report["hit_rate"] == 1.0
    assert report["mrr"] == 0.5
    assert report["precision"] == 0.5


def test_score_retrieval_counts_a_miss():
    report = score_retrieval([{"gold.txt"}], [["other.txt"]], [])
    assert report["hit_rate"] == 0.0
    assert report["mrr"] == 0.0


def test_score_retrieval_distractor_rejection_requires_empty_results():
    assert score_retrieval([], [], [0, 0])["distractor_rejection"] == 1.0
    assert score_retrieval([], [], [0, 3])["distractor_rejection"] == 0.5


def test_score_retrieval_handles_empty_benchmark():
    report = score_retrieval([], [], [])
    assert report["hit_rate"] == 0.0
    assert report["distractor_rejection"] == 0.0


# --- BM25 over a real collection ---

def corpus():
    return [
        Document(page_content="Hardware security key support uses FIDO2 on the Enterprise plan.",
                 metadata={"source": "security.txt"}),
        Document(page_content="Refunds are available within 14 days on monthly plans.",
                 metadata={"source": "billing.txt"}),
        Document(page_content="The uptime commitment is 99.9 percent for Team plans.",
                 metadata={"source": "sla.txt"}),
    ]


def make_store(docs=None):
    # Chroma.from_documents rejects an empty list, so an empty store has
    # to go through the constructor.
    store = Chroma(
        embedding_function=DeterministicFakeEmbedding(size=384),
        collection_name=f"test-{uuid.uuid4().hex}",
    )
    docs = corpus() if docs is None else docs
    if docs:
        store.add_documents(docs)
    return store


def test_bm25_finds_an_exact_rare_term():
    store = make_store()
    results = BM25Index().search(store, "FIDO2", k=3)
    assert results
    assert results[0][0].metadata["source"] == "security.txt"


def test_bm25_idf_coverage_is_high_for_a_rare_term_match():
    store = make_store()
    _doc, _score, coverage = BM25Index().search(store, "FIDO2", k=1)[0]
    assert coverage > 0.9


def test_bm25_idf_coverage_is_low_for_a_stopword_only_match():
    # The whole reason the gate measures IDF coverage rather than a
    # normalised BM25 score: every document shares "the"/"is"/"on" with
    # every question, and a normalised score makes the best such match
    # look like a perfect one.
    store = make_store()
    results = BM25Index().search(store, "what is the on a", k=3)
    assert all(coverage < 0.5 for _doc, _score, coverage in results)


def test_bm25_index_rebuilds_when_documents_are_added():
    # Enrichment writes accepted web findings into the store. If the
    # lexical index didn't notice, the loop would keep re-searching the
    # web for a term already sitting in the knowledge base.
    store = make_store()
    index = BM25Index()
    assert index.search(store, "quantum", k=3) == []

    store.add_documents([
        Document(page_content="Quantum resistant ciphers are not supported.",
                 metadata={"source": "web:https://example.com"})
    ])
    results = index.search(store, "quantum", k=3)
    assert len(results) == 1


def test_bm25_on_an_empty_store_returns_nothing():
    assert BM25Index().search(make_store([]), "anything", k=3) == []


# --- the gate ---
#
# Tested as pure logic on explicit score dictionaries rather than through
# a store. DeterministicFakeEmbedding isn't unit-normalised, so its L2
# distances are in the hundreds and Chroma reports relevance scores far
# outside [0,1] - which the gate deliberately treats as "unusable, fail
# open". That behaviour is itself asserted below, but it means a fake
# store can't be used to exercise a gate *decision*.

def gate(dense=None, lexical=None, threshold=0.4, dense_available=True,
         rerank=None, rerank_threshold=0.05):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.min_relevance = threshold
    retriever.rerank_min_score = rerank_threshold
    return retriever._passes_gate(
        "k",
        {"k": dense} if dense is not None else {},
        {"k": lexical or 0.0},
        dense_available,
        {"k": rerank} if rerank is not None else {},
    )


def test_gate_keeps_a_dense_relevant_chunk():
    assert gate(dense=0.7) is True


def test_gate_rejects_a_dense_irrelevant_chunk():
    assert gate(dense=0.12) is False


def test_gate_keeps_a_strong_lexical_match_with_no_dense_support():
    # The point of hybrid retrieval: a rare exact term is evidence on its
    # own, even when the embedding model has nothing to say. A chunk that
    # wasn't a dense candidate at all has dense=None.
    assert gate(dense=None, lexical=0.8) is True


def test_gate_rejects_a_lexical_only_chunk_with_weak_coverage():
    # The bug this pins down: a BM25 candidate outside the dense
    # candidate set has no dense score, and treating that as "the store
    # can't score" let every such chunk bypass the gate. Measured
    # effect was 0% rejection of off-topic questions under hybrid
    # retrieval, against 100% under dense-only.
    assert gate(dense=None, lexical=0.1) is False


def test_gate_fails_open_when_the_store_cannot_score_at_all():
    assert gate(dense=None, lexical=0.0, dense_available=False) is True


def test_gate_fails_open_on_an_out_of_range_score():
    assert gate(dense=-519.45) is True


def test_gate_disabled_by_zero_threshold():
    assert gate(dense=0.0, lexical=0.0, threshold=0.0) is True


def test_rerank_score_decides_alone_when_present():
    # The cross-encoder scored this pair directly, so the weaker proxies
    # don't get to overrule it in either direction.
    assert gate(dense=0.99, lexical=0.99, rerank=0.001) is False
    assert gate(dense=0.01, lexical=0.0, rerank=0.9) is True


def test_rerank_failure_falls_back_to_the_other_signals():
    # _rerank_scores returns {} when the model is unavailable, which must
    # degrade to the previous behaviour rather than reject everything.
    assert gate(dense=0.7, rerank=None) is True
    assert gate(dense=0.12, rerank=None) is False


def test_reranker_is_off_in_tests_by_default():
    # Guards the offline promise: conftest.py disables it suite-wide.
    from src.config import config
    assert config.RERANK is False
    assert HybridRetriever(make_store()).rerank is False


# --- retriever wiring ---

def test_gate_fails_open_for_fake_embeddings_end_to_end():
    # Documents the consequence of the rule above: a store whose
    # relevance scores are unusable degrades to unfiltered retrieval
    # rather than to returning nothing.
    retriever = HybridRetriever(make_store(), min_relevance=0.9)
    assert len(retriever.retrieve("a completely unrelated question")) > 0


def test_hybrid_surfaces_a_rare_term_match_first():
    retriever = HybridRetriever(make_store(), hybrid=True, min_relevance=0.0)
    docs = retriever.retrieve("FIDO2", k=1)
    assert [d.metadata["source"] for d in docs] == ["security.txt"]


def test_retrieve_respects_k():
    retriever = HybridRetriever(make_store(), min_relevance=0.0)
    assert len(retriever.retrieve("plans", k=2)) == 2


def test_retrieve_on_empty_store_returns_empty_and_sets_debug():
    retriever = HybridRetriever(make_store([]), min_relevance=0.0)
    assert retriever.retrieve("anything") == []
    assert retriever.last_debug["kept"] == 0


def test_last_debug_reports_gate_activity():
    retriever = HybridRetriever(make_store(), min_relevance=0.0)
    retriever.retrieve("plans")
    assert retriever.last_debug["lexical"] > 0
    assert retriever.last_debug["gated_out"] == 0
