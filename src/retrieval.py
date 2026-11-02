"""
Hybrid retrieval: dense vector search fused with BM25 lexical search,
then gated on relevance.

Why this replaced a plain `vectorstore.similarity_search(question, k=4)`:

**Dense search alone misses exact terms.** An embedding model maps text
to meaning, which is what makes it good at "what happens if a payment
fails" matching a chunk that says "failed payments result in a grace
period". The same property makes it weak on rare literal tokens -
"FIDO2", "APPI", "SOC 2 Type II", a plan name, an error code. Those are
precisely the tokens a user quotes when they want one specific fact.
BM25 is the opposite: no semantics at all, but it ranks a rare matching
term very highly. Fusing the two covers both, and it's the single
cheapest retrieval improvement available here - no extra model, no extra
API call.

**A vector store always answers.** Ask for the 4 nearest neighbours and
you get 4, however far away they are. With no relevance floor, an
off-topic question retrieves four unrelated chunks, the context looks
full, and the model is invited to assemble something plausible out of
noise. That matters much more here than in a normal RAG app: Lacuna's
entire premise is distinguishing "retrieved something relevant" from
"retrieved enough", and an unfiltered retriever quietly destroys the
first signal. With the floor, an off-topic question retrieves *nothing*,
which is an honest and far more useful input to the rest of the graph.

The two rankings are combined with Reciprocal Rank Fusion rather than by
mixing scores. Dense relevance and BM25 scores aren't on a common scale
and BM25's upper bound moves with the corpus, so any weighted sum needs
recalibrating every time the corpus changes. RRF only reads *ranks*, so
there is nothing to calibrate.
"""
import hashlib
import math
import re

from src.config import config

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Lowercase alphanumeric tokens. Deliberately simple: BM25 wants
    consistency between indexing and querying far more than it wants
    linguistic sophistication."""
    return _TOKEN.findall(text.lower())


def _doc_key(doc) -> str:
    """
    Fusion needs a stable identity for a chunk across two rankings.
    Chroma populates Document.id, which is exactly that. Content hash is
    the fallback for stores that don't (the in-memory fakes in tests, for
    instance) - two chunks with identical text are interchangeable for
    our purposes anyway.
    """
    if getattr(doc, "id", None):
        return str(doc.id)
    return hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(rankings: list, rrf_k: int = None) -> dict:
    """
    Standard RRF: an item's score is the sum of 1/(rrf_k + rank) over
    every ranking it appears in (rank is 1-based).

    `rankings` is a list of lists of keys, best-first. Returns
    {key: fused_score}. Pure function, no I/O - unit-tested directly.
    """
    rrf_k = config.RRF_K if rrf_k is None else rrf_k
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return scores


class BM25Index:
    """
    A lexical index over the whole collection.

    Rebuilt lazily, keyed on a fingerprint of the collection's ids. That
    matters because of enrichment: when an accepted answer persists web
    findings back into the store, those new chunks have to become
    *lexically* searchable too, or the enrichment loop would keep
    re-searching the web for a term that is now sitting in the knowledge
    base. Watching the id set means an upload or an auto-ingest
    invalidates the index without anything having to remember to call us.
    """

    def __init__(self):
        self._fingerprint = None
        self._keys = []
        self._docs = []
        self._tokenized = []
        self._df = {}
        self._avg_len = 0.0

    @staticmethod
    def _fingerprint_of(vectorstore) -> str:
        ids = vectorstore.get(include=[])["ids"]
        h = hashlib.sha1()
        for i in sorted(ids):
            h.update(str(i).encode("utf-8"))
        return f"{len(ids)}:{h.hexdigest()}"

    def _rebuild(self, vectorstore):
        from langchain_core.documents import Document

        raw = vectorstore.get(include=["documents", "metadatas"])
        self._keys = [str(i) for i in raw["ids"]]
        self._docs = [
            Document(page_content=text or "", metadata=meta or {}, id=str(doc_id))
            for doc_id, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
        self._tokenized = [tokenize(d.page_content) for d in self._docs]

        self._df = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        lengths = [len(t) for t in self._tokenized]
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _idf(self, term: str) -> float:
        """Standard BM25 inverse document frequency."""
        n = len(self._docs)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, vectorstore, query: str, k: int) -> list:
        """
        Return [(Document, bm25_score, idf_coverage)] best-first, at most
        k items.

        `idf_coverage` is the share of the query's total IDF mass that
        this document actually matched, and it exists because the BM25
        score alone cannot support a relevance decision. BM25 scores have
        no absolute scale, so the natural move is to normalise against
        the best candidate for the query - at which point the top hit
        always scores 1.0, even when the only terms it matched were "what"
        and "the". Measured consequence: the relevance gate let through
        every single off-topic question, because every question shares
        stopwords with every document.

        IDF coverage fixes that by asking a different question: not "how
        does this document rank" but "how much of what the user asked for
        is rare, and did this document match the rare part". A document
        matching only high-document-frequency terms has a coverage near
        zero however it ranks.
        """
        fingerprint = self._fingerprint_of(vectorstore)
        if fingerprint != self._fingerprint:
            self._rebuild(vectorstore)
            self._fingerprint = fingerprint

        if not self._docs:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        query_idf = {term: self._idf(term) for term in set(query_terms)}
        total_idf = sum(query_idf.values())

        k1, b = 1.5, 0.75
        scored = []
        for doc, tokens in zip(self._docs, self._tokenized):
            if not tokens:
                continue
            doc_len = len(tokens)
            score = 0.0
            matched_idf = 0.0
            for term in set(query_terms):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                matched_idf += query_idf[term]
                score += query_idf[term] * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * doc_len / (self._avg_len or 1))
                )
            if score > 0:
                scored.append((doc, score, matched_idf / total_idf if total_idf else 0.0))

        scored.sort(key=lambda triple: triple[1], reverse=True)
        return scored[:k]


class HybridRetriever:
    """
    Holds the BM25 index alongside the vector store. One instance per
    vector store, built once by the graph, so the lexical index is
    reused across questions instead of rebuilt per query.
    """

    def __init__(
        self,
        vectorstore,
        hybrid: bool = None,
        min_relevance: float = None,
        rerank: bool = None,
        rerank_min_score: float = None,
    ):
        self.vectorstore = vectorstore
        self.hybrid = config.HYBRID_RETRIEVAL if hybrid is None else hybrid
        self.min_relevance = config.MIN_RELEVANCE if min_relevance is None else min_relevance
        self.rerank = config.RERANK if rerank is None else rerank
        self.rerank_min_score = (
            config.RERANK_MIN_SCORE if rerank_min_score is None else rerank_min_score
        )
        self._bm25 = BM25Index()
        # Populated by the last retrieve() call, for /stats and for the
        # CLI's --show-context output. Diagnostics only; nothing routes on it.
        self.last_debug = {}

    def _rerank_scores(self, question: str, docs: list) -> dict:
        """
        {doc_key: relevance} from the cross-encoder, or {} if reranking
        is off or unavailable. Returning {} on failure is deliberate:
        the reranker is a precision refinement over a candidate pool
        that is already correctly ordered by fusion, so losing it should
        degrade retrieval to the previous behaviour rather than break it.
        """
        if not self.rerank or not docs:
            return {}
        try:
            from src.rerank import get_reranker

            scores = get_reranker().score(question, [d.page_content for d in docs])
        except Exception:
            return {}
        return {_doc_key(doc): score for doc, score in zip(docs, scores)}

    def _dense(self, question: str, k: int) -> list:
        """[(Document, relevance)] best-first. relevance is in [0,1]."""
        try:
            return self.vectorstore.similarity_search_with_relevance_scores(question, k=k)
        except Exception:
            # Some stores/metrics can't produce a normalised relevance.
            # Fall back to an unscored search and let the gate fail open
            # rather than dropping retrieval entirely.
            return [(doc, None) for doc in self.vectorstore.similarity_search(question, k=k)]

    def retrieve(self, question: str, k: int = None) -> list:
        k = k or config.TOP_K
        candidate_k = max(config.CANDIDATE_K, k)

        dense = self._dense(question, candidate_k)
        by_key = {}
        dense_relevance = {}
        for doc, score in dense:
            key = _doc_key(doc)
            by_key[key] = doc
            dense_relevance[key] = score

        lexical = []
        lexical_coverage = {}
        if self.hybrid:
            try:
                lexical = self._bm25.search(self.vectorstore, question, candidate_k)
            except Exception:
                lexical = []  # lexical search is additive; never fatal
            for doc, _score, coverage in lexical:
                key = _doc_key(doc)
                by_key.setdefault(key, doc)
                lexical_coverage[key] = coverage

        if not by_key:
            self.last_debug = {"dense": 0, "lexical": 0, "kept": 0, "gated_out": 0}
            return []

        fused = reciprocal_rank_fusion([
            [_doc_key(doc) for doc, _ in dense],
            [_doc_key(doc) for doc, _score, _cov in lexical],
        ])
        ranked = sorted(by_key, key=lambda key: fused.get(key, 0.0), reverse=True)

        # Precision stage. Fusion decided what is in the pool; the
        # cross-encoder decides the order within it, by actually reading
        # the question and the chunk together. When it's available its
        # score also replaces vector distance as the gate's input, which
        # is the question the gate was always trying to ask.
        rerank_scores = self._rerank_scores(question, [by_key[key] for key in ranked])
        if rerank_scores:
            ranked.sort(key=lambda key: rerank_scores.get(key, 0.0), reverse=True)

        # Whether the store could produce normalised relevance at all, as
        # distinct from a particular chunk having no dense score because
        # it was a lexical-only candidate. Conflating those two is a trap:
        # it makes every BM25 hit outside the dense candidate set bypass
        # the gate, which measured out as the relevance floor rejecting
        # 0% of off-topic questions under hybrid retrieval while rejecting
        # half of them under dense-only.
        dense_scores_available = any(score is not None for _doc, score in dense)

        kept = [
            key for key in ranked
            if self._passes_gate(
                key, dense_relevance, lexical_coverage, dense_scores_available, rerank_scores
            )
        ]
        self.last_debug = {
            "dense": len(dense),
            "lexical": len(lexical),
            "reranked": len(rerank_scores),
            "kept": len(kept[:k]),
            "gated_out": len(ranked) - len(kept),
            "scores": {
                key: {
                    "dense": dense_relevance.get(key),
                    "lexical": round(lexical_coverage.get(key, 0.0), 3),
                    "rerank": round(rerank_scores[key], 4) if key in rerank_scores else None,
                    "fused": round(fused.get(key, 0.0), 5),
                }
                for key in ranked[:k]
            },
        }
        return [by_key[key] for key in kept[:k]]

    def _passes_gate(
        self,
        key,
        dense_relevance: dict,
        lexical_coverage: dict,
        dense_scores_available: bool = True,
        rerank_scores: dict = None,
    ) -> bool:
        """
        A chunk survives if something has real evidence for it.

        When a reranker score is available it decides alone, and the
        other two signals are ignored. That isn't a shortcut: dense
        distance and IDF coverage are both proxies for "is this chunk
        about the question", used because scoring the pair directly over
        a whole collection is too expensive. Once the pair *has* been
        scored directly over the candidate pool, consulting the proxies
        as well would only let a chunk the cross-encoder rejected back in
        on weaker evidence.

        Dense side: relevance >= min_relevance. The threshold is
        calibrated against Chroma's default L2 metric, where an off-topic
        query scores around 0.1-0.3 and a good match 0.5-0.8. A collection
        created with cosine distance compresses that range upwards (an
        off-topic query floors out near 0.4), so a cosine collection needs
        a higher MIN_RELEVANCE. Scores outside [0,1], or absent entirely,
        fail the gate open - a store that can't report relevance should
        degrade to the old unfiltered behaviour, not to returning nothing.

        Lexical side: the share of the query's IDF mass this chunk
        matched (see BM25Index.search). A chunk that matched a rare term
        the embedding model missed - "FIDO2", "429", "X-RateLimit-Reset" -
        clears the bar on its own, which is the entire reason for having
        lexical search here. A chunk that merely shares stopwords with the
        question does not.
        """
        if self.min_relevance <= 0:
            return True

        rerank_scores = rerank_scores or {}
        if key in rerank_scores:
            return rerank_scores[key] >= self.rerank_min_score

        if lexical_coverage.get(key, 0.0) >= self.min_relevance:
            return True

        if not dense_scores_available:
            return True  # the store can't score at all: fail open

        relevance = dense_relevance.get(key)
        if relevance is None:
            # A lexical-only candidate that didn't clear the lexical bar
            # above. No evidence from either side.
            return False
        if not (0.0 <= relevance <= 1.0):
            return True  # unusable score for this chunk: fail open
        return relevance >= self.min_relevance
