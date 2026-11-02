"""
Central place for all configuration. Everything is read from environment
variables (loaded from .env), so nothing here should be a hardcoded secret.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Silence Chroma's anonymous telemetry (harmless if left on, just noisy)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _optional_float(name: str, default: float):
    """Like _float, but an explicitly empty value means None ("omit")
    rather than falling back to the default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return float(raw) if raw else None


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Config:
    # --- LLM provider ---
    # "groq" | "gemini" | "stub". "stub" is an offline, deterministic
    # provider (see src/llm.py) used by the test suite and for wiring
    # checks without a key - it does not produce real answers.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    # gemini-1.5-flash was retired; 2.5-flash is the current free-tier
    # workhorse and the same drop-in shape.
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Set LLM_TEMPERATURE to an empty value to omit it from the request
    # entirely. Several current models fix their own sampling and warn on
    # every single call if you send one anyway - gemini-*-flash-lite and
    # the reasoning models among them. Omitting beats suppressing the
    # warning, because the warning is telling the truth: the parameter
    # genuinely has no effect there.
    LLM_TEMPERATURE = _optional_float("LLM_TEMPERATURE", 0.1)
    # Attempts, not retries: 1 means "try once, don't retry".
    LLM_MAX_ATTEMPTS = _int("LLM_MAX_ATTEMPTS", 3)
    LLM_BACKOFF_SECONDS = _float("LLM_BACKOFF_SECONDS", 1.0)

    # --- Embeddings (local, no key needed) ---
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Chunking ---
    CHUNK_SIZE = _int("CHUNK_SIZE", 800)
    CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)

    # --- Retrieval ---
    TOP_K = _int("TOP_K", 4)
    # Hybrid retrieval: dense (vector) and lexical (BM25) candidates are
    # fused with Reciprocal Rank Fusion. CANDIDATE_K is how deep each
    # retriever is asked to go before fusion; it needs to exceed TOP_K or
    # fusion has nothing to rerank.
    HYBRID_RETRIEVAL = _bool("HYBRID_RETRIEVAL", True)
    CANDIDATE_K = _int("CANDIDATE_K", 12)
    RRF_K = _int("RRF_K", 60)  # the standard RRF smoothing constant
    # Chunks with neither enough dense relevance nor enough lexical
    # evidence are dropped before they ever reach the prompt. A vector
    # store always returns its k nearest neighbours, however far away
    # they are; without a floor, off-topic chunks pad the context and
    # invite the model to answer from noise. 0 disables the gate.
    #
    # 0.4 is measured, not guessed: `python eval/retrieval_eval.py
    # --sweep` walks the threshold over the labelled benchmark, and 0.4
    # is where off-topic questions are rejected 100% of the time while
    # hit@k is still 100%. It degrades past 0.45 as real answers start
    # being gated out. That knee depends on the corpus and on the
    # collection's distance metric, so re-run the sweep for a different
    # document set rather than inheriting this number.
    MIN_RELEVANCE = _float("MIN_RELEVANCE", 0.4)

    # Cross-encoder reranking of the fused candidates: hybrid retrieval
    # decides what makes the candidate pool (recall), the reranker reads
    # question and chunk together to decide order and what survives the
    # gate (precision).
    #
    # Default OFF, because it was measured and it lost on this corpus.
    # It trades 8 points of hit@k (100% -> 92%) for 15 points of
    # precision (56% -> 71%), and that is the wrong direction here: a
    # gated-out gold chunk means the corpus can no longer answer at all
    # and the question goes to web search, whereas thinner precision
    # just leaves padding the critic already copes with. It gated out
    # everything for two plainly answerable paraphrase questions
    # ("which permission levels can I assign" against a document that
    # says "roles"). It also had no ordering left to win - fusion alone
    # already scores MRR 1.000 on these 25 questions.
    #
    # Kept because the implementation is sound and the reasoning for it
    # still holds on a corpus where MRR isn't saturated; it needs
    # re-measuring there rather than trusting either default. Run
    # `eval/retrieval_eval.py` to compare, and `--sweep` for the
    # threshold.
    RERANK = _bool("RERANK", False)
    RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    # When reranking is on, the gate reads the reranker's score instead
    # of vector distance, so it needs its own threshold. Also measured
    # with `eval/retrieval_eval.py --sweep`.
    RERANK_MIN_SCORE = _float("RERANK_MIN_SCORE", 0.05)

    # --- Clarification loop ---
    # How many times the system may come back asking the user to
    # disambiguate before it stops asking and answers as best it can.
    # Same reasoning as the enrichment cap: a critic that can request a
    # clarification can request one forever, and "the question is still
    # unclear" is not a reason to keep a user in a loop.
    MAX_CLARIFICATIONS = _int("MAX_CLARIFICATIONS", 2)
    # How long an unanswered clarification request stays resumable.
    CLARIFICATION_TTL_SECONDS = _int("CLARIFICATION_TTL_SECONDS", 3600)
    MAX_PENDING_CLARIFICATIONS = _int("MAX_PENDING_CLARIFICATIONS", 1000)

    # --- Enrichment loop ---
    MAX_ENRICHMENT_ITERATIONS = _int("MAX_ENRICHMENT_ITERATIONS", 2)
    WEB_SEARCH_RESULTS = _int("WEB_SEARCH_RESULTS", 3)
    # On accept, only persist web results the answer actually cited, and
    # only if they aren't near-duplicates of something already stored.
    PERSIST_ONLY_CITED = _bool("PERSIST_ONLY_CITED", True)
    # Persist web findings only when the accepted answer also cites at
    # least one document from the user's own corpus.
    #
    # This exists because the unguarded version poisoned the knowledge
    # base during testing, and did it in a self-reinforcing way. The
    # vague question "What is the price?" was enriched, the search
    # returned Wikipedia and Merriam-Webster on the *concept* of price,
    # the critic accepted an answer built entirely from them, and all
    # three were written into the store. After that the same question
    # retrieved those chunks, looked well-supported, and was accepted
    # immediately - so the contamination made the bad answer more likely
    # every time, not less.
    #
    # A document citation is the signal that a web result was filling a
    # gap in a corpus-grounded answer, which is the designed use case,
    # rather than standing in for the corpus entirely. An answer with no
    # document behind it may still be a perfectly good answer - it just
    # isn't evidence about the user's documents, so it doesn't earn a
    # permanent place among them.
    PERSIST_REQUIRES_DOC_ANCHOR = _bool("PERSIST_REQUIRES_DOC_ANCHOR", True)
    PERSIST_DEDUP_THRESHOLD = _float("PERSIST_DEDUP_THRESHOLD", 0.95)

    # --- Observability (optional - tracing no-ops if these are unset) ---
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # --- Service ---
    # Comma-separated list of allowed CORS origins. Empty means no CORS
    # middleware at all, which is the right default for a service with no
    # browser frontend - adding "*" by default would be a gratuitous
    # loosening.
    CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

    # --- Paths ---
    # No UPLOAD_DIR: uploads are streamed to a temp file, ingested, and
    # deleted within the request, so there is nowhere for one to live.
    # The setting used to exist and nothing read it, which is worse than
    # not having it - a reader assumes uploads are being retained
    # somewhere, and a configurer assumes setting it changes something.
    DEFAULT_PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")


config = Config()
