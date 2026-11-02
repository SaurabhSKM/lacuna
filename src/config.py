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

    LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
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

    # --- Enrichment loop ---
    MAX_ENRICHMENT_ITERATIONS = _int("MAX_ENRICHMENT_ITERATIONS", 2)
    WEB_SEARCH_RESULTS = _int("WEB_SEARCH_RESULTS", 3)
    # On accept, only persist web results the answer actually cited, and
    # only if they aren't near-duplicates of something already stored.
    PERSIST_ONLY_CITED = _bool("PERSIST_ONLY_CITED", True)
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
