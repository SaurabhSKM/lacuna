"""
Central place for all configuration. Everything is read from environment
variables (loaded from .env), so nothing here should be a hardcoded secret.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Silence Chroma's anonymous telemetry (harmless if left on, just noisy)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class Config:
    # --- LLM provider ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # --- Embeddings (local, no key needed) ---
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Chunking ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

    # --- Retrieval ---
    TOP_K = int(os.getenv("TOP_K", "4"))

    # --- Enrichment loop ---
    MAX_ENRICHMENT_ITERATIONS = int(os.getenv("MAX_ENRICHMENT_ITERATIONS", "2"))
    WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))

    # --- Observability (optional - tracing no-ops if these are unset) ---
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # --- Paths ---
    DEFAULT_PERSIST_DIR = "./chroma_db"
    UPLOAD_DIR = "./data/uploads"


config = Config()
