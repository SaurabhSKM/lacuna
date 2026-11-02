"""
A thin HTTP layer over the exact same functions the CLI scripts already
call. No logic lives here that isn't already in src/ingest.py,
src/graph.py, etc - this file's only job is request/response handling.

Design choice: lazy, in-memory state instead of a startup event.
_state["vectorstore"] is created on first use (whichever request
happens first, /upload or /query) rather than in a lifespan hook. Two
reasons:
  1. It means both /upload and /query mutate the SAME live vectorstore
     object across requests, so a document uploaded via the API is
     immediately searchable without reopening anything from disk.
  2. It makes the whole HTTP layer testable without a real embedding
     model download - tests can pre-populate _state before the first
     request, and get_state() will see it's already set and skip
     real initialization entirely.
"""
import json
import os
import secrets
import shutil
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from src.config import config
from src.ingest import (
    load_vectorstore,
    load_single_document,
    chunk_documents,
    SUPPORTED_EXTENSIONS,
)
from src.embeddings import get_embedding_model
from src.graph import build_graph, run_question
from src.retrieval import HybridRetriever
from src.schemas import Citation

app = FastAPI(
    title="Lacuna",
    description="A self-correcting RAG knowledge base.",
    version="0.8.0",
)

if config.CORS_ORIGINS:
    # Only mounted when origins are configured. A service with no browser
    # frontend has no reason to advertise CORS, and defaulting to "*"
    # would be loosening the default for nobody's benefit.
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

FEEDBACK_LOG_PATH = "./data/feedback.jsonl"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

_state = {"vectorstore": None, "graph_app": None, "retriever": None}


class PendingClarifications:
    """
    Open clarification requests, keyed by an opaque id.

    Deliberately small: a clarification turn needs the original question
    and the exchange so far, and nothing else. The graph run itself is
    not checkpointed - resuming re-runs it with the clarification
    attached, because a clarification changes which documents are
    relevant and invalidates whatever the previous run retrieved (see
    graph.run_question).

    That keeps what has to be stored to a few strings, which is why an
    in-process dict is defensible here. It is still process-local: a
    second replica won't recognise the id, and a restart forgets them.
    Moving this to Redis is the change to make before running more than
    one instance, and it is a change of backing store only - nothing
    else in the pipeline becomes stateful.

    Entries expire, and the store is bounded, so an abandoned
    conversation cannot accumulate into a slow memory leak or let a
    caller pin memory open by never answering.
    """

    def __init__(self, ttl_seconds: int = None, max_entries: int = None):
        # `is None`, not `or`: a ttl of 0 is a legitimate value meaning
        # "expire immediately", and `0 or 3600` silently becomes 3600.
        self.ttl = config.CLARIFICATION_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self.max_entries = (
            config.MAX_PENDING_CLARIFICATIONS if max_entries is None else max_entries
        )
        self._entries = OrderedDict()

    def _evict(self):
        now = time.monotonic()
        for key in [k for k, v in self._entries.items() if now - v["created"] > self.ttl]:
            del self._entries[key]
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)  # oldest first

    def open(self, question: str, clarifications: list, request: str) -> str:
        token = secrets.token_urlsafe(16)
        self._entries[token] = {
            "question": question,
            "clarifications": list(clarifications or []),
            "request": request,
            "created": time.monotonic(),
        }
        # Evicted after inserting, not before: evicting first leaves room
        # for the new entry and lets the store sit one over its bound.
        self._evict()
        return token

    def take(self, token: str):
        """
        Fetch and remove. Removing on use means a token is single-shot:
        answering the same request twice would otherwise fork the
        conversation off one id, and the second answer would silently
        replace the first.
        """
        self._evict()
        return self._entries.pop(token, None)

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)


_pending = PendingClarifications()


def get_state() -> dict:
    if _state["vectorstore"] is None:
        from langchain_chroma import Chroma

        persist_dir = config.DEFAULT_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)

        if os.listdir(persist_dir):
            vectorstore = load_vectorstore(persist_dir)
        else:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=get_embedding_model(),
            )

        _state["vectorstore"] = vectorstore

    if _state["graph_app"] is None:
        # Built separately from the vectorstore so a test that injects
        # only a vectorstore still gets a graph, and so the retriever
        # (which owns the BM25 index) is reachable from /stats.
        retriever = _state.get("retriever") or HybridRetriever(_state["vectorstore"])
        _state["retriever"] = retriever
        _state["graph_app"] = build_graph(_state["vectorstore"], retriever=retriever)

    return _state


# --- Request/response contracts ---

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


class ClarificationTurn(BaseModel):
    question_asked: str
    user_answer: str = Field(min_length=1, max_length=2000)


class QueryRequest(BaseModel):
    # min_length=1 so a blank question is a 422 from the contract rather
    # than an embedding call on an empty string followed by a confused
    # answer about nothing.
    question: str = Field(min_length=1, max_length=4000)
    # Lets a stateless client carry the exchange itself instead of
    # holding a server-side token - useful behind a load balancer, and
    # the reason the graph takes clarifications as plain data rather
    # than hiding them in a checkpoint.
    clarifications: List[ClarificationTurn] = []


class ClarifyRequest(BaseModel):
    clarification_id: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    status: str
    # Present when status is "needs_clarification": what to ask the user,
    # and the token to send their reply back with.
    clarification_request: Optional[str] = None
    clarification_id: Optional[str] = None
    clarifications: List[ClarificationTurn] = []
    answer: Optional[str] = None
    confidence: Optional[float] = None
    missing_info: List[str] = []
    sources: List[str] = []
    citations: List[Citation] = []
    web_sources: List[dict] = []
    completeness_action: Optional[str] = None
    completeness_sufficiency: Optional[float] = None
    completeness_gap: Optional[str] = None
    completeness_reasoning: Optional[str] = None
    iteration_count: int = 0
    enrichment_log: List[dict] = []
    retrieval_debug: dict = {}
    persisted_web_sources: int = 0
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    chunks: int
    auto_ingested_chunks: int
    sources: int


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats():
    """
    What's actually in the knowledge base. Mostly useful for seeing the
    self-improving behaviour from the outside: auto_ingested_chunks is
    the count of chunks that entered the store through an accepted
    enrichment rather than through an upload.
    """
    state = get_state()
    try:
        stored = state["vectorstore"].get(include=["metadatas"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read the store: {e}")

    metadatas = [m or {} for m in (stored.get("metadatas") or [])]
    return StatsResponse(
        chunks=len(metadatas),
        auto_ingested_chunks=sum(1 for m in metadatas if m.get("auto_ingested")),
        sources=len({m.get("source") for m in metadatas if m.get("source")}),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    state = get_state()

    # Streamed to a temp file, ingested, then deleted in the `finally`
    # below. Nothing is retained: the chunks live in the vector store and
    # the original upload has no further use, so keeping it would just be
    # an unbounded directory of other people's documents.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
        tmp_path = tmp.name

    try:
        size = os.path.getsize(tmp_path)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is {size} bytes; the limit is {MAX_UPLOAD_BYTES}.",
            )

        docs = load_single_document(tmp_path)
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in that file (an image-only PDF?).",
            )
        for doc in docs:
            doc.metadata["source"] = file.filename  # real name, not the temp path
        chunks = chunk_documents(docs)
        state["vectorstore"].add_documents(chunks)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {e}")
    finally:
        os.unlink(tmp_path)

    return UploadResponse(filename=file.filename, chunks_added=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    state = get_state()
    clarifications = [c.model_dump() for c in payload.clarifications]
    result = run_question(state["graph_app"], payload.question, clarifications)
    return _to_response(payload.question, clarifications, result)


@app.post("/clarify", response_model=QueryResponse)
def clarify(payload: ClarifyRequest):
    """
    Answer an outstanding clarification request and continue.

    This is what makes `ask_user` a real outcome rather than a dead end:
    the critic identifies that a question is underspecified, names the
    ambiguity, and the user's reply comes back in here to finish the job.
    """
    pending = _pending.take(payload.clarification_id)
    if pending is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Unknown or expired clarification_id. Clarification requests "
                "are single-use and expire; ask the question again to get a "
                "new one."
            ),
        )

    clarifications = pending["clarifications"] + [
        {"question_asked": pending["request"], "user_answer": payload.answer}
    ]
    state = get_state()
    result = run_question(state["graph_app"], pending["question"], clarifications)
    return _to_response(pending["question"], clarifications, result)


def _to_response(question: str, clarifications: list, result: dict) -> QueryResponse:
    answer = result.get("structured_answer")
    verdict = result.get("completeness_verdict")
    status = result.get("final_status") or "unknown"

    # A clarification request is only resumable if there's a token to
    # resume it with, so the token is minted here rather than in the
    # graph - the graph stays free of transport concerns.
    clarification_id = None
    request = result.get("clarification_request")
    if status == "needs_clarification" and request:
        clarification_id = _pending.open(question, clarifications, request)

    return QueryResponse(
        status=status,
        clarification_request=request,
        clarification_id=clarification_id,
        clarifications=[ClarificationTurn(**c) for c in clarifications],
        answer=answer.answer if answer else None,
        confidence=answer.confidence if answer else None,
        missing_info=answer.missing_info if answer else [],
        sources=result.get("sources", []),
        citations=result.get("citations", []),
        web_sources=result.get("web_sources", []),
        completeness_action=verdict.action if verdict else None,
        completeness_sufficiency=verdict.sufficiency_score if verdict else None,
        completeness_gap=verdict.gap_description if verdict else None,
        completeness_reasoning=verdict.reasoning if verdict else None,
        iteration_count=result.get("iteration_count", 0),
        enrichment_log=result.get("enrichment_log", []),
        retrieval_debug=result.get("retrieval_debug", {}),
        persisted_web_sources=result.get("persisted_web_sources", 0),
        error=result.get("error"),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest):
    """
    Lets users rate answer quality. This is deliberately minimal - it
    appends a JSONL record, it doesn't feed back into the pipeline
    automatically. That's an honest scope boundary: turning this into an
    actual improvement loop (e.g. re-weighting retrieval, flagging
    documents for review) is real further work, not something to fake as
    already built.
    """
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    entry = payload.model_dump()
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return FeedbackResponse(status="recorded")
