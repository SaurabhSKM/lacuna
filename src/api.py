"""
Phase 5: a thin HTTP layer over the exact same functions the CLI scripts
already call. No logic lives here that isn't already in src/ingest.py,
src/graph.py, etc - this file's only job is request/response handling.

Design choice: lazy, in-memory state instead of a startup event.
_state["vectorstore"] is created on first use (whichever request
happens first, /upload or /query) rather than in an @app.on_event
("startup") hook. Two reasons:
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
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.config import config
from src.ingest import load_vectorstore, load_single_document, chunk_documents, get_embedding_model
from src.graph import build_graph

app = FastAPI(
    title="Lacuna",
    description="A self-correcting RAG knowledge base.",
    version="0.7.0",
)

FEEDBACK_LOG_PATH = "./data/feedback.jsonl"

_state = {"vectorstore": None, "graph_app": None}


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
        _state["graph_app"] = build_graph(vectorstore)

    return _state


# --- Request/response contracts ---

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    confidence: Optional[float] = None
    missing_info: List[str] = []
    sources: List[str] = []
    web_sources: List[dict] = []
    completeness_action: Optional[str] = None
    completeness_sufficiency: Optional[float] = None
    completeness_gap: Optional[str] = None
    iteration_count: int = 0
    persisted_web_sources: int = 0
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".txt", ".md", ".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .txt, .md, .pdf",
        )

    state = get_state()

    # Write to a temp file rather than into UPLOAD_DIR directly during
    # the request - avoids partially-written files being picked up if
    # something reads the upload directory concurrently.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        docs = load_single_document(tmp_path)
        for doc in docs:
            doc.metadata["source"] = file.filename  # real name, not the temp path
        chunks = chunk_documents(docs)
        state["vectorstore"].add_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {e}")
    finally:
        os.unlink(tmp_path)

    return UploadResponse(filename=file.filename, chunks_added=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    state = get_state()
    graph_app = state["graph_app"]

    initial_state = {
        "question": payload.question,
        "retrieved_docs": [],
        "doc_context": "",
        "web_sources": [],
        "full_context": "",
        "structured_answer": None,
        "sources": [],
        "completeness_verdict": None,
        "iteration_count": 0,
        "enrichment_log": [],
        "persisted_web_sources": 0,
        "final_status": "",
        "error": None,
    }

    result = graph_app.invoke(initial_state)

    answer = result.get("structured_answer")
    verdict = result.get("completeness_verdict")

    return QueryResponse(
        status=result.get("final_status", "unknown"),
        answer=answer.answer if answer else None,
        confidence=answer.confidence if answer else None,
        missing_info=answer.missing_info if answer else [],
        sources=result.get("sources", []),
        web_sources=result.get("web_sources", []),
        completeness_action=verdict.action if verdict else None,
        completeness_sufficiency=verdict.sufficiency_score if verdict else None,
        completeness_gap=verdict.gap_description if verdict else None,
        iteration_count=result.get("iteration_count", 0),
        persisted_web_sources=result.get("persisted_web_sources", 0),
        error=result.get("error"),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest):
    """
    Stretch goal from the original brief: let users rate answer quality.
    This is deliberately minimal - it appends a JSONL record, it doesn't
    feed back into the pipeline automatically. That's an honest scope
    boundary: turning this into an actual improvement loop (e.g.
    re-weighting retrieval, flagging documents for review) is real
    further work, not something to fake as already built.
    """
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    entry = payload.model_dump()
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(FEEDBACK_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return FeedbackResponse(status="recorded")
