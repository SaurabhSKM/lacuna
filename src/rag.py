"""
Core RAG building blocks: retrieve, format context, generate a structured
answer, resolve sources.

generate_answer() is deliberately a standalone function (not buried inside
answer_question()) - Phase 3's LangGraph Answerer node calls this exact
function, so the CLI path (answer_question) and the graph path share one
implementation instead of drifting into two slightly different versions
of "how do we generate an answer."
"""
from src.config import config
from src.llm import get_llm, invoke_structured, MissingAPIKeyError
from src.schemas import RAGResponse
from src import tracing

STRUCTURED_PROMPT_TEMPLATE = """You are answering a question using ONLY the \
numbered context blocks below. Do not use outside knowledge.

Context:
---
{context}
---

Question: {question}

Fill in the response fields honestly. In particular: your confidence score \
must reflect how well the CONTEXT covers the question, not how fluent your \
answer sounds. If the context only partially addresses the question, say \
so in missing_info rather than filling gaps with assumptions."""


def format_context(retrieved_docs) -> str:
    """Join retrieved chunks into a single context block, each labeled
    with a [Source N] tag the LLM can reference by number."""
    parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        parts.append(f"[Source {i}]\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieve(vectorstore, question: str, k: int = None):
    k = k or config.TOP_K
    return vectorstore.similarity_search(question, k=k)


def resolve_sources(retrieved_docs, used_source_ids) -> list:
    """
    Map the LLM's chosen [Source N] numbers back to actual filenames.
    Kept as a standalone, pure function (no LLM call inside it) so it can
    be unit-tested directly without needing an API key or network access -
    the LLM only ever has to pick integers from a small known range, which
    is a far lower-hallucination-risk task than generating filenames itself.
    """
    sources = []
    for source_id in used_source_ids:
        if 1 <= source_id <= len(retrieved_docs):
            filename = retrieved_docs[source_id - 1].metadata.get("source", "unknown")
            if filename not in sources:
                sources.append(filename)
    return sources


def generate_answer(context: str, question: str, max_retries: int = 2) -> RAGResponse:
    """
    The Answerer step, on its own: given context + question, return a
    validated RAGResponse. Raises MissingAPIKeyError or RuntimeError on
    failure - callers decide how to handle that (the CLI turns it into an
    error dict; the LangGraph node turns it into a state["error"] field).
    """
    prompt = STRUCTURED_PROMPT_TEMPLATE.format(context=context, question=question)
    llm = get_llm()  # raises MissingAPIKeyError if not configured
    return invoke_structured(
        llm, RAGResponse, prompt, max_retries=max_retries, callbacks=tracing.get_callbacks()
    )


def answer_question(vectorstore, question: str, k: int = None, max_retries: int = 2) -> dict:
    """
    Full non-graph pipeline (Phase 2's entry point, kept as-is for the CLI
    and as a baseline to compare the LangGraph version against later).
    Returns a dict containing either a validated RAGResponse (under
    "structured") or an "error" - callers should always check "error"
    before touching "structured".
    """
    retrieved_docs = retrieve(vectorstore, question, k=k)
    context = format_context(retrieved_docs)

    result = {
        "question": question,
        "retrieved_docs": retrieved_docs,
        "context": context,
        "structured": None,
        "sources": [],
        "error": None,
    }

    try:
        parsed = generate_answer(context, question, max_retries=max_retries)
    except (MissingAPIKeyError, RuntimeError) as e:
        result["error"] = str(e)
        return result

    result["structured"] = parsed
    result["sources"] = resolve_sources(retrieved_docs, parsed.used_source_ids)
    return result
