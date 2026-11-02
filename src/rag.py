"""
Core RAG building blocks: retrieve, assemble a numbered source list,
format context, generate a structured answer, resolve citations.

generate_answer() is deliberately a standalone function (not buried
inside answer_question()) - the LangGraph Answerer node calls this exact
function, so the CLI path (answer_question) and the graph path share one
implementation instead of drifting into two slightly different versions
of "how do we generate an answer."

The source list is the other thing this file owns. Document chunks and
web results are numbered into a single [Source N] sequence, and
resolve_citations() maps the model's chosen numbers back to concrete
provenance. Previously the two lived in separate numbering universes:
web results were appended into the same visible sequence the model cites
from, but the resolver indexed only into the retrieved documents, so any
id pointing at a web result fell out of range and was silently dropped.
An answer could be built mostly from a web result and report no source
for it at all.
"""
from dataclasses import dataclass
from typing import List, Optional

from src.config import config
from src.llm import get_llm, invoke_structured, MissingAPIKeyError
from src.schemas import RAGResponse, Citation
from src import tracing

NO_CONTEXT_MARKER = (
    "(no relevant context was retrieved - the knowledge base contains "
    "nothing that passed the relevance threshold for this question)"
)

STRUCTURED_PROMPT_TEMPLATE = """You are answering a question using ONLY the \
numbered context blocks below. Do not use outside knowledge.

Each block is labelled with where it came from. Blocks marked `document:` \
are from the user's own knowledge base and are authoritative about their \
subject. Blocks marked `web:` came from a general web search and may be \
about a different product, company, or version than the question asks \
about - do not attribute a fact to the question's subject just because a \
web result states it in general terms. Where a document and a web result \
conflict, the document wins. Where only a web result covers something, say \
so in the answer.

If no context is provided at all, say that you cannot answer from the \
knowledge base. Do not fill the gap from memory.

Context:
---
{context}
---

Question: {question}
{clarifications}
Fill in the response fields honestly. In particular: your confidence score \
must reflect how well the CONTEXT covers the question, not how fluent your \
answer sounds. If the context only partially addresses the question, say \
so in missing_info rather than filling gaps with assumptions."""


CLARIFICATION_BLOCK_TEMPLATE = """
The user's question was ambiguous, and they have since clarified it. \
Treat the clarifications as part of the question - they are the user's \
own words and override any other reading of it:
{exchanges}
Answer the question as clarified. Do not ask for the same clarification \
again."""


def format_clarifications(clarifications: list) -> str:
    """
    Render the clarification exchange for a prompt. Pure function, so
    it's unit-testable without an LLM, same as resolve_citations.
    """
    if not clarifications:
        return ""
    lines = []
    for item in clarifications:
        lines.append(f"  Asked: {item.get('question_asked', '')}")
        lines.append(f"  User replied: {item.get('user_answer', '')}")
    return CLARIFICATION_BLOCK_TEMPLATE.format(exchanges="\n".join(lines))


def retrieval_query(question: str, clarifications: list = None) -> str:
    """
    The string actually used for retrieval, as distinct from the text
    shown to the model.

    The user's replies are appended, not the questions we asked them.
    The replies are where the disambiguating terms live ("the Team
    plan"), and they are what retrieval needs; the question we asked is
    our own wording and only adds noise to a lexical match.

    This is why the clarification loop re-retrieves instead of resuming
    a checkpoint. Measured on the sample corpus: "What is the price?"
    retrieves nothing at all - correctly, it is too vague to match
    anything above the relevance floor - while "What is the price? Team
    plan" retrieves three chunks. There is no useful retrieval state to
    restore, so the loop recomputes it.
    """
    if not clarifications:
        return question
    answers = " ".join(
        str(item.get("user_answer", "")).strip() for item in clarifications
    )
    return f"{question} {answers}".strip()


@dataclass
class SourceRef:
    """
    One numbered entry in the context the model sees. `source_id` is the
    N in [Source N]; it is the only thing the model has to reproduce, and
    it only ever has to pick an integer out of a small known range.

    That is still the original design property - the model never writes a
    filename, so it can never hallucinate one. The labels ARE now shown
    to it, which is a deliberate change: resolution happens in plain
    Python by integer lookup either way, so showing the provenance costs
    nothing in hallucination risk and buys the model the ability to tell a
    document from a web result and weigh them differently.
    """

    source_id: int
    kind: str  # "document" | "web"
    label: str
    text: str
    url: Optional[str] = None

    def to_citation(self) -> Citation:
        return Citation(
            kind=self.kind, label=self.label, url=self.url, source_id=self.source_id
        )


def _document_label(doc) -> str:
    """Filename (not the full path) plus a page number for PDFs."""
    raw = doc.metadata.get("source", "unknown")
    label = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or raw
    page = doc.metadata.get("page")
    return f"{label} p.{page}" if page else label


def build_source_refs(retrieved_docs, web_results=None) -> List[SourceRef]:
    """
    Number document chunks first, then web results, into one sequence.

    Documents come first and keep their numbering stable across
    enrichment rounds: the vector search runs once, so [Source 1] means
    the same chunk in round 2 as it did in round 1, and only the
    higher-numbered web entries grow. If web results were interleaved or
    numbered first, the same id would point at different text between
    rounds and the model's citations from the previous round would be
    quietly wrong.
    """
    refs = []
    for doc in retrieved_docs:
        refs.append(
            SourceRef(
                source_id=len(refs) + 1,
                kind="document",
                label=_document_label(doc),
                text=doc.page_content,
            )
        )
    for result in web_results or []:
        refs.append(
            SourceRef(
                source_id=len(refs) + 1,
                kind="web",
                label=result.get("title") or "untitled",
                text=result.get("body") or "",
                url=result.get("href") or None,
            )
        )
    return refs


def format_context(refs: List[SourceRef]) -> str:
    """
    Render the numbered source list as the context block.

    An empty list renders as an explicit marker rather than an empty
    string. A blank context reads to the model as "no instructions about
    context" and invites it to answer from training data; a sentence
    saying the knowledge base had nothing relevant is a fact it can
    actually report, and the critic can see the same thing.
    """
    if not refs:
        return NO_CONTEXT_MARKER

    parts = []
    for ref in refs:
        if ref.kind == "web":
            header = f"[Source {ref.source_id}] (web: {ref.label}"
            header += f" - {ref.url})" if ref.url else ")"
        else:
            header = f"[Source {ref.source_id}] (document: {ref.label})"
        parts.append(f"{header}\n{ref.text}")
    return "\n\n".join(parts)


def resolve_citations(refs: List[SourceRef], used_source_ids) -> List[Citation]:
    """
    Map the model's chosen [Source N] numbers back to concrete provenance.

    Pure function - no LLM call inside it - so it can be unit-tested
    without an API key or network access. Out-of-range ids (a
    hallucinated or off-by-one citation) are dropped rather than raising:
    a bad citation should cost the caller one source line, not the whole
    answer.
    """
    by_id = {ref.source_id: ref for ref in refs}
    citations = []
    seen = set()
    for source_id in used_source_ids:
        ref = by_id.get(source_id)
        if ref is None:
            continue
        # Dedupe on provenance, not on id: several chunks of one file are
        # one source as far as the caller is concerned.
        identity = (ref.kind, ref.url or ref.label)
        if identity in seen:
            continue
        seen.add(identity)
        citations.append(ref.to_citation())
    return citations


def resolve_sources(refs: List[SourceRef], used_source_ids) -> List[str]:
    """Flat display labels, for the CLI and for the API's `sources` field."""
    return [
        c.url or c.label for c in resolve_citations(refs, used_source_ids)
    ]


def retrieve(vectorstore, question: str, k: int = None):
    """
    Retrieve for a one-shot caller. The graph builds a HybridRetriever
    once and reuses it (so the BM25 index is built once, not per
    question); this wrapper exists for scripts and tests that just want
    chunks back for a single question.
    """
    from src.retrieval import HybridRetriever

    return HybridRetriever(vectorstore).retrieve(question, k=k)


def generate_answer(
    context: str, question: str, clarifications: list = None, max_attempts: int = None
) -> RAGResponse:
    """
    The Answerer step, on its own: given context + question, return a
    validated RAGResponse. Raises MissingAPIKeyError or RuntimeError on
    failure - callers decide how to handle that (the CLI turns it into an
    error dict; the LangGraph node turns it into a state["error"] field).
    """
    prompt = STRUCTURED_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        clarifications=format_clarifications(clarifications or []),
    )
    llm = get_llm()  # raises MissingAPIKeyError if not configured
    return invoke_structured(
        llm,
        RAGResponse,
        prompt,
        max_attempts=max_attempts,
        callbacks=tracing.get_callbacks(),
    )


def answer_question(
    vectorstore,
    question: str,
    k: int = None,
    max_attempts: int = None,
    clarifications: list = None,
) -> dict:
    """
    Full non-graph pipeline, kept as the CLI's simple path and as a
    baseline to compare the LangGraph version against. Returns a dict
    containing either a validated RAGResponse (under "structured") or an
    "error" - callers should always check "error" before touching
    "structured".
    """
    retrieved_docs = retrieve(
        vectorstore, retrieval_query(question, clarifications), k=k
    )
    refs = build_source_refs(retrieved_docs)
    context = format_context(refs)

    result = {
        "question": question,
        "retrieved_docs": retrieved_docs,
        "source_refs": refs,
        "context": context,
        "structured": None,
        "sources": [],
        "citations": [],
        "error": None,
    }

    try:
        parsed = generate_answer(
            context, question, clarifications=clarifications, max_attempts=max_attempts
        )
    except (MissingAPIKeyError, RuntimeError) as e:
        result["error"] = str(e)
        return result

    result["structured"] = parsed
    result["citations"] = resolve_citations(refs, parsed.used_source_ids)
    result["sources"] = [c.url or c.label for c in result["citations"]]
    return result
