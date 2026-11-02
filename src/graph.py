"""
The LangGraph state machine.

Retriever runs once. Answerer -> Completeness-checker can cycle through
Enrichment and back to Answerer multiple times - this is a real graph
cycle, not just a DAG - bounded by MAX_ENRICHMENT_ITERATIONS so an
ambiguous or genuinely unanswerable question can't loop forever.

Design choice: the loop goes back to Answerer, not back to Retriever.
Retrieval is a vector search over the local document store; searching it
again with the same question would return the same chunks. What actually
changes between iterations is the CONTEXT (new web results get appended),
so the loop only needs to re-run the Answerer and the Completeness-checker
against the enlarged context, not repeat a vector search that would be
identical to the first one.

On an ACCEPTED answer, the web sources the answer actually cited are
persisted back into the vectorstore (see make_accept_node) - the gap
doesn't just get patched for one answer, it closes permanently.

Two structural additions worth calling out:

**Retrieval can now legitimately return nothing**, because the retriever
gates on relevance instead of always handing back its k nearest
neighbours. When it does, the graph skips the first Answerer and critic
round entirely and goes straight to enrichment. Asking a model to answer
from an empty context, then asking a second model whether that answer was
supported, is two LLM calls to discover something already known for
certain before either call was made.

**Every terminal node resolves citations**, not just `accept`. A capped
or best-effort answer still cites sources, and a caller who is being
handed a flagged answer needs its provenance more than usual, not less.
"""
from typing import TypedDict, List, Optional

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from src import rag
from src import completeness
from src import websearch
from src import tracing
from src.config import config
from src.llm import MissingAPIKeyError
from src.retrieval import HybridRetriever
from src.schemas import RAGResponse, CompletenessVerdict, Citation


class GraphState(TypedDict):
    question: str
    # Clarification turns already supplied, oldest first. Each is
    # {"question_asked": ..., "user_answer": ...}. Carried INTO a run
    # rather than accumulated during one: a clarification can only come
    # from outside the graph, so a run either has them at the start or
    # ends by asking for one.
    clarifications: List[dict]
    # Set when the run ends by asking the user something. This is what a
    # caller needs in order to resume.
    clarification_request: Optional[str]
    retrieved_docs: List[Document]
    web_sources: List[dict]          # deduped search results, accumulated across loops
    source_refs: List[rag.SourceRef]  # documents + web results in one numbered sequence
    full_context: str                # rendered from source_refs, rebuilt each loop
    structured_answer: Optional[RAGResponse]
    citations: List[Citation]
    sources: List[str]
    completeness_verdict: Optional[CompletenessVerdict]
    iteration_count: int
    enrichment_log: List[dict]
    searched_queries: List[str]
    persisted_web_sources: int
    retrieval_debug: dict
    final_status: str  # "accepted" | "needs_clarification" | "needs_enrichment_capped" | "error"
    error: Optional[str]


def initial_state(question: str, clarifications: list = None) -> dict:
    """
    The starting state, in one place.

    This dict used to be written out by hand in api.py, scripts/ask_graph.py
    and eval/run_eval.py. Three copies of a fifteen-key literal is a
    standing trap: adding a state field means remembering all three, and
    LangGraph won't complain about the one that was missed - the node
    that reads it just gets a KeyError at runtime, on whichever entry
    point was forgotten.
    """
    return {
        "question": question,
        "clarifications": list(clarifications or []),
        "clarification_request": None,
        "retrieved_docs": [],
        "web_sources": [],
        "source_refs": [],
        "full_context": "",
        "structured_answer": None,
        "citations": [],
        "sources": [],
        "completeness_verdict": None,
        "iteration_count": 0,
        "enrichment_log": [],
        "searched_queries": [],
        "persisted_web_sources": 0,
        "retrieval_debug": {},
        "final_status": "",
        "error": None,
    }


def _rebuild_context(state: GraphState, web_sources: list) -> dict:
    """Renumber documents + web results and re-render the context."""
    refs = rag.build_source_refs(state["retrieved_docs"], web_sources)
    return {"source_refs": refs, "full_context": rag.format_context(refs)}


def _citations_for(state: GraphState) -> dict:
    """Resolve the answer's cited ids against the current source list."""
    answer = state.get("structured_answer")
    if not answer:
        return {"citations": [], "sources": []}
    citations = rag.resolve_citations(state.get("source_refs", []), answer.used_source_ids)
    return {
        "citations": citations,
        "sources": [c.url or c.label for c in citations],
    }


# --- Nodes ---

def make_retrieve_node(retriever: HybridRetriever):
    def retrieve_node(state: GraphState) -> dict:
        # Retrieve on the clarified query, not the raw question. On a
        # first pass these are identical; after a clarification the
        # user's own disambiguating words are what make retrieval work
        # at all.
        docs = retriever.retrieve(
            rag.retrieval_query(state["question"], state.get("clarifications"))
        )
        refs = rag.build_source_refs(docs)
        return {
            "retrieved_docs": docs,
            "source_refs": refs,
            "full_context": rag.format_context(refs),
            "retrieval_debug": dict(retriever.last_debug),
        }

    return retrieve_node


def answer_node(state: GraphState) -> dict:
    try:
        answer = rag.generate_answer(
            state["full_context"],
            state["question"],
            clarifications=state.get("clarifications"),
        )
    except (MissingAPIKeyError, RuntimeError) as e:
        return {"error": str(e), "final_status": "error"}
    return {"structured_answer": answer}


def completeness_node(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    answer = state["structured_answer"]
    try:
        verdict = completeness.check_completeness(
            state["question"],
            state["full_context"],
            answer.answer,
            enrichment_log=state.get("enrichment_log", []),
            clarifications=state.get("clarifications"),
        )
    except (MissingAPIKeyError, RuntimeError) as e:
        return {"error": str(e), "final_status": "error"}
    return {"completeness_verdict": verdict}


def enrichment_node(state: GraphState) -> dict:
    """
    Searches the web for the critic's gap_description, appends any new
    results to the context, and increments iteration_count.

    A failed search degrades gracefully - it still increments the
    iteration count (so the cap still protects against infinite retrying
    on a broken search) rather than crashing the graph.

    The verdict can legitimately be absent here: when retrieval found
    nothing relevant, the graph routes straight to enrichment without a
    critic round, so there is no gap_description yet and the question
    itself is the only available query.
    """
    verdict = state.get("completeness_verdict")
    question = state["question"]
    query = (verdict.gap_description if verdict else None) or question
    iteration = state.get("iteration_count", 0) + 1
    searched = state.get("searched_queries", [])

    log_entry = {"iteration": iteration, "query": query}

    # The critic is told not to repeat a query, but it's a model and may
    # do it anyway. Re-running a byte-identical search spends an
    # iteration to receive the results already in the context, so widen
    # it with the original question instead - different query, and the
    # question's own terms are the ones most likely to be missing from a
    # narrow gap description.
    if _normalize(query) in {_normalize(q) for q in searched}:
        query = f"{question} {query}"
        log_entry["query"] = query
        log_entry["broadened"] = True

    try:
        results = websearch.web_search(query, max_results=config.WEB_SEARCH_RESULTS)
        new_results = websearch.dedupe_results(state.get("web_sources", []), results)
        log_entry["num_results"] = len(results)
        log_entry["num_new"] = len(new_results)
    except Exception as e:
        log_entry["error"] = str(e)
        log_entry["num_results"] = 0
        new_results = []

    tracing.log_tool_span(
        "web_search", input_data={"query": query}, output_data={"results": new_results}
    )

    web_sources = state.get("web_sources", []) + new_results
    update = {
        "web_sources": web_sources,
        "iteration_count": iteration,
        "enrichment_log": state.get("enrichment_log", []) + [log_entry],
        "searched_queries": searched + [query],
    }
    update.update(_rebuild_context(state, web_sources))
    return update


def make_accept_node(vectorstore, persist_enrichment: bool = True):
    """
    Factory (like make_retrieve_node) because persisting enrichment
    findings needs the vectorstore. persist_enrichment defaults to True
    for normal interactive/production use, but eval/run_eval.py passes
    False - repeatedly running the benchmark against the same persist
    directory would otherwise accumulate auto-ingested web content and
    silently change retrieval results between runs, making eval numbers
    non-reproducible.
    """
    def accept_node(state: GraphState) -> dict:
        resolved = _citations_for(state)
        persisted = 0
        if persist_enrichment and state.get("web_sources"):
            citations = resolved["citations"]
            cited_urls = {c.url for c in citations if c.kind == "web" and c.url}
            # An answer that cites no document isn't grounded in the
            # user's corpus at all, so writing its web sources into that
            # corpus is how the corpus stops being theirs. See
            # PERSIST_REQUIRES_DOC_ANCHOR for the incident that put this
            # guard here.
            anchored = any(c.kind == "document" for c in citations)
            if anchored or not config.PERSIST_REQUIRES_DOC_ANCHOR:
                persisted = websearch.persist_web_sources(
                    vectorstore,
                    state["web_sources"],
                    state["question"],
                    cited_urls=cited_urls,
                )
        return {
            "final_status": "accepted",
            "persisted_web_sources": persisted,
            **resolved,
        }

    return accept_node


def clarification_node(state: GraphState) -> dict:
    """
    Terminal, but resumable: it publishes the question to put to the
    user in `clarification_request`, which is the whole input a caller
    needs to start the next turn.

    This used to be a dead end - it reported that the question was
    ambiguous and stopped, so the most interesting routing decision the
    critic could make produced the least useful outcome.
    """
    verdict = state.get("completeness_verdict")
    request = (verdict.gap_description if verdict else None) or (
        "Could you be more specific about what you're asking?"
    )
    return {
        "final_status": "needs_clarification",
        "clarification_request": request,
        **_citations_for(state),
    }


def max_iterations_node(state: GraphState) -> dict:
    """
    Reached when the critic keeps saying "enrich" past the iteration cap.
    Returns the best-effort answer gathered so far rather than nothing -
    an honest "capped out, here's my best shot" is more useful than a
    dead end.
    """
    return {"final_status": "needs_enrichment_capped", **_citations_for(state)}


def error_node(state: GraphState) -> dict:
    return {"final_status": "error"}


# --- Routing ---

def route_after_retrieval(state: GraphState) -> str:
    """
    Skip straight to enrichment when retrieval came back empty.

    This is only reachable because the retriever gates on relevance; an
    ungated vector search always returns k chunks. Answering from a
    knowingly-empty context and then grading that answer costs two LLM
    calls to establish a fact already known with certainty.
    """
    if not state.get("retrieved_docs"):
        return "enrich"
    return "answer"


def route_after_completeness(state: GraphState) -> str:
    """
    Pure function of state -> next node name. The iteration cap is
    enforced HERE, at the decision point, rather than inside the
    enrichment node itself - the cap governs whether to loop at all, so
    it belongs in the routing logic, not buried in a node that runs
    after the decision to loop has already been made.
    """
    if state.get("error"):
        return "error"

    verdict = state.get("completeness_verdict")
    if verdict is None:
        return "error"

    if verdict.action == "accept":
        return "accept"

    if verdict.action == "ask_user":
        # Same reasoning as the enrichment cap, one level up: a critic
        # that can ask for a clarification can ask forever, and "still
        # ambiguous" is not a reason to keep a user answering questions.
        # Past the cap, answer as best we can and flag it rather than
        # bouncing the question back a third time.
        if len(state.get("clarifications") or []) >= config.MAX_CLARIFICATIONS:
            return "max_iterations"
        return "ask_user"

    # action == "enrich"
    if state.get("iteration_count", 0) >= config.MAX_ENRICHMENT_ITERATIONS:
        return "max_iterations"
    return "enrich"


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


# --- Graph assembly ---

def build_graph(vectorstore, persist_enrichment: bool = True, retriever=None):
    """
    `retriever` is injectable so a caller can override retrieval settings
    (the retrieval benchmark compares a dense-only against a hybrid
    retriever over one graph build). It defaults to a HybridRetriever
    built once per graph, which is what keeps the BM25 index from being
    rebuilt on every question.
    """
    retriever = retriever or HybridRetriever(vectorstore)
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("answer", answer_node)
    graph.add_node("check_completeness", completeness_node)
    graph.add_node("accept", make_accept_node(vectorstore, persist_enrichment=persist_enrichment))
    graph.add_node("enrich_step", enrichment_node)
    graph.add_node("clarification_needed", clarification_node)
    graph.add_node("max_iterations_reached", max_iterations_node)
    graph.add_node("handle_error", error_node)
    # "handle_error" rather than "error": LangGraph rejects a node name
    # that collides with a state field name (the state has an "error"
    # field) - this failed loudly at build time the first time it was
    # tried, which is exactly what test_graph_compiles_* exists to catch.

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"answer": "answer", "enrich": "enrich_step"},
    )
    graph.add_edge("answer", "check_completeness")

    graph.add_conditional_edges(
        "check_completeness",
        route_after_completeness,
        {
            "accept": "accept",
            "enrich": "enrich_step",
            "ask_user": "clarification_needed",
            "max_iterations": "max_iterations_reached",
            "error": "handle_error",
        },
    )

    graph.add_edge("enrich_step", "answer")  # the loop

    graph.add_edge("accept", END)
    graph.add_edge("clarification_needed", END)
    graph.add_edge("max_iterations_reached", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


def run_question(app, question: str, clarifications: list = None) -> dict:
    """
    Invoke the compiled graph for one question, inside one trace.

    `clarifications` resumes an earlier run that ended in
    `needs_clarification`. Resuming is a fresh graph run carrying the
    clarification history, not a restored checkpoint, because a
    clarification changes which documents are relevant and therefore
    invalidates everything the previous run computed. Measured on the
    sample corpus: the ambiguous "What is the price?" retrieves nothing
    at all, so there is no retrieval state worth restoring - a
    checkpointer would faithfully reinstate an empty result. Re-running
    is both simpler and more correct here, and it keeps the service
    stateless enough to run behind more than one replica.

    The trace wrapper is here rather than at each call site so that every
    entry point - HTTP, CLI, eval - produces the same trace shape. It
    also fixes what the README listed as a known limitation: each LLM
    call used to become its own top-level trace, because nothing
    established a parent span, so a single question appeared in Langfuse
    as three unrelated traces with no way to tell which answer went with
    which critique.
    """
    with tracing.question_trace(question):
        return app.invoke(initial_state(question, clarifications))
