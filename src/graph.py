"""
Phase 4: the enrichment loop is now real.

Retriever runs once. Answerer -> Completeness-checker can now cycle
through Enrichment and back to Answerer multiple times - this is a real
graph cycle, not just a DAG - bounded by MAX_ENRICHMENT_ITERATIONS so an
ambiguous or genuinely unanswerable question can't loop forever.

Design choice: the loop goes back to Answerer, not back to Retriever.
Retrieval is a vector search over the local document store; searching it
again with the same question would return the same chunks. What actually
changes between iterations is the CONTEXT (new web results get appended),
so the loop only needs to re-run the Answerer and the Completeness-checker
against the enlarged context, not repeat a vector search that would be
identical to the first one.

Phase 7 addition: on an ACCEPTED answer, any web sources that contributed
to it are persisted back into the vectorstore (see make_accept_node) -
the gap doesn't just get patched for one answer, it closes permanently.
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
from src.schemas import RAGResponse, CompletenessVerdict


class GraphState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    doc_context: str          # context from the vectorstore only, set once
    web_sources: List[dict]   # raw search results, accumulated across loops
    full_context: str         # doc_context + formatted web_sources, rebuilt each loop
    structured_answer: Optional[RAGResponse]
    sources: List[str]
    completeness_verdict: Optional[CompletenessVerdict]
    iteration_count: int
    enrichment_log: List[dict]
    persisted_web_sources: int
    final_status: str  # "accepted" | "needs_clarification" | "needs_enrichment_capped" | "error"
    error: Optional[str]


# --- Nodes ---

def make_retrieve_node(vectorstore):
    def retrieve_node(state: GraphState) -> dict:
        docs = rag.retrieve(vectorstore, state["question"])
        doc_context = rag.format_context(docs)
        return {
            "retrieved_docs": docs,
            "doc_context": doc_context,
            "full_context": doc_context,
            "web_sources": [],
            "iteration_count": 0,
            "enrichment_log": [],
            "persisted_web_sources": 0,
        }

    return retrieve_node


def answer_node(state: GraphState) -> dict:
    try:
        answer = rag.generate_answer(state["full_context"], state["question"])
    except (MissingAPIKeyError, RuntimeError) as e:
        return {"error": str(e), "final_status": "error"}
    return {"structured_answer": answer}


def completeness_node(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    answer = state["structured_answer"]
    try:
        verdict = completeness.check_completeness(
            state["question"], state["full_context"], answer.answer
        )
    except (MissingAPIKeyError, RuntimeError) as e:
        return {"error": str(e), "final_status": "error"}
    return {"completeness_verdict": verdict}


def enrichment_node(state: GraphState) -> dict:
    """
    The real enrichment step. Searches the web using the critic's
    gap_description (falling back to the raw question if none was given),
    appends the results to the context, and increments iteration_count.
    A failed search degrades gracefully - it still increments the
    iteration count (so the cap still protects against infinite retrying
    on a broken search) rather than crashing the graph.
    """
    verdict = state["completeness_verdict"]
    query = verdict.gap_description or state["question"]
    iteration = state.get("iteration_count", 0) + 1

    log_entry = {"iteration": iteration, "query": query}

    try:
        results = websearch.web_search(query, max_results=config.WEB_SEARCH_RESULTS)
        log_entry["num_results"] = len(results)
    except Exception as e:
        log_entry["error"] = str(e)
        results = []

    tracing.log_tool_span(
        "web_search", input_data={"query": query}, output_data={"results": results}
    )

    web_sources = state.get("web_sources", []) + results
    num_doc_sources = len(state["retrieved_docs"])
    web_block = websearch.format_web_results(web_sources, start_index=num_doc_sources + 1)
    full_context = state["doc_context"] + ("\n\n" + web_block if web_block else "")

    return {
        "web_sources": web_sources,
        "full_context": full_context,
        "iteration_count": iteration,
        "enrichment_log": state.get("enrichment_log", []) + [log_entry],
    }


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
        sources = rag.resolve_sources(
            state["retrieved_docs"], state["structured_answer"].used_source_ids
        )
        persisted = 0
        if persist_enrichment and state.get("web_sources"):
            persisted = websearch.persist_web_sources(
                vectorstore, state["web_sources"], state["question"]
            )
        return {"final_status": "accepted", "sources": sources, "persisted_web_sources": persisted}

    return accept_node


def clarification_node(state: GraphState) -> dict:
    return {"final_status": "needs_clarification"}


def max_iterations_node(state: GraphState) -> dict:
    """
    Reached when the critic keeps saying "enrich" past the iteration cap.
    Returns the best-effort answer gathered so far rather than nothing -
    an honest "capped out, here's my best shot" is more useful than a
    dead end.
    """
    sources = []
    if state.get("structured_answer"):
        sources = rag.resolve_sources(
            state["retrieved_docs"], state["structured_answer"].used_source_ids
        )
    return {"final_status": "needs_enrichment_capped", "sources": sources}


def error_node(state: GraphState) -> dict:
    return {"final_status": "error"}


# --- Routing ---

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
        return "ask_user"

    # action == "enrich"
    if state.get("iteration_count", 0) >= config.MAX_ENRICHMENT_ITERATIONS:
        return "max_iterations"
    return "enrich"


# --- Graph assembly ---

def build_graph(vectorstore, persist_enrichment: bool = True):
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", make_retrieve_node(vectorstore))
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
    graph.add_edge("retrieve", "answer")
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
