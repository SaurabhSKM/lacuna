"""
Usage:
    python scripts/ask_graph.py "What is the refund policy for monthly plans?"
    python scripts/ask_graph.py "Does Aria support quantum encryption?"

Uses app.stream(..., stream_mode="updates") rather than app.invoke() so
progress prints as each node actually runs, instead of only seeing the
final result after everything finishes. Each node's return value is a
partial state update (LangGraph merges it into the overall state); this
script replicates that same merge itself, which is a correct stand-in
here since every field in GraphState uses default overwrite semantics
(no custom reducers) - so the last write really does win, same as what
LangGraph does internally.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import load_vectorstore
from src.graph import build_graph


def describe_step(node_name: str, delta: dict) -> str:
    if node_name == "retrieve":
        n = len(delta.get("retrieved_docs", []))
        return f"[Retriever] Retrieved {n} chunk(s) from the knowledge base."

    if node_name == "answer":
        if delta.get("error"):
            return f"[Answerer] Failed: {delta['error']}"
        a = delta["structured_answer"]
        return f"[Answerer] Drafted an answer (self-reported confidence: {a.confidence:.2f})."

    if node_name == "check_completeness":
        if delta.get("error"):
            return f"[Completeness-checker] Failed: {delta['error']}"
        v = delta["completeness_verdict"]
        line = f"[Completeness-checker] Verdict: {v.action} (sufficiency: {v.sufficiency_score:.2f})"
        if v.gap_description:
            line += f" — gap: {v.gap_description}"
        return line

    if node_name == "enrich_step":
        log = delta["enrichment_log"][-1]
        if "error" in log:
            return f"[Enrichment] Round {log['iteration']}: search failed ({log['error']})."
        return (
            f"[Enrichment] Round {log['iteration']}: searched web for "
            f"'{log['query']}', found {log['num_results']} result(s). Re-answering..."
        )

    if node_name == "accept":
        persisted = delta.get("persisted_web_sources", 0)
        if persisted:
            return (
                f"[Orchestrator] Answer accepted. Permanently added {persisted} "
                f"web result(s) to the knowledge base - this gap won't need "
                f"re-searching next time."
            )
        return "[Orchestrator] Answer accepted."

    if node_name == "clarification_needed":
        return "[Orchestrator] Question is ambiguous - needs clarification from the user."

    if node_name == "max_iterations_reached":
        return "[Orchestrator] Hit the enrichment iteration cap - returning best-effort answer."

    if node_name == "handle_error":
        return "[Orchestrator] Pipeline error - see final output."

    return f"[{node_name}] done."


def main():
    parser = argparse.ArgumentParser(description="Ask Lacuna a question via the LangGraph pipeline.")
    parser.add_argument("question", help="Your question, in quotes")
    parser.add_argument("--persist", default="./chroma_db", help="Vector store directory")
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)
    app = build_graph(vectorstore)

    initial_state = {
        "question": args.question,
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

    print("=" * 60)
    print("LIVE PROGRESS")
    print("=" * 60)

    state = dict(initial_state)
    for update in app.stream(initial_state, stream_mode="updates"):
        for node_name, delta in update.items():
            state.update(delta)
            print(describe_step(node_name, delta))

    print("\n" + "=" * 60)
    print(f"FINAL STATUS: {state['final_status']}")
    print("=" * 60)

    if state.get("error"):
        print(f"[Error: {state['error']}]")
        return

    answer = state["structured_answer"]
    verdict = state["completeness_verdict"]

    print("\nAnswer:")
    print(answer.answer)
    print(f"\nGenerator's self-reported confidence: {answer.confidence:.2f}")
    if verdict:
        print(f"Independent critic's final verdict: {verdict.action} (sufficiency: {verdict.sufficiency_score:.2f})")

    if state["iteration_count"] > 0:
        print(f"\nEnrichment rounds used: {state['iteration_count']} / {app_max_note()}")

    if state.get("sources"):
        print("\nDocument sources used:")
        for s in state["sources"]:
            print(f"  - {s}")

    if state.get("web_sources"):
        print("\nWeb sources consulted:")
        for w in state["web_sources"]:
            print(f"  - {w.get('title', 'untitled')} ({w.get('href', '')})")


def app_max_note():
    from src.config import config
    return config.MAX_ENRICHMENT_ITERATIONS


if __name__ == "__main__":
    main()
