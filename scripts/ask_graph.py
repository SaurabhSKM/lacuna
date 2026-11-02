"""
Usage:
    python scripts/ask_graph.py "What is the refund policy for monthly plans?"
    python scripts/ask_graph.py "Does Aria support quantum encryption?"
    python scripts/ask_graph.py "What is the price?" --show-retrieval

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

from src import tracing
from src.config import config
from src.graph import build_graph, initial_state
from src.ingest import load_vectorstore


def describe_step(node_name: str, delta: dict) -> str:
    if node_name == "retrieve":
        docs = delta.get("retrieved_docs", [])
        debug = delta.get("retrieval_debug", {})
        if not docs:
            return (
                "[Retriever] Nothing passed the relevance threshold "
                f"({debug.get('gated_out', 0)} candidate(s) rejected). "
                "Skipping straight to enrichment."
            )
        return (
            f"[Retriever] Kept {len(docs)} chunk(s) "
            f"(dense: {debug.get('dense', 0)}, lexical: {debug.get('lexical', 0)}, "
            f"rejected by relevance gate: {debug.get('gated_out', 0)})."
        )

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
        note = " (query broadened: the critic repeated itself)" if log.get("broadened") else ""
        return (
            f"[Enrichment] Round {log['iteration']}: searched web for "
            f"'{log['query']}', found {log['num_results']} result(s), "
            f"{log.get('num_new', 0)} new{note}. Re-answering..."
        )

    if node_name == "accept":
        persisted = delta.get("persisted_web_sources", 0)
        if persisted:
            return (
                f"[Orchestrator] Answer accepted. Permanently added {persisted} "
                f"cited web result(s) to the knowledge base - this gap won't "
                f"need re-searching next time."
            )
        return "[Orchestrator] Answer accepted."

    if node_name == "clarification_needed":
        return (
            "[Orchestrator] Question is ambiguous - asking the user: "
            f"{delta.get('clarification_request', '(no specific request)')}"
        )

    if node_name == "max_iterations_reached":
        return "[Orchestrator] Hit the enrichment iteration cap - returning best-effort answer."

    if node_name == "handle_error":
        return "[Orchestrator] Pipeline error - see final output."

    return f"[{node_name}] done."


def main():
    parser = argparse.ArgumentParser(description="Ask Lacuna a question via the LangGraph pipeline.")
    parser.add_argument("question", help="Your question, in quotes")
    parser.add_argument("--persist", default=config.DEFAULT_PERSIST_DIR, help="Vector store directory")
    parser.add_argument(
        "--show-retrieval", action="store_true", help="Print per-chunk retrieval scores"
    )
    parser.add_argument(
        "--clarify",
        action="append",
        default=[],
        metavar="ANSWER",
        help="Pre-supply a clarification answer (repeatable), instead of being prompted",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Don't ask interactively when the question is ambiguous; just report it",
    )
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)
    app = build_graph(vectorstore)

    print("=" * 60)
    print("LIVE PROGRESS")
    print("=" * 60)

    # Answers queued on the command line are consumed first, so a
    # clarification run can be scripted and tested non-interactively.
    queued = list(args.clarify)
    clarifications = []

    while True:
        state = initial_state(args.question, clarifications)
        with tracing.question_trace(args.question):
            for update in app.stream(dict(state), stream_mode="updates"):
                for node_name, delta in update.items():
                    state.update(delta)
                    print(describe_step(node_name, delta))

        if state.get("final_status") != "needs_clarification":
            break

        request = state.get("clarification_request") or "Could you be more specific?"
        if queued:
            reply = queued.pop(0)
            print(f"\n[You] (from --clarify) {reply}")
        elif args.no_prompt or not sys.stdin.isatty():
            break
        else:
            try:
                reply = input(f"\n[Lacuna asks] {request}\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n(no clarification given)")
                break
            if not reply:
                break

        clarifications.append({"question_asked": request, "user_answer": reply})
        print("\n" + "-" * 60)
        print(f"RESUMING with clarification {len(clarifications)}/"
              f"{config.MAX_CLARIFICATIONS} - re-retrieving on the clarified question")
        print("-" * 60)

    tracing.flush()

    if args.show_retrieval and state.get("retrieval_debug", {}).get("scores"):
        print("\nRetrieval scores (dense relevance / lexical IDF coverage / fused RRF):")
        for key, scores in state["retrieval_debug"]["scores"].items():
            dense = scores.get("dense")
            dense_text = f"{dense:.3f}" if isinstance(dense, float) else "n/a"
            print(
                f"  {key[:12]}  dense={dense_text}  "
                f"lexical={scores.get('lexical', 0.0):.3f}  "
                f"fused={scores.get('fused', 0.0):.5f}"
            )

    print("\n" + "=" * 60)
    print(f"FINAL STATUS: {state['final_status']}")
    print("=" * 60)

    if state.get("error"):
        print(f"[Error: {state['error']}]")
        return

    if state.get("clarifications"):
        print("\nClarifications used:")
        for turn in state["clarifications"]:
            print(f"  asked: {turn['question_asked']}")
            print(f"  you  : {turn['user_answer']}")

    if state["final_status"] == "needs_clarification":
        print(f"\nStill needs clarification: {state.get('clarification_request')}")
        print("Re-run with --clarify \"your answer\" to continue.")
        return

    answer = state.get("structured_answer")
    verdict = state.get("completeness_verdict")

    if answer:
        print("\nAnswer:")
        print(answer.answer)
        print(f"\nGenerator's self-reported confidence: {answer.confidence:.2f}")
        if answer.missing_info:
            print("\nStill missing:")
            for item in answer.missing_info:
                print(f"  - {item}")

    if verdict:
        print(
            f"Independent critic's final verdict: {verdict.action} "
            f"(sufficiency: {verdict.sufficiency_score:.2f})"
        )
        if verdict.gap_description:
            print(f"Gap: {verdict.gap_description}")

    if state["iteration_count"] > 0:
        print(
            f"\nEnrichment rounds used: {state['iteration_count']} / "
            f"{config.MAX_ENRICHMENT_ITERATIONS}"
        )

    if state.get("citations"):
        print("\nSources cited:")
        for c in state["citations"]:
            location = f" ({c.url})" if c.url else ""
            print(f"  - [{c.kind}] {c.label}{location}")

    web_uncited = [
        w for w in state.get("web_sources", [])
        if w.get("href") not in {c.url for c in state.get("citations", [])}
    ]
    if web_uncited:
        print("\nWeb results consulted but not cited:")
        for w in web_uncited:
            print(f"  - {w.get('title', 'untitled')} ({w.get('href', '')})")


if __name__ == "__main__":
    main()
