"""
Usage:
    python eval/run_eval.py --persist ./chroma_db
    python eval/run_eval.py --persist ./chroma_db --output results.json

Runs every question in eval/benchmark.py through the full LangGraph
pipeline and reports:
  - overall routing accuracy: did the critic reach the expected kind of
    outcome for each category?
  - the false-negative rate specifically: how often an "unanswerable"
    question was still confidently "accept"ed. This is the headline
    number, not overall accuracy - it's the one failure mode that
    produces a fabricated answer a user might actually trust.

score() is kept as a pure function (benchmark + already-computed results
in, a report out) specifically so the scoring logic itself is fully
unit-testable without ever calling an LLM - see tests/test_eval.py. Only
run_single() needs live API access.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import load_vectorstore
from src.graph import build_graph
from eval.benchmark import BENCHMARK


def run_single(app, question: str) -> dict:
    initial_state = {
        "question": question,
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
    return app.invoke(initial_state)


def score(benchmark: list, results: list) -> dict:
    """
    Pure scoring logic: no LLM calls, no I/O. `results` is a list of
    already-computed graph output dicts (or, in tests, hand-built stand-ins
    with the same shape), aligned index-for-index with `benchmark`.
    """
    correct = 0
    false_negatives = []  # unanswerable questions that still ended up "accepted"
    rows = []

    unanswerable_total = sum(1 for item in benchmark if item["category"] == "unanswerable")

    for item, result in zip(benchmark, results):
        category = item["category"]
        final_status = result.get("final_status")
        verdict = result.get("completeness_verdict")
        actual_action = verdict.action if verdict else None

        if category == "answerable":
            is_correct = final_status == "accepted"
        elif category == "unanswerable":
            is_correct = final_status != "accepted"
            if not is_correct:
                false_negatives.append(item["question"])
        elif category == "ambiguous":
            is_correct = actual_action == "ask_user"
        else:
            is_correct = False

        correct += int(is_correct)
        rows.append({
            "question": item["question"],
            "category": category,
            "final_status": final_status,
            "critic_action": actual_action,
            "correct": is_correct,
        })

    accuracy = correct / len(benchmark) if benchmark else 0.0
    false_negative_rate = (
        len(false_negatives) / unanswerable_total if unanswerable_total else 0.0
    )

    return {
        "accuracy": accuracy,
        "false_negative_rate": false_negative_rate,
        "false_negatives": false_negatives,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Lacuna evaluation benchmark.")
    parser.add_argument("--persist", default="./chroma_db")
    parser.add_argument("--output", default=None, help="Optional path to save the full report as JSON")
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)
    # persist_enrichment=False: running this benchmark repeatedly against
    # the same persist directory should give the same results each time.
    # With persistence on, an earlier run's auto-ingested web content
    # would silently change what later runs retrieve - reproducibility
    # matters more here than the self-improving behavior does.
    app = build_graph(vectorstore, persist_enrichment=False)

    print(f"Running {len(BENCHMARK)} benchmark questions...\n")
    results = []
    for item in BENCHMARK:
        print(f"  - [{item['category']}] {item['question']}")
        results.append(run_single(app, item["question"]))

    report = score(BENCHMARK, results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Overall routing accuracy: {report['accuracy']:.1%}")
    print(f"False-negative rate (unanswerable confidently accepted): {report['false_negative_rate']:.1%}")

    if report["false_negatives"]:
        print("\n⚠ Questions WRONGLY accepted (hallucination risk):")
        for q in report["false_negatives"]:
            print(f"  - {q}")

    print("\nPer-question breakdown:")
    for row in report["rows"]:
        mark = "✓" if row["correct"] else "✗"
        print(f"  {mark} [{row['category']}] {row['question']} -> {row['final_status']} (critic: {row['critic_action']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
