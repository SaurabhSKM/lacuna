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

from src import tracing
from src.config import config
from src.ingest import load_vectorstore
from src.graph import build_graph, run_question
from eval.benchmark import BENCHMARK


def run_single(app, question: str) -> dict:
    return run_question(app, question)


def score(benchmark: list, results: list) -> dict:
    """
    Pure scoring logic: no LLM calls, no I/O. `results` is a list of
    already-computed graph output dicts (or, in tests, hand-built stand-ins
    with the same shape), aligned index-for-index with `benchmark`.
    """
    correct = 0
    false_negatives = []  # unanswerable questions that still ended up "accepted"
    errored = []
    rows = []

    unanswerable_total = sum(1 for item in benchmark if item["category"] == "unanswerable")

    for item, result in zip(benchmark, results):
        category = item["category"]
        final_status = result.get("final_status")
        verdict = result.get("completeness_verdict")
        actual_action = verdict.action if verdict else None

        if final_status == "error":
            # An error is never a correct outcome, in any category.
            #
            # This used to score as a success for "unanswerable", because
            # correctness there was defined as `final_status !=
            # "accepted"` - so a crashed pipeline counted as having
            # correctly declined to fabricate. A metric that credits its
            # own failures is worse than no metric: a provider outage
            # mid-run would quietly improve the headline number. Two
            # questions hit this on a real run, which is how it surfaced.
            is_correct = False
            errored.append(item["question"])
        elif category == "answerable":
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
            "iteration_count": result.get("iteration_count", 0),
            "retrieved": len(result.get("retrieved_docs") or []),
            # Kept so a failed run is diagnosable from the report alone,
            # rather than only from whatever was on the terminal.
            "error": result.get("error"),
            "correct": is_correct,
        })

    accuracy = correct / len(benchmark) if benchmark else 0.0
    false_negative_rate = (
        len(false_negatives) / unanswerable_total if unanswerable_total else 0.0
    )

    # Per-category accuracy, because the three categories fail in
    # different ways and a single number hides which one regressed.
    by_category = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(row["correct"])
    for bucket in by_category.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "accuracy": accuracy,
        "false_negative_rate": false_negative_rate,
        "false_negatives": false_negatives,
        # Surfaced rather than buried: a run with a high error count is
        # not a measurement of routing quality, whatever its accuracy
        # happens to say.
        "errors": errored,
        "error_rate": len(errored) / len(benchmark) if benchmark else 0.0,
        "by_category": by_category,
        "enrichment_rounds": sum(row["iteration_count"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Lacuna evaluation benchmark.")
    parser.add_argument("--persist", default=config.DEFAULT_PERSIST_DIR)
    parser.add_argument("--output", default=None, help="Optional path to save the full report as JSON")
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)
    # persist_enrichment=False: running this benchmark repeatedly against
    # the same persist directory should give the same results each time.
    # With persistence on, an earlier run's auto-ingested web content
    # would silently change what later runs retrieve - reproducibility
    # matters more here than the self-improving behavior does.
    app = build_graph(vectorstore, persist_enrichment=False)

    print(
        f"Running {len(BENCHMARK)} benchmark questions "
        f"(provider: {config.LLM_PROVIDER})...\n"
    )
    results = []
    for item in BENCHMARK:
        print(f"  - [{item['category']}] {item['question']}")
        results.append(run_single(app, item["question"]))
    tracing.flush()

    report = score(BENCHMARK, results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Overall routing accuracy: {report['accuracy']:.1%}")
    print(f"False-negative rate (unanswerable confidently accepted): {report['false_negative_rate']:.1%}")
    print(f"Total enrichment rounds spent: {report['enrichment_rounds']}")
    if report["errors"]:
        print(
            f"\n⚠ {len(report['errors'])} question(s) ended in a pipeline ERROR "
            f"({report['error_rate']:.1%}). These count as incorrect, and a run "
            f"with errors is not a clean measurement:"
        )
        for row in report["rows"]:
            if row["final_status"] == "error":
                print(f"  - {row['question']}")
                print(f"      {row.get('error') or '(no error message recorded)'}")
    print()
    for category, bucket in sorted(report["by_category"].items()):
        print(f"  {category:14s} {bucket['correct']}/{bucket['total']} ({bucket['accuracy']:.0%})")

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
