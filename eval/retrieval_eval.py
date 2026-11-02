"""
Usage:
    python eval/retrieval_eval.py --persist ./chroma_db
    python eval/retrieval_eval.py --persist ./chroma_db --output retrieval.json

Scores retrieval against eval/retrieval_benchmark.py, and compares
retriever configurations head to head so the effect of each piece is
visible rather than asserted:

    dense              - the original behaviour: vector search, no gate
    dense + gate       - vector search with the relevance floor
    hybrid + gate      - vector + BM25 fused with RRF, with the floor

Needs no API key and no network beyond the one-time embedding model
download, because the gold label is a filename and the embedding model is
local. That is the point: it gives the project real measured numbers,
whereas the end-to-end benchmark in run_eval.py cannot run at all without
a configured provider.

score_retrieval() is pure (candidates in, metrics out), so the metric
definitions are unit-tested without touching a vector store - see
tests/test_retrieval.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.ingest import load_vectorstore
from src.retrieval import HybridRetriever
from eval.retrieval_benchmark import RETRIEVAL_BENCHMARK, RETRIEVAL_DISTRACTORS

CONFIGURATIONS = [
    ("dense", {"hybrid": False, "min_relevance": 0.0, "rerank": False}),
    ("dense + gate", {"hybrid": False, "min_relevance": None, "rerank": False}),
    ("hybrid + gate", {"hybrid": True, "min_relevance": None, "rerank": False}),
    ("hybrid + rerank", {"hybrid": True, "min_relevance": None, "rerank": True}),
]


def _filename(doc) -> str:
    raw = (doc.metadata or {}).get("source", "")
    return raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def score_retrieval(gold_sets: list, retrieved_filenames: list, distractor_counts: list) -> dict:
    """
    Pure scoring.

    `gold_sets[i]` is the set of acceptable filenames for question i.
    `retrieved_filenames[i]` is the ranked list of filenames retrieved
    for it (one entry per chunk, so a filename can repeat).
    `distractor_counts[i]` is how many chunks were retrieved for
    distractor question i.

    Metrics:
      hit_rate  - share of questions where at least one retrieved chunk
                  came from a gold document. The headline number: if this
                  misses, no amount of downstream cleverness recovers the
                  answer.
      mrr       - mean reciprocal rank of the first gold chunk. Separates
                  "found it first" from "found it fourth", which hit_rate
                  cannot.
      precision - share of all retrieved chunks that came from a gold
                  document. Low precision means the context is padded
                  with near-misses the model has to ignore.
      distractor_rejection
                - share of off-topic questions that retrieved nothing.
                  An ungated retriever scores 0.0 here by construction.
    """
    hits, reciprocal_ranks, gold_chunks, total_chunks = 0, [], 0, 0

    for gold, filenames in zip(gold_sets, retrieved_filenames):
        total_chunks += len(filenames)
        gold_chunks += sum(1 for name in filenames if name in gold)
        rank = next(
            (i for i, name in enumerate(filenames, start=1) if name in gold), None
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)

    n = len(gold_sets)
    return {
        "questions": n,
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "precision": gold_chunks / total_chunks if total_chunks else 0.0,
        "chunks_retrieved": total_chunks,
        "distractors": len(distractor_counts),
        "distractor_rejection": (
            sum(1 for c in distractor_counts if c == 0) / len(distractor_counts)
            if distractor_counts
            else 0.0
        ),
    }


def evaluate_configuration(
    vectorstore, hybrid, min_relevance, rerank=None, rerank_min_score=None
) -> dict:
    retriever = HybridRetriever(
        vectorstore,
        hybrid=hybrid,
        min_relevance=config.MIN_RELEVANCE if min_relevance is None else min_relevance,
        rerank=rerank,
        rerank_min_score=rerank_min_score,
    )

    retrieved_filenames, misses = [], []
    for item in RETRIEVAL_BENCHMARK:
        docs = retriever.retrieve(item["question"])
        names = [_filename(d) for d in docs]
        retrieved_filenames.append(names)
        if not any(name in item["gold"] for name in names):
            misses.append({
                "question": item["question"],
                "expected": sorted(item["gold"]),
                "retrieved": names,
            })

    distractor_counts, leaks = [], []
    for question in RETRIEVAL_DISTRACTORS:
        docs = retriever.retrieve(question)
        distractor_counts.append(len(docs))
        if docs:
            leaks.append({"question": question, "retrieved": [_filename(d) for d in docs]})

    report = score_retrieval(
        [item["gold"] for item in RETRIEVAL_BENCHMARK], retrieved_filenames, distractor_counts
    )
    report["misses"] = misses
    report["distractor_leaks"] = leaks
    return report


SWEEP_THRESHOLDS = [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


def sweep(vectorstore):
    """
    Walk MIN_RELEVANCE across the benchmark so the default is a measured
    choice rather than a plausible-looking constant. The number that
    matters is where distractor rejection reaches 1.0 while hit@k is
    still 1.0 - past that point the gate starts discarding real answers.
    """
    header = f"{'MIN_RELEVANCE':>14s} {'hit@k':>7s} {'MRR':>7s} {'precision':>10s} {'rejected':>9s}"
    print(header)
    print("-" * len(header))
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        report = evaluate_configuration(vectorstore, hybrid=True, min_relevance=threshold)
        rows.append({"min_relevance": threshold, **{
            k: v for k, v in report.items() if k not in ("misses", "distractor_leaks")
        }})
        print(
            f"{threshold:>14.2f} {report['hit_rate']:>6.1%} {report['mrr']:>7.3f} "
            f"{report['precision']:>10.1%} {report['distractor_rejection']:>8.1%}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Score Lacuna's retriever.")
    parser.add_argument("--persist", default=config.DEFAULT_PERSIST_DIR)
    parser.add_argument("--output", default=None, help="Optional path to save the report as JSON")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep MIN_RELEVANCE instead of comparing retriever configurations",
    )
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)

    if args.sweep:
        rows = sweep(vectorstore)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, default=str)
            print(f"\nSweep saved to {args.output}")
        return

    stored = len(vectorstore.get(include=[])["ids"])
    print(
        f"{len(RETRIEVAL_BENCHMARK)} labelled questions + "
        f"{len(RETRIEVAL_DISTRACTORS)} distractors against {stored} chunks, "
        f"top_k={config.TOP_K}\n"
    )

    reports = {}
    for name, settings in CONFIGURATIONS:
        reports[name] = evaluate_configuration(vectorstore, **settings)

    header = f"{'configuration':16s} {'hit@k':>7s} {'MRR':>7s} {'precision':>10s} {'distractors rejected':>21s}"
    print(header)
    print("-" * len(header))
    for name, report in reports.items():
        print(
            f"{name:16s} {report['hit_rate']:>6.1%} {report['mrr']:>7.3f} "
            f"{report['precision']:>10.1%} {report['distractor_rejection']:>20.1%}"
        )

    worst = reports[CONFIGURATIONS[-1][0]]
    if worst["misses"]:
        print(f"\nMissed by '{CONFIGURATIONS[-1][0]}':")
        for miss in worst["misses"]:
            print(f"  - {miss['question']}")
            print(f"      expected one of {miss['expected']}, got {miss['retrieved']}")
    else:
        print(f"\nNo misses for '{CONFIGURATIONS[-1][0]}'.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
