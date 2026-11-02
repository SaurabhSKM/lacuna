"""
Optional: RAGAS-based faithfulness and answer-relevancy scoring, on top
of the custom completeness-detection harness in run_eval.py.

Not exercised by the test suite, and the one file here that can't be.
RAGAS's metrics use an LLM as a judge internally, so unlike the
retrieval benchmark there's no offline mode, and unlike the rest of the
project its dependencies aren't installed by default:

    pip install -r requirements-eval.txt
    python eval/ragas_eval.py --persist ./chroma_db

Only the "answerable" subset of the benchmark is used. RAGAS's
faithfulness/relevancy metrics assume a real answer grounded in real
context, not a deliberate refusal, so scoring the "unanswerable" and
"ambiguous" categories with it wouldn't mean anything - a correct refusal
would score as an unfaithful answer.

Note on the dataset shape: RAGAS renamed its columns at 0.2
(question/answer/contexts became user_input/response/retrieved_contexts)
and `evaluate` now takes an EvaluationDataset rather than a HuggingFace
Dataset. This file is written against the current API, with a fallback to
the legacy shape, because the version actually installed here can't be
exercised offline to confirm which one is in play.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_dataset(rows: list):
    """
    Current RAGAS API first, legacy second. Returns whatever `evaluate`
    accepts.
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample

        return EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
            )
            for row in rows
        ])
    except ImportError:
        from datasets import Dataset

        return Dataset.from_dict({
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
        })


def main():
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
    except ImportError:
        print("RAGAS isn't installed. Run: pip install -r requirements-eval.txt")
        return

    from src.config import config
    from src.ingest import load_vectorstore
    from src.graph import build_graph
    from eval.benchmark import BENCHMARK
    from eval.run_eval import run_single

    parser = argparse.ArgumentParser()
    parser.add_argument("--persist", default=config.DEFAULT_PERSIST_DIR)
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)
    app = build_graph(vectorstore, persist_enrichment=False)

    answerable = [item for item in BENCHMARK if item["category"] == "answerable"]

    rows = []
    for item in answerable:
        result = run_single(app, item["question"])
        if result.get("structured_answer") and result.get("retrieved_docs"):
            rows.append({
                "question": item["question"],
                "answer": result["structured_answer"].answer,
                "contexts": [doc.page_content for doc in result["retrieved_docs"]],
            })

    if not rows:
        print("No answerable-category questions produced a usable answer - nothing to score.")
        return

    print(evaluate(_build_dataset(rows), metrics=[faithfulness, answer_relevancy]))


if __name__ == "__main__":
    main()
