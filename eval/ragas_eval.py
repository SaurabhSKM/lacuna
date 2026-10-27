"""
Optional: RAGAS-based faithfulness and answer-relevancy scoring, on top
of the custom completeness-detection harness in run_eval.py.

Not exercised by the test suite. RAGAS's metrics use an LLM as a judge
internally, so - unlike everything else in this project - there's no
way to verify this offline. Run it once you have a working GROQ_API_KEY
(or other provider) configured.

Usage:
    pip install ragas datasets
    python eval/ragas_eval.py --persist ./chroma_db

Only the "answerable" subset of the benchmark is used here - RAGAS's
faithfulness/relevancy metrics assume a real answer grounded in real
context, not a deliberate refusal, so scoring the "unanswerable" and
"ambiguous" categories with it wouldn't mean anything.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        print("RAGAS isn't installed. Run: pip install ragas datasets")
        return

    from src.ingest import load_vectorstore
    from src.graph import build_graph
    from eval.benchmark import BENCHMARK
    from eval.run_eval import run_single

    parser = argparse.ArgumentParser()
    parser.add_argument("--persist", default="./chroma_db")
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)
    app = build_graph(vectorstore)

    answerable = [item for item in BENCHMARK if item["category"] == "answerable"]

    questions, answers, contexts = [], [], []
    for item in answerable:
        result = run_single(app, item["question"])
        if result.get("structured_answer"):
            questions.append(item["question"])
            answers.append(result["structured_answer"].answer)
            contexts.append([doc.page_content for doc in result["retrieved_docs"]])

    if not questions:
        print("No answerable-category questions produced a usable answer - nothing to score.")
        return

    dataset = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts})
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(scores)


if __name__ == "__main__":
    main()
