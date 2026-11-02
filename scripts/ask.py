"""
Usage:
    python scripts/ask.py "What is the refund policy?"
    python scripts/ask.py "What is the refund policy?" --show-context

The single-shot path: retrieve, answer, print. No completeness check and
no enrichment - use scripts/ask_graph.py for the full pipeline. This one
stays around as the baseline to compare the graph against, which is the
only way to tell whether the extra LLM calls are earning their cost.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tracing
from src.config import config
from src.ingest import load_vectorstore
from src.rag import answer_question

LOW_CONFIDENCE_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser(description="Ask Lacuna a question (single-shot path).")
    parser.add_argument("question", help="Your question, in quotes")
    parser.add_argument("--persist", default=config.DEFAULT_PERSIST_DIR, help="Vector store directory")
    parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument(
        "--show-context", action="store_true", help="Print the retrieved chunks too"
    )
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)
    with tracing.question_trace(args.question):
        result = answer_question(vectorstore, args.question, k=args.k)
    tracing.flush()

    if args.show_context:
        print("=" * 60)
        print("RETRIEVED CHUNKS")
        print("=" * 60)
        if not result["source_refs"]:
            print("\n(nothing passed the relevance threshold)")
        for ref in result["source_refs"]:
            print(f"\n--- [Source {ref.source_id}] ({ref.kind}: {ref.label}) ---")
            print(ref.text)
        print()

    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    if result["error"]:
        print(f"[Error: {result['error']}]")
        print("\n(Retrieval worked fine above/via --show-context - this "
              "failure is in the generation step.)")
        return

    r = result["structured"]

    if r.confidence < LOW_CONFIDENCE_THRESHOLD:
        print(f"⚠ LOW CONFIDENCE ({r.confidence:.2f}) — treat this answer with caution\n")
    else:
        print(f"Confidence: {r.confidence:.2f}\n")

    print("Answer:")
    print(r.answer)

    if r.missing_info:
        print("\nMissing info (would improve this answer):")
        for item in r.missing_info:
            print(f"  - {item}")

    if result["citations"]:
        print("\nSources used:")
        for c in result["citations"]:
            location = f" ({c.url})" if c.url else ""
            print(f"  - [{c.kind}] {c.label}{location}")


if __name__ == "__main__":
    main()
