"""
Usage:
    python scripts/ask.py "What is the refund policy?"
    python scripts/ask.py "What is the refund policy?" --show-context

A confidence below LOW_CONFIDENCE_THRESHOLD is flagged clearly - this is
the manual version of what Phase 3's completeness-checker node will do
automatically (and route back through the enrichment loop for).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import load_vectorstore
from src.rag import answer_question

LOW_CONFIDENCE_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser(description="Ask Lacuna a question.")
    parser.add_argument("question", help="Your question, in quotes")
    parser.add_argument("--persist", default="./chroma_db", help="Vector store directory")
    parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument(
        "--show-context", action="store_true", help="Print the retrieved chunks too"
    )
    args = parser.parse_args()

    vectorstore = load_vectorstore(args.persist)
    result = answer_question(vectorstore, args.question, k=args.k)

    if args.show_context:
        print("=" * 60)
        print("RETRIEVED CHUNKS")
        print("=" * 60)
        for i, doc in enumerate(result["retrieved_docs"], start=1):
            source = doc.metadata.get("source", "unknown")
            print(f"\n--- [Source {i}] ({source}) ---")
            print(doc.page_content)
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

    if result["sources"]:
        print("\nSources used:")
        for s in result["sources"]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
