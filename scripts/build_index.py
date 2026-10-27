"""
Usage:
    python scripts/build_index.py --docs data/sample_docs --persist ./chroma_db
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import ingest


def main():
    parser = argparse.ArgumentParser(description="Build the Lacuna vector index.")
    parser.add_argument("--docs", required=True, help="Directory of source documents")
    parser.add_argument(
        "--persist", default="./chroma_db", help="Where to persist the vector store"
    )
    args = parser.parse_args()

    ingest(args.docs, args.persist)


if __name__ == "__main__":
    main()
