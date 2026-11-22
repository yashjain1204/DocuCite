"""Load and clean a corpus, then print stats and a sample.

Usage (from backend/): python -m scripts.inspect_corpus [corpus_name] [sample_source]
"""

import sys

from app.core.config import get_settings
from app.rag.ingestion.loaders import load_directory


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else "fastapi"
    sample = sys.argv[2] if len(sys.argv) > 2 else "tutorial/query-params.md"

    docs = load_directory(get_settings().raw_dir / corpus)
    chars = sum(len(d.text) for d in docs)
    print(f"{len(docs)} documents, {chars:,} characters (~{chars // 4:,} tokens)")

    longest = sorted(docs, key=lambda d: len(d.text), reverse=True)[:3]
    print("Longest:", ", ".join(f"{d.source} ({len(d.text):,})" for d in longest))

    doc = next((d for d in docs if d.source == sample), docs[0])
    print(f"\n--- {doc.source} (first 1200 chars) ---\n{doc.text[:1200]}")


if __name__ == "__main__":
    main()
