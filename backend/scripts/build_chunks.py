"""Load a corpus, chunk it at each configured size and save to data/processed/.

Usage (from backend/): python -m scripts.build_chunks [corpus_name]
Writes: data/processed/{corpus}/chunks_{size}.json
"""

import json
import random
import statistics
import sys

from app.core.config import get_settings
from app.rag.ingestion.chunker import chunk_documents
from app.rag.ingestion.loaders import load_directory


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else "fastapi"
    settings = get_settings()
    docs = load_directory(settings.raw_dir / corpus)
    out_dir = settings.processed_dir / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(docs)} documents from {corpus}\n")
    print(f"{'size':>6} {'chunks':>7} {'avg':>6} {'median':>7} {'max':>5} {'<100 chars':>11}")
    for size in settings.chunk_sizes:
        chunks = chunk_documents(docs, size, settings.chunk_overlap)
        lengths = [len(c.text) for c in chunks]
        path = out_dir / f"chunks_{size}.json"
        path.write_text(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
        print(
            f"{size:>6} {len(chunks):>7} {statistics.mean(lengths):>6.0f} {statistics.median(lengths):>7.0f}"
            f" {max(lengths):>5} {sum(n < 100 for n in lengths):>11}"
        )

    sample = random.Random(0).choice(chunks)
    print(f"\nSample chunk (size {size}) #{sample.id}: {sample.source} › {sample.heading}\n{sample.text}")


if __name__ == "__main__":
    main()
