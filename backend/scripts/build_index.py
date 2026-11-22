"""Build FAISS indexes for a corpus.

Usage (from backend/):
    python -m scripts.build_index                 # all configured chunk sizes
    python -m scripts.build_index --size 800
    python -m scripts.build_index --corpus fastapi --size 300
Writes: storage/indexes/{corpus}_{size}/{index.faiss, chunks.json, meta.json}
"""

import argparse

from app.core.config import get_settings
from app.services.index_service import build_corpus_index, index_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="fastapi")
    parser.add_argument("--size", type=int, help="chunk size; default: all configured sizes")
    args = parser.parse_args()

    sizes = [args.size] if args.size else get_settings().chunk_sizes
    for size in sizes:
        print(f"\nBuilding {args.corpus} @ chunk size {size}")
        store = build_corpus_index(args.corpus, size)
        print(f"  {len(store)} vectors × {store.index.d} dims in {store.meta['embed_seconds']}s → {index_dir(args.corpus, size)}")


if __name__ == "__main__":
    main()
