"""Ask a question from the terminal.

Usage (from backend/):
    python -m scripts.ask "how do I make a query parameter optional"
    python -m scripts.ask "..." --size 300 --k 5
Currently retrieval-only: prints the top-k chunks. Answer generation is added in Step 4.
"""

import argparse

from app.core.config import get_settings
from app.services.index_service import load_retriever


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--corpus", default="fastapi")
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()

    retriever = load_retriever(args.corpus, args.size)
    for rank, chunk in enumerate(retriever.retrieve(args.question, args.k), start=1):
        preview = " ".join(chunk.text.split())[:160]
        print(f"\n[{rank}] score={chunk.score:.3f}  {chunk.source} › {chunk.heading}\n    {preview}")


if __name__ == "__main__":
    main()
