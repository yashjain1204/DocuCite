"""Exact cosine-similarity search with FAISS IndexFlatIP.

FAISS only stores vectors; chunk text and metadata are kept alongside in chunks.json,
in the same order, so FAISS row i is chunks[i].
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.rag.vectorstore.base import VectorStore
from app.schemas.document import Chunk, RetrievedChunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


class FaissStore(VectorStore):
    def __init__(self, dim: int, meta: dict | None = None):
        self.index = faiss.IndexFlatIP(dim)  # inner product on unit vectors == cosine
        self.chunks: list[Chunk] = []
        self.meta = meta or {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.chunks.extend(chunks)

    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        query = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        scores, rows = self.index.search(query, min(k, len(self)))
        return [
            RetrievedChunk(**self.chunks[row].model_dump(), score=float(score))
            for score, row in zip(scores[0], rows[0])
            if row != -1
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        (directory / CHUNKS_FILE).write_text(
            json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        meta = {**self.meta, "dim": self.index.d, "count": len(self)}
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "FaissStore":
        if not (directory / INDEX_FILE).exists():
            raise FileNotFoundError(f"No index in {directory}. Build it with: python -m scripts.build_index")
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        store = cls(meta["dim"], meta)
        store.index = faiss.read_index(str(directory / INDEX_FILE))
        raw = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        store.chunks = [Chunk(**c) for c in raw]
        return store

    def __len__(self) -> int:
        return self.index.ntotal
