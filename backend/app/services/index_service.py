"""Build and load vector indexes: load → clean → chunk → embed → store."""

import time
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.rag.embeddings.encoder import Encoder
from app.rag.ingestion.chunker import chunk_documents
from app.rag.ingestion.loaders import load_directory
from app.rag.retrieval.retriever import Retriever
from app.rag.vectorstore.faiss_store import FaissStore


@lru_cache
def get_encoder(model_name: str | None = None) -> Encoder:
    return Encoder(model_name or get_settings().embed_model)


def index_dir(corpus: str, chunk_size: int) -> Path:
    return get_settings().index_dir / f"{corpus}_{chunk_size}"


def build_corpus_index(corpus: str, chunk_size: int, encoder: Encoder | None = None) -> FaissStore:
    settings = get_settings()
    encoder = encoder or get_encoder()

    docs = load_directory(settings.raw_dir / corpus)
    chunks = chunk_documents(docs, chunk_size, settings.chunk_overlap)

    start = time.perf_counter()
    vectors = encoder.encode([c.text for c in chunks], show_progress=True)
    seconds = round(time.perf_counter() - start, 1)

    store = FaissStore(
        encoder.dim,
        meta={"corpus": corpus, "chunk_size": chunk_size, "embed_model": encoder.model_name, "embed_seconds": seconds},
    )
    store.add(chunks, vectors)
    store.save(index_dir(corpus, chunk_size))
    return store


def load_retriever(corpus: str, chunk_size: int) -> Retriever:
    store = FaissStore.load(index_dir(corpus, chunk_size))
    encoder = get_encoder(store.meta["embed_model"])  # always query with the model that built the index
    return Retriever(encoder, store)
