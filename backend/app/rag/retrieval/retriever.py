from app.rag.embeddings.encoder import Encoder
from app.rag.vectorstore.base import VectorStore
from app.schemas.document import RetrievedChunk


class Retriever:
    """Question in, top-k most similar chunks out."""

    def __init__(self, encoder: Encoder, store: VectorStore):
        self.encoder = encoder
        self.store = store

    def retrieve(self, question: str, k: int = 5) -> list[RetrievedChunk]:
        vector = self.encoder.encode([question])[0]
        return self.store.search(vector, k)
