"""Turn text into normalized embedding vectors with a sentence-transformers model."""

from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer


class Encoder:
    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size

    @cached_property
    def model(self) -> SentenceTransformer:
        # Loaded on first use; downloads to the Hugging Face cache the first time.
        return SentenceTransformer(self.model_name, device="cpu")

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of unit-length vectors."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # so dot product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vectors.astype(np.float32)
