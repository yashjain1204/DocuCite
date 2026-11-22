from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.schemas.document import Chunk, RetrievedChunk


class VectorStore(ABC):
    """Stores chunk vectors and finds the ones closest to a query vector."""

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "VectorStore": ...

    @abstractmethod
    def __len__(self) -> int: ...
