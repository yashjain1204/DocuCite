from pathlib import Path

import numpy as np
import pytest

from app.rag.vectorstore.faiss_store import FaissStore
from app.schemas.document import Chunk


def unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def store() -> FaissStore:
    s = FaissStore(dim=3, meta={"embed_model": "test"})
    chunks = [Chunk(id=i, text=t, source=f"{t}.md") for i, t in enumerate(["x", "y", "z"])]
    s.add(chunks, np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([0, 0, 1])]))
    return s


def test_search_ranks_by_cosine(store: FaissStore):
    results = store.search(unit([0.9, 0.1, 0]), k=2)
    assert [r.text for r in results] == ["x", "y"]
    assert results[0].score > results[1].score
    assert results[0].score == pytest.approx(0.9939, abs=1e-3)


def test_k_larger_than_store(store: FaissStore):
    assert len(store.search(unit([1, 1, 1]), k=10)) == 3


def test_add_rejects_mismatched_lengths(store: FaissStore):
    with pytest.raises(ValueError):
        store.add([Chunk(id=9, text="a", source="a.md")], np.zeros((2, 3), dtype=np.float32))


def test_save_and_load_roundtrip(store: FaissStore, tmp_path: Path):
    store.save(tmp_path)
    loaded = FaissStore.load(tmp_path)
    assert len(loaded) == 3
    assert loaded.meta["embed_model"] == "test" and loaded.meta["dim"] == 3
    assert loaded.search(unit([0, 0, 1]), k=1)[0].source == "z.md"


def test_load_missing_index_has_helpful_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        FaissStore.load(tmp_path / "nope")
