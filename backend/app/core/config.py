from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # Paths
    data_dir: Path = REPO_ROOT / "data"
    storage_dir: Path = REPO_ROOT / "storage"
    eval_dir: Path = REPO_ROOT / "eval"

    # Retrieval
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_sizes: list[int] = [300, 800, 1500]
    chunk_overlap: float = 0.12
    top_k: int = 5

    # LLM
    llm_provider: str = "groq"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.storage_dir / "indexes"


@lru_cache
def get_settings() -> Settings:
    return Settings()
