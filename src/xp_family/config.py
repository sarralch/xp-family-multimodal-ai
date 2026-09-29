"""Typed application settings, loaded from environment variables and `.env`.

Every secret and every tunable lives here, so no module reads `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # configs/default.env holds committed, non-secret defaults; .env (git-ignored)
    # holds secrets and local overrides; real environment variables win over both.
    model_config = SettingsConfigDict(
        env_file=("configs/default.env", ".env"),
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # --- LLM -------------------------------------------------------------
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2

    # --- Data & caches ---------------------------------------------------
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")

    # --- Retrieval -------------------------------------------------------
    # Multilingual models: the knowledge base mixes French and English.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    use_reranker: bool = True
    candidates_k: int = Field(20, ge=1, description="Candidates fetched per retriever")
    top_k: int = Field(4, ge=1, description="Chunks passed to the LLM after fusion/rerank")
    rrf_k: int = Field(60, ge=1, description="Reciprocal-rank-fusion damping constant")
    chunk_size: int = Field(800, ge=100, description="Max characters per chunk")
    chunk_overlap: int = Field(120, ge=0)

    # --- OCR -------------------------------------------------------------
    tesseract_cmd: str | None = None  # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe

    # --- Observability ---------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
