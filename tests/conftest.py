from __future__ import annotations

from pathlib import Path

import pytest

from xp_family.config import Settings
from xp_family.llm import EchoLLM
from xp_family.rag.embeddings import HashingEmbedder
from xp_family.services import Services

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=SAMPLE_DATA,
        cache_dir=tmp_path / "cache",
        gemini_api_key=None,
        use_reranker=False,
        top_k=3,
    )


@pytest.fixture
def services(settings: Settings) -> Services:
    """Fully offline services: hashing embeddings, no reranker, echo LLM."""
    return Services(
        settings,
        llm_override=EchoLLM(),
        embedder_override=HashingEmbedder(),
        use_default_models=False,
    )
