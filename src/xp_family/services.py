"""Wiring: builds the pipelines once, lazily, from settings.

v1 built indexes and loaded models at *import* time in each Django view module, so one
missing file crashed the whole site. Here nothing heavy happens until first use, and
both front-ends (FastAPI and Django) share the same instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

from xp_family import prompts
from xp_family.config import Settings, get_settings
from xp_family.llm import LLM, GeminiLLM
from xp_family.observability import get_logger
from xp_family.rag.chunking import chunk_documents
from xp_family.rag.documents import Document, load_products, load_records
from xp_family.rag.embeddings import Embedder
from xp_family.rag.pipeline import RAGPipeline
from xp_family.rag.reranker import CrossEncoderReranker, Reranker
from xp_family.rag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from xp_family.safety import SafetyChecker

log = get_logger(__name__)

# File names inside DATA_DIR (see data/README.md for provenance and schema).
KNOWLEDGE_FILE = "knowledge.json"
BANNED_FILE = "banned_ingredients.json"
INGREDIENT_INFO_FILE = "ingredients_info.json"
PRODUCTS_FILE = "products_xp_checked.json"


class DataMissingError(FileNotFoundError):
    pass


def _require(path: Path) -> Path:
    if not path.exists():
        raise DataMissingError(f"{path} not found — see data/README.md")
    return path


def load_knowledge_documents(data_dir: Path) -> list[Document]:
    docs = load_records(_require(data_dir / KNOWLEDGE_FILE), "knowledge")
    docs += load_records(_require(data_dir / BANNED_FILE), "banned")
    if (data_dir / INGREDIENT_INFO_FILE).exists():
        docs += load_records(data_dir / INGREDIENT_INFO_FILE, "ingredients")
    return docs


def load_product_documents(data_dir: Path) -> list[Document]:
    return load_products(_require(data_dir / PRODUCTS_FILE))


def build_retriever(
    docs: list[Document], settings: Settings, embedder: Embedder | None, mode: str = "hybrid"
) -> Retriever:
    """mode: 'bm25', 'dense' or 'hybrid'. Falls back to BM25 when no embedder is available."""
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if mode == "bm25" or embedder is None:
        return BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embedder, settings.cache_dir / "embeddings")
    if mode == "dense":
        return dense
    return HybridRetriever(
        [BM25Retriever(chunks), dense], candidates_k=settings.candidates_k, rrf_k=settings.rrf_k
    )


def default_embedder(settings: Settings) -> Embedder | None:
    try:
        from xp_family.rag.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(settings.embedding_model)
    except ImportError:
        log.warning("dense_retrieval_disabled", reason="install the 'ml' extra")
        return None


def default_reranker(settings: Settings) -> Reranker | None:
    if not settings.use_reranker:
        return None
    try:
        return CrossEncoderReranker(settings.reranker_model)
    except ImportError:
        log.warning("reranker_disabled", reason="install the 'ml' extra")
        return None


@dataclass
class Services:
    settings: Settings
    llm_override: LLM | None = None
    embedder_override: Embedder | None = None
    reranker_override: Reranker | None = None
    use_default_models: bool = True

    @cached_property
    def llm(self) -> LLM:
        return self.llm_override or GeminiLLM(self.settings)

    @cached_property
    def embedder(self) -> Embedder | None:
        if self.embedder_override or not self.use_default_models:
            return self.embedder_override
        return default_embedder(self.settings)

    @cached_property
    def reranker(self) -> Reranker | None:
        if self.reranker_override or not self.use_default_models:
            return self.reranker_override
        return default_reranker(self.settings)

    @cached_property
    def knowledge(self) -> RAGPipeline:
        docs = load_knowledge_documents(self.settings.data_dir)
        return RAGPipeline(
            build_retriever(docs, self.settings, self.embedder),
            self.llm,
            prompts.KNOWLEDGE_PROMPT,
            reranker=self.reranker,
            candidates_k=self.settings.candidates_k,
            top_k=self.settings.top_k,
            name="knowledge",
        )

    @cached_property
    def recipes(self) -> RAGPipeline:
        docs = load_product_documents(self.settings.data_dir)
        return RAGPipeline(
            build_retriever(docs, self.settings, self.embedder),
            self.llm,
            prompts.RECIPE_PROMPT,
            reranker=self.reranker,
            candidates_k=self.settings.candidates_k,
            top_k=self.settings.top_k,
            name="recipes",
        )

    @cached_property
    def safety(self) -> SafetyChecker:
        data_dir = self.settings.data_dir
        return SafetyChecker.from_files(
            _require(data_dir / BANNED_FILE), data_dir / INGREDIENT_INFO_FILE
        )


@lru_cache
def get_services() -> Services:
    return Services(get_settings())
