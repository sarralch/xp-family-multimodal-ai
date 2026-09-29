"""Sparse (BM25), dense and hybrid retrieval.

Why hybrid: the knowledge base is dominated by exact identifiers — INCI names, CAS
numbers like 106-88-7, French regulatory spellings — where lexical BM25 is strong and
embeddings are weak, while user questions are free-form, where embeddings are strong.
Reciprocal Rank Fusion combines the two rankings without having to calibrate scores.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from xp_family.rag.documents import Document
from xp_family.rag.embeddings import Embedder
from xp_family.rag.text import tokenize


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[ScoredDocument]: ...


class BM25Retriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        corpus = [tokenize(d.text) or ["_"] for d in docs]
        self._vocab = [set(tokens) for tokens in corpus]
        self._index = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[ScoredDocument]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        # Filter on actual term overlap, not score > 0: Okapi IDF is exactly zero for a
        # term present in half the corpus, which would silently drop true matches.
        matching = [i for i, vocab in enumerate(self._vocab) if vocab.intersection(tokens)]
        top = sorted(matching, key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredDocument(self._docs[i], float(scores[i])) for i in top]


class DenseRetriever:
    """Exact cosine search over normalised embeddings, cached as .npy (no pickle)."""

    def __init__(self, docs: list[Document], embedder: Embedder, cache_dir: Path | None = None):
        self._docs = docs
        self._embedder = embedder
        self._matrix = self._load_or_build(cache_dir)

    def _load_or_build(self, cache_dir: Path | None) -> np.ndarray:
        fingerprint = hashlib.sha1(
            "|".join([self._embedder.name, *(d.id for d in self._docs)]).encode()
        ).hexdigest()[:16]
        path = cache_dir / f"embeddings-{fingerprint}.npy" if cache_dir else None
        if path and path.exists():
            return np.load(path, allow_pickle=False)
        matrix = self._embedder.embed([d.text for d in self._docs])
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix, allow_pickle=False)
        return matrix

    def search(self, query: str, k: int) -> list[ScoredDocument]:
        if not self._docs:
            return []
        query_vec = self._embedder.embed([query])[0]
        scores = self._matrix @ query_vec
        top = np.argsort(-scores)[:k]
        return [ScoredDocument(self._docs[i], float(scores[i])) for i in top]


def reciprocal_rank_fusion(
    rankings: list[list[ScoredDocument]], k: int, rrf_k: int = 60
) -> list[ScoredDocument]:
    fused: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            fused[hit.document.id] = fused.get(hit.document.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            by_id[hit.document.id] = hit.document
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
    return [ScoredDocument(by_id[doc_id], score) for doc_id, score in ordered]


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], candidates_k: int = 20, rrf_k: int = 60):
        self._retrievers = retrievers
        self._candidates_k = candidates_k
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[ScoredDocument]:
        rankings = [r.search(query, self._candidates_k) for r in self._retrievers]
        return reciprocal_rank_fusion(rankings, k=k, rrf_k=self._rrf_k)
