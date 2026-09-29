"""Cross-encoder reranking of fused candidates.

Retrieval casts a wide net (20 candidates per retriever); the cross-encoder reads each
(query, chunk) pair jointly and keeps only the few chunks the LLM actually needs.
"""

from __future__ import annotations

from typing import Protocol

from xp_family.rag.retrievers import ScoredDocument


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[ScoredDocument], k: int) -> list[ScoredDocument]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[ScoredDocument], k: int) -> list[ScoredDocument]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.document.text) for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [ScoredDocument(h.document, float(s)) for h, s in ranked[:k]]
