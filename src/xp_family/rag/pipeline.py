"""Retrieve → rerank → generate, returning the answer together with its evidence."""

from __future__ import annotations

from dataclasses import dataclass

from xp_family.llm import LLM
from xp_family.observability import trace
from xp_family.rag.reranker import Reranker
from xp_family.rag.retrievers import Retriever, ScoredDocument


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    contexts: list[ScoredDocument]


def format_context(hits: list[ScoredDocument]) -> str:
    return "\n\n".join(f"[{i}] {hit.document.text}" for i, hit in enumerate(hits, start=1))


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        prompt_template: str,
        *,
        reranker: Reranker | None = None,
        candidates_k: int = 20,
        top_k: int = 4,
        name: str = "rag",
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt = prompt_template
        self._reranker = reranker
        self._candidates_k = candidates_k
        self._top_k = top_k
        self.name = name

    def retrieve(self, question: str) -> list[ScoredDocument]:
        with trace("retrieve", pipeline=self.name, reranked=self._reranker is not None) as span:
            if self._reranker is None:
                hits = self._retriever.search(question, self._top_k)
            else:
                candidates = self._retriever.search(question, self._candidates_k)
                hits = self._reranker.rerank(question, candidates, self._top_k)
            span.set(hits=len(hits), top_ids=[h.document.id for h in hits])
        return hits

    def answer(self, question: str) -> RAGAnswer:
        hits = self.retrieve(question)
        prompt = self._prompt.format(context=format_context(hits), question=question)
        return RAGAnswer(question=question, answer=self._llm.generate(prompt), contexts=hits)
