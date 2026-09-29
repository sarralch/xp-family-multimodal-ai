"""End-to-end RAG evaluation with RAGAS: v1-style retrieval vs the v2 pipeline.

Metrics (RAGAS 0.4 collections API, Gemini as judge):
  faithfulness       — are the answer's claims supported by the retrieved context?
  answer_relevancy   — does the answer address the question?
  context_precision  — are the relevant chunks ranked first? (needs a reference)
  context_recall     — does the context cover the reference answer? (needs a reference)

Both systems share the same LLM and prompt, so differences come from retrieval only.

Test set: JSON Lines with {"question": ..., "reference": ...} per line.
  data/eval_testset.jsonl     your real set (git-ignored, like the data)
  eval/testset.sample.jsonl   synthetic set matching data/sample (committed)

Usage (needs GEMINI_API_KEY and the ml + eval extras; runs well in Docker):
  python eval/ragas_eval.py --testset data/eval_testset.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from xp_family import prompts
from xp_family.config import get_settings
from xp_family.llm import GeminiLLM
from xp_family.rag.embeddings import SentenceTransformerEmbedder
from xp_family.rag.pipeline import RAGAnswer, RAGPipeline
from xp_family.rag.retrievers import DenseRetriever
from xp_family.services import build_retriever, default_reranker, load_knowledge_documents

V1_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def load_testset(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_systems(settings) -> dict[str, RAGPipeline]:
    llm = GeminiLLM(settings)
    docs = load_knowledge_documents(settings.data_dir)
    v1 = DenseRetriever(docs, SentenceTransformerEmbedder(V1_EMBEDDING_MODEL))
    hybrid = build_retriever(docs, settings, SentenceTransformerEmbedder(settings.embedding_model))
    return {
        "v1 retrieval (dense MiniLM-L6, k=3)": RAGPipeline(
            v1, llm, prompts.KNOWLEDGE_PROMPT, top_k=3, name="v1"
        ),
        "v2 (hybrid + rerank, k=4)": RAGPipeline(
            hybrid,
            llm,
            prompts.KNOWLEDGE_PROMPT,
            reranker=default_reranker(settings),
            candidates_k=settings.candidates_k,
            top_k=settings.top_k,
            name="v2",
        ),
    }


def build_metrics(settings):
    from google import genai
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    judge = llm_factory(
        settings.gemini_model, provider="google", client=client, adapter="instructor"
    )
    embeddings = HuggingFaceEmbeddings(model=settings.embedding_model)
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


async def score_one(metrics, sample: dict[str, str], result: RAGAnswer, limit: asyncio.Semaphore):
    contexts = [h.document.text for h in result.contexts]
    q, ref = sample["question"], sample["reference"]
    calls = {
        "faithfulness": lambda m: m.ascore(
            user_input=q, response=result.answer, retrieved_contexts=contexts
        ),
        "answer_relevancy": lambda m: m.ascore(user_input=q, response=result.answer),
        "context_precision": lambda m: m.ascore(
            user_input=q, reference=ref, retrieved_contexts=contexts
        ),
        "context_recall": lambda m: m.ascore(
            user_input=q, retrieved_contexts=contexts, reference=ref
        ),
    }
    scores = {}
    for name, call in calls.items():
        async with limit:
            try:
                scores[name] = float((await call(metrics[name])).value)
            except Exception as exc:  # one failed judgement must not sink the whole run
                print(f"  ! {name} failed on {sample['question'][:50]!r}: {exc}")
    return scores


async def run(args) -> None:
    settings = get_settings()
    testset = load_testset(args.testset)[: args.limit or None]
    systems = build_systems(settings)
    metrics = build_metrics(settings)
    limit = asyncio.Semaphore(args.concurrency)

    rows = []
    for name, pipeline in systems.items():
        answers = [pipeline.answer(s["question"]) for s in testset]
        per_sample = await asyncio.gather(
            *(score_one(metrics, s, a, limit) for s, a in zip(testset, answers, strict=True))
        )
        means = {
            m: statistics.fmean(vals) if (vals := [s[m] for s in per_sample if m in s]) else None
            for m in METRICS
        }
        rows.append((name, means))
        if args.runs_dir:  # per-question detail stays local (git-ignored)
            args.runs_dir.mkdir(parents=True, exist_ok=True)
            out = args.runs_dir / f"{pipeline.name}.jsonl"
            with out.open("w", encoding="utf-8") as fh:
                for s, a, sc in zip(testset, answers, per_sample, strict=True):
                    fh.write(json.dumps({**s, "answer": a.answer, **sc}, ensure_ascii=False) + "\n")

    header = "| System | " + " | ".join(m.replace("_", " ").title() for m in METRICS) + " |"
    lines = [
        f"RAGAS {len(testset)} questions, judge `{settings.gemini_model}`.",
        "",
        header,
        "|---|" + "---:|" * len(METRICS),
    ]
    for name, means in rows:
        cells = " | ".join(f"{means[m]:.2f}" if means[m] is not None else "–" for m in METRICS)
        lines.append(f"| {name} | {cells} |")
    table = "\n".join(lines)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--testset", type=Path, default=Path("data/eval_testset.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("eval/results/ragas.md"))
    parser.add_argument("--runs-dir", type=Path, default=Path("eval/runs"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
