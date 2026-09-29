"""Retrieval benchmark: v1 baseline vs BM25 vs dense vs hybrid vs hybrid + reranker.

Known-item evaluation: queries are generated deterministically from the records
themselves, so no LLM and no API key are needed and results are reproducible. A query
counts as a hit when a retrieved chunk comes from a relevant record (same title, or for
CAS queries any record containing that CAS number).

Query types:
  name        "<substance> est-il dangereux pour une peau XP ?"          (lexical)
  synonym     "<synonym> est-il autorisé en cosmétique ?"                  (lexical, alias)
  cas         "Que dit la réglementation pour le numéro CAS <cas> ?"       (identifier)
  description ingredient function text with the ingredient name masked     (semantic)
  product     "Quels sont les ingrédients de <product> ?"                  (lexical, long)

Usage:
  python eval/retrieval_eval.py                    # uses DATA_DIR (default ./data)
  python eval/retrieval_eval.py --per-type 60 --out eval/results/retrieval.md
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from xp_family.config import get_settings
from xp_family.rag.documents import Document
from xp_family.rag.embeddings import SentenceTransformerEmbedder
from xp_family.rag.reranker import CrossEncoderReranker
from xp_family.rag.retrievers import DenseRetriever, ScoredDocument
from xp_family.rag.text import fold
from xp_family.services import build_retriever, load_knowledge_documents, load_product_documents

V1_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # what v1 used
_CAS = re.compile(r"\b\d{2,7}-\d{2}-\d\b")


@dataclass(frozen=True)
class Query:
    kind: str
    text: str
    relevant: frozenset[str]  # parent record ids


def parent_id(doc: Document) -> str:
    return doc.id.split("#", 1)[0]


def build_queries(docs: list[Document], products: list[Document], per_type: int, seed: int):
    rng = random.Random(seed)
    by_title: dict[str, set[str]] = defaultdict(set)
    for d in [*docs, *products]:
        by_title[fold(d.title).strip()].add(d.id)

    def same_title(d: Document) -> frozenset[str]:
        return frozenset(by_title[fold(d.title).strip()])

    queries: dict[str, list[Query]] = defaultdict(list)
    for d in docs:
        meta = d.metadata
        if d.source == "banned":
            queries["name"].append(
                Query("name", f"{d.title} est-il dangereux pour une peau XP ?", same_title(d))
            )
            for syn in meta.get("synonyms") or []:
                queries["synonym"].append(
                    Query("synonym", f"{syn} est-il autorisé en cosmétique ?", same_title(d))
                )
        for cas in dict.fromkeys(_CAS.findall(d.text)):
            holders = frozenset(x.id for x in docs if cas in x.text)
            queries["cas"].append(
                Query("cas", f"Que dit la réglementation pour le numéro CAS {cas} ?", holders)
            )
        if d.source == "ingredients" and meta.get("function"):
            masked = re.sub(re.escape(d.title), "cet ingrédient", str(meta["function"]), flags=re.I)
            sentence = re.split(r"(?<=[.!?])\s", masked.strip())[0][:220]
            queries["description"].append(
                Query("description", f"Quel ingrédient : {sentence}", same_title(d))
            )
    for p in products:
        queries["product"].append(
            Query("product", f"Quels sont les ingrédients de {p.title} ?", same_title(p))
        )

    # Deduplicate query text, then sample a fixed number per type.
    sampled: list[Query] = []
    for kind in ("name", "synonym", "cas", "description", "product"):
        unique = list({q.text: q for q in queries[kind]}.values())
        rng.shuffle(unique)
        sampled.extend(unique[:per_type])
    return sampled


def evaluate(
    search: Callable[[str], list[ScoredDocument]], queries: list[Query]
) -> dict[str, dict[str, float]]:
    per_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in queries:
        start = time.perf_counter()
        hits = search(q.text)
        latency = (time.perf_counter() - start) * 1000
        ranks = [i for i, h in enumerate(hits, 1) if parent_id(h.document) in q.relevant]
        first = ranks[0] if ranks else None
        for kind in (q.kind, "all"):
            m = per_kind[kind]
            m["hit@1"].append(float(first == 1))
            m["hit@3"].append(float(first is not None and first <= 3))
            m["mrr@10"].append(1.0 / first if first and first <= 10 else 0.0)
            m["latency_ms"].append(latency)
    return {
        kind: {
            name: statistics.median(vals) if name == "latency_ms" else statistics.fmean(vals)
            for name, vals in metrics.items()
        }
        for kind, metrics in per_kind.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("eval/results/retrieval.md"))
    args = parser.parse_args()

    settings = get_settings()
    knowledge = load_knowledge_documents(settings.data_dir)
    products = load_product_documents(settings.data_dir)
    corpus = [*knowledge, *products]
    queries = build_queries(knowledge, products, args.per_type, args.seed)
    print(f"{len(corpus)} records, {len(queries)} queries")

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    reranker = CrossEncoderReranker(settings.reranker_model)
    v1 = DenseRetriever(corpus, SentenceTransformerEmbedder(V1_EMBEDDING_MODEL))
    bm25 = build_retriever(corpus, settings, embedder, mode="bm25")
    dense = build_retriever(corpus, settings, embedder, mode="dense")
    hybrid = build_retriever(corpus, settings, embedder, mode="hybrid")

    systems: dict[str, Callable[[str], list[ScoredDocument]]] = {
        "v1: dense MiniLM-L6 (EN), whole records": lambda q: v1.search(q, 10),
        "BM25": lambda q: bm25.search(q, 10),
        "Dense multilingual, chunked": lambda q: dense.search(q, 10),
        "Hybrid (BM25 + dense, RRF)": lambda q: hybrid.search(q, 10),
        "Hybrid + cross-encoder rerank": lambda q: reranker.rerank(
            q, hybrid.search(q, settings.candidates_k), 10
        ),
    }
    results = {name: evaluate(fn, queries) for name, fn in systems.items()}

    kinds = ["all", "name", "synonym", "cas", "description", "product"]
    counts = {k: sum(q.kind == k for q in queries) for k in kinds[1:]}
    lines = [
        f"Known-item retrieval on the project datasets: {len(corpus)} records, "
        f"{len(queries)} queries ({', '.join(f'{k} {n}' for k, n in counts.items())}); "
        f"seed {args.seed}.",
        "",
        "| System | Hit@1 | Hit@3 | MRR@10 | "
        + " | ".join(f"Hit@3 {k}" for k in kinds[1:])
        + " | p50 latency |",
        "|---|---:|---:|---:|" + "---:|" * len(kinds[1:]) + "---:|",
    ]
    for name, res in results.items():
        overall = res["all"]
        per_type = " | ".join(f"{res[k]['hit@3']:.2f}" if k in res else "–" for k in kinds[1:])
        lines.append(
            f"| {name} | {overall['hit@1']:.2f} | {overall['hit@3']:.2f} | "
            f"{overall['mrr@10']:.2f} | {per_type} | {overall['latency_ms']:.0f} ms |"
        )
    table = "\n".join(lines)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
