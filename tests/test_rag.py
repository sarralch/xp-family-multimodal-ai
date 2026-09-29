from xp_family.llm import EchoLLM
from xp_family.rag.chunking import chunk_document
from xp_family.rag.documents import Document, load_products, load_records, record_to_document
from xp_family.rag.embeddings import HashingEmbedder
from xp_family.rag.pipeline import RAGPipeline
from xp_family.rag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ScoredDocument,
    reciprocal_rank_fusion,
)
from xp_family.rag.text import tokenize

from .conftest import SAMPLE_DATA


def doc(i: str, text: str) -> Document:
    return Document(id=i, title=i, text=text, source="t")


def test_tokenize_keeps_cas_numbers_and_folds_accents():
    assert tokenize("1,2-Époxybutane CAS 106-88-7") == ["1", "2-epoxybutane", "cas", "106-88-7"]


def test_record_rendering_skips_empty_and_none_values():
    d = record_to_document({"nom": "Zinc", "synonymes": "None", "cas": "", "x": ["a", "b"]}, "k")
    assert d.title == "Zinc"
    assert d.text == "Nom: Zinc\nX: a, b"


def test_loaders_handle_list_and_keyed_formats():
    assert len(load_records(SAMPLE_DATA / "knowledge.json", "knowledge")) == 3
    banned = load_records(SAMPLE_DATA / "banned_ingredients.json", "banned")
    assert {d.title for d in banned} >= {"Hydroquinone", "Benzophenones"}
    products = load_products(SAMPLE_DATA / "products_xp_checked.json")
    assert "Compatible XP: non" in products[1].text


def test_short_record_is_one_chunk_with_title_header():
    [chunk] = chunk_document(doc("a", "short text"), size=200)
    assert chunk.text == "[a]\nshort text"


def test_long_record_splits_on_boundaries_within_budget_with_overlap():
    sentences = " ".join(f"Sentence number {i} about zinc." for i in range(60))
    chunks = chunk_document(doc("long", sentences), size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.text.startswith("[long]\n") for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks)  # never cut mid-sentence
    first_tail = chunks[0].text.splitlines()[-1]
    assert first_tail in chunks[1].text  # overlap carried forward


def test_bm25_finds_exact_identifier():
    docs = [doc("a", "Hydroquinone CAS 123-31-9"), doc("b", "Zinc oxide sunscreen")]
    assert BM25Retriever(docs).search("123-31-9", 1)[0].document.id == "a"


def test_dense_cache_roundtrip(tmp_path):
    docs = [doc("a", "zinc oxide"), doc("b", "retinol vitamin")]
    first = DenseRetriever(docs, HashingEmbedder(), tmp_path)
    assert list(tmp_path.glob("embeddings-*.npy"))
    second = DenseRetriever(docs, HashingEmbedder(), tmp_path)
    assert second.search("zinc", 1)[0].document.id == first.search("zinc", 1)[0].document.id


def test_rrf_rewards_agreement_between_rankers():
    a, b, c = doc("a", ""), doc("b", ""), doc("c", "")
    fused = reciprocal_rank_fusion(
        [
            [ScoredDocument(a, 1), ScoredDocument(b, 1)],
            [ScoredDocument(b, 1), ScoredDocument(c, 1)],
        ],
        k=3,
    )
    assert fused[0].document.id == "b"


def test_hybrid_pipeline_returns_grounded_answer_with_sources():
    docs = [
        doc("a", "Zinc oxide is a mineral UV filter."),
        doc("b", "Retinol is photosensitising."),
    ]
    retriever = HybridRetriever([BM25Retriever(docs), DenseRetriever(docs, HashingEmbedder())])
    llm = EchoLLM()
    result = RAGPipeline(retriever, llm, "{context}\nQ: {question}", top_k=1).answer("zinc oxide?")
    assert result.contexts[0].document.id == "a"
    assert "Zinc oxide" in result.answer
    assert "Q: zinc oxide?" in llm.prompts[0]
