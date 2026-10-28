import pytest
from fakes import HashingEmbedder, KeywordReranker

from contributor_copilot.bm25 import BM25Index
from contributor_copilot.dense import ChromaStore
from contributor_copilot.ingest import build_chunks
from contributor_copilot.retriever import HybridRetriever

REPO = "https://github.com/microsoft/apm"


@pytest.fixture(scope="module")
def chunks():
    from conftest import SAMPLE_CORPUS

    return build_chunks(SAMPLE_CORPUS, REPO, 1000, 150)


def test_bm25_only_mode(chunks):
    retriever = HybridRetriever(BM25Index(chunks), top_k=3)
    assert retriever.mode == "bm25"
    hits = retriever.retrieve("What does apm run do with no script name?")
    assert len(hits) == 3
    assert hits[0].chunk.source.endswith("consumer/run-scripts.md")


def test_hybrid_mode_fuses_bm25_and_dense(chunks, tmp_path):
    dense = ChromaStore(chunks, HashingEmbedder(), tmp_path / "chroma", "test")
    retriever = HybridRetriever(BM25Index(chunks), dense=dense, candidates=10, top_k=5)
    assert retriever.mode == "hybrid"
    hits = retriever.retrieve("GITLAB_APM_PAT")
    assert len(hits) == 5
    assert any(h.chunk.source.endswith("consumer/authentication.md") for h in hits)


def test_reranker_reorders_candidates_using_rerank_query(chunks):
    reranker = KeywordReranker("Nadav")
    retriever = HybridRetriever(BM25Index(chunks), reranker=reranker, top_k=2)
    assert retriever.mode == "bm25+rerank"
    hits = retriever.retrieve(["maintainers approval scope"], rerank_query="who owns the registry API?")
    assert "Nadav" in hits[0].chunk.text
    assert reranker.queries == ["who owns the registry API?"]


def test_multiple_queries_are_fused(chunks):
    retriever = HybridRetriever(BM25Index(chunks), candidates=5, top_k=10)
    sources = {h.chunk.source for h in retriever.retrieve(["GITLAB_APM_PAT", "apm doctor network probe"])}
    assert any(s.endswith("consumer/authentication.md") for s in sources)
    assert any(s.endswith("reference/cli/doctor.md") for s in sources)


def test_top_k_override(chunks):
    retriever = HybridRetriever(BM25Index(chunks), top_k=5)
    assert len(retriever.retrieve("apm install", top_k=12)) == 12
