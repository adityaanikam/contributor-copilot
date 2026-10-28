"""Wire settings into a ready-to-use Copilot: chunks, indexes, retriever and LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from contributor_copilot.answer import Citation
from contributor_copilot.bm25 import BM25Index
from contributor_copilot.config import Settings
from contributor_copilot.dense import (
    ChromaStore,
    Embedder,
    SentenceTransformerEmbedder,
    collection_name,
)
from contributor_copilot.graph import build_graph
from contributor_copilot.ingest import build_chunks
from contributor_copilot.llm import get_llm
from contributor_copilot.rerank import CrossEncoderReranker, Reranker
from contributor_copilot.retriever import HybridRetriever


@dataclass
class RetrievedSource:
    location: str
    section: str
    score: float


@dataclass
class AskResult:
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    attempts: int
    queries: list[str] = field(default_factory=list)
    retrieved: list[RetrievedSource] = field(default_factory=list)


class Copilot:
    def __init__(self, retriever: HybridRetriever, llm: BaseChatModel, max_attempts: int = 2):
        self.retriever = retriever
        self.llm = llm
        self.graph = build_graph(retriever, llm, max_attempts=max_attempts)

    def ask(self, question: str) -> AskResult:
        state = self.graph.invoke({"question": question})
        return AskResult(
            question=question,
            answer=state["answer"],
            citations=state["citations"],
            grounded=state["grounded"],
            attempts=state["attempts"],
            queries=state["queries"],
            retrieved=[
                RetrievedSource(h.chunk.location, h.chunk.section, h.score) for h in state["hits"]
            ],
        )


def build_retriever(
    settings: Settings,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    """Build chunks and indexes. Pass embedder/reranker to reuse loaded models."""
    chunks = build_chunks(
        settings.corpus_dir,
        settings.repo_url,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.issues_path,
    )
    dense = None
    if settings.embeddings == "bge":
        embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
        name = collection_name(embedder.name, settings.chunk_size, settings.chunk_overlap)
        dense = ChromaStore(chunks, embedder, settings.index_dir / "chroma", name)
    if settings.rerank:
        reranker = reranker or CrossEncoderReranker(settings.rerank_model)
    else:
        reranker = None
    return HybridRetriever(
        BM25Index(chunks),
        dense=dense,
        reranker=reranker,
        candidates=settings.candidates,
        top_k=settings.top_k,
    )


def build_copilot(settings: Settings) -> Copilot:
    return Copilot(build_retriever(settings), get_llm(settings))
