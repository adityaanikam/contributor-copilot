"""Wire settings into a ready-to-use Copilot: chunks, indexes, retriever and LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from contributor_copilot.answer import Citation, generate_answer
from contributor_copilot.bm25 import BM25Index
from contributor_copilot.config import Settings
from contributor_copilot.dense import ChromaStore, SentenceTransformerEmbedder, collection_name
from contributor_copilot.ingest import build_chunks
from contributor_copilot.llm import get_llm
from contributor_copilot.rerank import CrossEncoderReranker
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
    retrieved: list[RetrievedSource] = field(default_factory=list)


class Copilot:
    def __init__(self, retriever: HybridRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str) -> AskResult:
        hits = self.retriever.retrieve(question)
        answer = generate_answer(question, hits, self.llm)
        return AskResult(
            question=question,
            answer=answer.text,
            citations=answer.citations,
            retrieved=[RetrievedSource(h.chunk.location, h.chunk.section, h.score) for h in hits],
        )


def build_retriever(settings: Settings) -> HybridRetriever:
    chunks = build_chunks(
        settings.corpus_dir,
        settings.repo_url,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.issues_path,
    )
    dense = None
    if settings.embeddings == "bge":
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        name = collection_name(embedder.name, settings.chunk_size, settings.chunk_overlap)
        dense = ChromaStore(chunks, embedder, settings.index_dir / "chroma", name)
    reranker = CrossEncoderReranker(settings.rerank_model) if settings.rerank else None
    return HybridRetriever(
        BM25Index(chunks),
        dense=dense,
        reranker=reranker,
        candidates=settings.candidates,
        top_k=settings.top_k,
    )


def build_copilot(settings: Settings) -> Copilot:
    return Copilot(build_retriever(settings), get_llm(settings))
