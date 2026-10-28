"""Cross-encoder reranking of fused candidates."""

from __future__ import annotations

from typing import Protocol

from contributor_copilot.bm25 import Hit
from contributor_copilot.ingest import Chunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Hit]: ...


class CrossEncoderReranker:
    """ms-marco-MiniLM-L-6-v2 scores each (query, chunk) pair jointly.

    Slower than BM25 or embeddings because it runs once per candidate, which
    is why it only sees the top fused candidates.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self.name = model_name
        self._model = model

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Hit]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.search_text) for c in chunks])
        hits = [Hit(c, float(s)) for c, s in zip(chunks, scores)]
        return sorted(hits, key=lambda h: h.score, reverse=True)
