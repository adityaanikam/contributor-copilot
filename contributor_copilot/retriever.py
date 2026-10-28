"""Hybrid retrieval: BM25 + dense, fused with RRF, then optionally reranked."""

from __future__ import annotations

from contributor_copilot.bm25 import BM25Index, Hit
from contributor_copilot.dense import ChromaStore
from contributor_copilot.fusion import reciprocal_rank_fusion
from contributor_copilot.rerank import Reranker


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: ChromaStore | None = None,
        reranker: Reranker | None = None,
        candidates: int = 20,
        top_k: int = 5,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k
        self._by_id = {c.id: c for c in bm25.chunks}

    @property
    def mode(self) -> str:
        retrieval = "hybrid" if self.dense else "bm25"
        return f"{retrieval}+rerank" if self.reranker else retrieval

    def retrieve(
        self, queries: str | list[str], rerank_query: str | None = None, top_k: int | None = None
    ) -> list[Hit]:
        """Return the top_k chunks for one or more queries.

        Every query is run through BM25 (and the dense store when present).
        All the ranked lists are merged with reciprocal rank fusion. The fused
        top candidates are then reranked against rerank_query, which defaults
        to the first query.
        """
        queries = [queries] if isinstance(queries, str) else list(queries)
        top_k = self.top_k if top_k is None else top_k
        rankings = []
        for query in queries:
            rankings.append([h.chunk.id for h in self.bm25.search(query, self.candidates)])
            if self.dense is not None:
                rankings.append([h.chunk.id for h in self.dense.search(query, self.candidates)])
        fused = reciprocal_rank_fusion(rankings)[: self.candidates]
        if self.reranker is None:
            return [Hit(self._by_id[chunk_id], score) for chunk_id, score in fused[:top_k]]
        chunks = [self._by_id[chunk_id] for chunk_id, _ in fused]
        return self.reranker.rerank(rerank_query or queries[0], chunks)[:top_k]
