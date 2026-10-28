"""Dense retrieval: bge-small embeddings stored in an embedded Chroma database."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from contributor_copilot.bm25 import Hit
from contributor_copilot.ingest import Chunk


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """bge-small-en-v1.5 through sentence-transformers. Downloads on first use."""

    # bge v1.5 recommends this prefix on short queries (not on documents).
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self.name = model_name
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([self.QUERY_INSTRUCTION + text], normalize_embeddings=True)[0]
        return list(map(float, vector))


def collection_name(embedder_name: str, chunk_size: int, chunk_overlap: int) -> str:
    """One Chroma collection per (model, chunk size, overlap)."""
    slug = re.sub(r"[^a-z0-9.]+", "-", embedder_name.lower()).strip("-.")
    return f"c{chunk_size}-o{chunk_overlap}-{slug}"[:63].rstrip("-.")


class ChromaStore:
    """Keeps a Chroma collection in sync with the chunk list and searches it.

    Only chunks whose id is missing get embedded, so restarts are cheap.
    Chunks that disappeared from the corpus are deleted.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder, persist_dir: Path, name: str):
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = client.get_or_create_collection(
            name=name, configuration={"hnsw": {"space": "cosine"}}
        )
        self._embedder = embedder
        self._by_id = {c.id: c for c in chunks}
        self._sync(chunks)

    def _sync(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        stored = set(self._collection.get(include=[])["ids"])
        stale = sorted(stored - self._by_id.keys())
        if stale:
            self._collection.delete(ids=stale)
        missing = [c for c in chunks if c.id not in stored]
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=self._embedder.embed_documents([c.search_text for c in batch]),
                metadatas=[{"source": c.source} for c in batch],
            )

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, k: int) -> list[Hit]:
        """Top k chunks by cosine similarity (score = 1 - cosine distance)."""
        n = min(k, self.count())
        if n == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=n,
            include=["distances"],
        )
        ids, distances = result["ids"][0], result["distances"][0]
        return [Hit(self._by_id[i], 1.0 - d) for i, d in zip(ids, distances, strict=True) if i in self._by_id]
