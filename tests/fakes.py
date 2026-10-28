"""Test doubles that need no network and no model downloads."""

import hashlib
import math

from contributor_copilot.bm25 import Hit, tokenize


class HashingEmbedder:
    """Bag-of-words vectors from hashed tokens. Only for exercising code paths."""

    name = "test-hashing"

    def __init__(self, dims: int = 64):
        self.dims = dims
        self.calls = 0

    def _embed(self, text):
        vector = [0.0] * self.dims
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class KeywordReranker:
    """Puts chunks containing the keyword first. Records what it was asked."""

    def __init__(self, keyword):
        self.keyword = keyword
        self.queries = []

    def rerank(self, query, chunks):
        self.queries.append(query)
        hits = [Hit(c, 1.0 if self.keyword in c.text else 0.0) for c in chunks]
        return sorted(hits, key=lambda h: h.score, reverse=True)
