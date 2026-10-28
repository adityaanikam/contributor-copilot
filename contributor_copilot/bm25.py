"""Keyword retrieval with BM25 (rank_bm25)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from contributor_copilot.ingest import Chunk

# Words like "env.var", "GITLAB_APM_PAT" or "self-update" are kept whole
# and also split into their parts, so both forms match.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*")
PART_RE = re.compile(r"[._\-]")
STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in into is it its "
    "my of on or so that the then there these this to was what when where which "
    "who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in TOKEN_RE.findall(text.lower()):
        if word not in STOPWORDS:
            tokens.append(word)
        parts = PART_RE.split(word)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p and p not in STOPWORDS)
    return tokens


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.search_text) for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        """Top k chunks by BM25 score. Chunks with no matching term are dropped."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [Hit(self.chunks[i], float(scores[i])) for i in order[:k] if scores[i] > 0]
