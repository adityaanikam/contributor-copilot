"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[Hashable]], k: int = RRF_K
) -> list[tuple[Hashable, float]]:
    """Merge ranked lists. Each item scores sum(1 / (k + rank)) over the lists.

    Only ranks are used, so BM25 scores and cosine similarities never need to
    be put on the same scale. Ties keep the order items were first seen in.
    """
    scores: dict[Hashable, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
