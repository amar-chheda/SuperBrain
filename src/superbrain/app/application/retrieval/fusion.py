"""Reciprocal Rank Fusion for combining vector and BM25 ranked results."""

import dataclasses
from collections import defaultdict
from uuid import UUID

from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk


def reciprocal_rank_fusion(
    vector_results: list[RankedChunk],
    bm25_results: list[RankedChunk],
    k: int = 60,
    top_n: int = 10,
) -> list[RankedChunk]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    A chunk appearing in both lists gets contributions from each, so it
    naturally rises above chunks that only appear in one list.
    k=60 is the standard constant — higher values reduce the impact of top ranks.
    """
    scores: dict[UUID, float] = defaultdict(float)
    chunk_map: dict[UUID, RankedChunk] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        scores[chunk.id] += 1.0 / (k + rank)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        scores[chunk.id] += 1.0 / (k + rank)
        chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    return [
        dataclasses.replace(chunk_map[cid], rrf_score=scores[cid])
        for cid in sorted_ids[:top_n]
    ]
