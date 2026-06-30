"""Reciprocal Rank Fusion for combining vector and BM25 ranked results."""

import dataclasses
from collections import defaultdict
from uuid import UUID

from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk


def reciprocal_rank_fusion(
    *ranked_lists: list[RankedChunk],
    k: int = 60,
    top_n: int = 10,
) -> list[RankedChunk]:
    """Fuse any number of ranked lists using Reciprocal Rank Fusion.

    Each list contributes 1/(k+rank) per chunk, so a chunk appearing in several
    lists rises above one that appears in only one. Accepts two lists (vector +
    BM25) or more — e.g. raw-query vector + HyDE vector + BM25 keywords.
    k=60 is the standard constant — higher values reduce the impact of top ranks.
    """
    scores: dict[UUID, float] = defaultdict(float)
    chunk_map: dict[UUID, RankedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.id] += 1.0 / (k + rank)
            chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    return [
        dataclasses.replace(chunk_map[cid], rrf_score=scores[cid])
        for cid in sorted_ids[:top_n]
    ]
