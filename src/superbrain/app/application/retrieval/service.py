"""Hybrid retrieval implementation for grounded QA."""

import math
import re

from superbrain.app.application.ports import EmbeddingProvider
from superbrain.app.application.retrieval.models import (
    EvidenceSet,
    RetrievalCandidate,
    RetrievalResult,
    ScoredChunk,
)
from superbrain.app.domain.repositories import RetrievalRepository


def normalize_query(question: str) -> str:
    """Normalize query text for retrieval operations."""

    return " ".join(question.lower().split())


def reciprocal_rank_fusion(vector_rank: int | None, lexical_rank: int | None, k: int = 60) -> float:
    """Fuse rank signals using reciprocal rank fusion."""

    score = 0.0
    if vector_rank is not None:
        score += 1 / (k + vector_rank)
    if lexical_rank is not None:
        score += 1 / (k + lexical_rank)
    return score


class HybridRetrievalService:
    """Combine vector and lexical retrieval with score fusion."""

    def __init__(
        self,
        retrieval_repository: RetrievalRepository,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        """Initialize retrieval service dependencies."""

        self._retrieval_repository = retrieval_repository
        self._embedding_provider = embedding_provider

    def retrieve(self, question: str, limit: int = 8) -> RetrievalResult:
        """Retrieve top evidence chunks for a question."""

        normalized = normalize_query(question)
        chunks = self._retrieval_repository.list_chunks(limit=1000)
        if not chunks:
            return RetrievalResult(
                normalized_query=normalized,
                evidence=EvidenceSet(chunks=tuple()),
            )

        query_vector = self._embedding_provider.embed_query(normalized)
        query_tokens = set(re.findall(r"\w+", normalized))
        pg_lexical_scores = self._retrieval_repository.lexical_scores(normalized, limit=500)

        vector_scores: list[tuple[str, float]] = []
        lexical_scores: list[tuple[str, float]] = []
        candidate_map: dict[str, RetrievalCandidate] = {}

        for chunk in chunks:
            vector_score = _cosine_similarity(query_vector, chunk.embedding)
            lexical_score = pg_lexical_scores.get(str(chunk.chunk_id))
            if lexical_score is None:
                lexical_score = _lexical_overlap_score(query_tokens, chunk.chunk_text)

            chunk_key = str(chunk.chunk_id)
            vector_scores.append((chunk_key, vector_score))
            lexical_scores.append((chunk_key, lexical_score))
            candidate_map[chunk_key] = RetrievalCandidate(
                chunk=chunk,
                vector_score=vector_score,
                lexical_score=lexical_score,
                fused_score=0.0,
            )

        vector_scores.sort(key=lambda item: item[1], reverse=True)
        lexical_scores.sort(key=lambda item: item[1], reverse=True)

        vector_ranks = {chunk_id: index + 1 for index, (chunk_id, _) in enumerate(vector_scores)}
        lexical_ranks = {chunk_id: index + 1 for index, (chunk_id, _) in enumerate(lexical_scores)}

        rescored: list[RetrievalCandidate] = []
        for chunk_id, candidate in candidate_map.items():
            fused = reciprocal_rank_fusion(
                vector_ranks.get(chunk_id),
                lexical_ranks.get(chunk_id),
            )
            rescored.append(
                RetrievalCandidate(
                    chunk=candidate.chunk,
                    vector_score=candidate.vector_score,
                    lexical_score=candidate.lexical_score,
                    fused_score=fused,
                )
            )

        rescored.sort(key=lambda candidate: candidate.fused_score, reverse=True)
        scored = tuple(
            ScoredChunk(chunk=candidate.chunk, fused_score=candidate.fused_score, rank=index + 1)
            for index, candidate in enumerate(rescored[:limit])
        )

        return RetrievalResult(normalized_query=normalized, evidence=EvidenceSet(chunks=scored))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_overlap_score(query_tokens: set[str], chunk_text: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = set(re.findall(r"\w+", chunk_text.lower()))
    if not chunk_tokens:
        return 0.0
    return len(query_tokens.intersection(chunk_tokens)) / len(query_tokens)
