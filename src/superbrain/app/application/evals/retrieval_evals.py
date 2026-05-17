"""Retrieval quality metrics: recall@k and URL coverage."""

from uuid import UUID

from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk


def check_recall_at_k(
    retrieved_chunks: list[RankedChunk],
    expected_chunk_ids: list[UUID],
    k: int,
) -> float:
    """Fraction of expected chunks that appear in the top-k retrieved results."""
    if not expected_chunk_ids:
        return 0.0
    top_k_ids = {c.id for c in retrieved_chunks[:k]}
    found = sum(1 for eid in expected_chunk_ids if eid in top_k_ids)
    return found / len(expected_chunk_ids)


def check_url_coverage(
    retrieved_chunks: list[RankedChunk],
    expected_urls: list[str],
) -> float:
    """Fraction of expected article URLs present anywhere in retrieved results."""
    if not expected_urls:
        return 0.0
    retrieved_urls = {c.url for c in retrieved_chunks}
    found = sum(1 for url in expected_urls if url in retrieved_urls)
    return found / len(expected_urls)
