"""Unit tests for retrieval score fusion."""

from superbrain.app.application.retrieval.service import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_higher_ranks() -> None:
    """Chunks with better ranks should receive higher fused scores."""

    high_score = reciprocal_rank_fusion(1, 2)
    low_score = reciprocal_rank_fusion(10, 12)

    assert high_score > low_score


def test_reciprocal_rank_fusion_handles_missing_signal() -> None:
    """Fusion should still produce score when one modality is missing."""

    only_vector = reciprocal_rank_fusion(3, None)
    only_lexical = reciprocal_rank_fusion(None, 3)

    assert only_vector > 0
    assert only_lexical > 0
