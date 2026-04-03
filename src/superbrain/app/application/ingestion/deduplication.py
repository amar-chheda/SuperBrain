"""Deduplication service for ingestion requests."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from superbrain.app.domain.repositories import ArticleRepository


class DeduplicationReason(StrEnum):
    """Reason why an ingestion candidate is considered duplicate."""

    EXACT_URL = "exact_url"
    CANONICAL_URL = "canonical_url"
    CONTENT_HASH = "content_hash"


@dataclass(slots=True, frozen=True)
class DeduplicationResult:
    """Deduplication decision and associated article identity."""

    is_duplicate: bool
    reason: DeduplicationReason | None = None
    article_id: UUID | None = None


class DeduplicationService:
    """Evaluate ingestion candidates against existing records."""

    def __init__(self, article_repository: ArticleRepository) -> None:
        """Initialize service with an article repository."""

        self._article_repository = article_repository

    def check_url(self, source_url: str, canonical_url: str) -> DeduplicationResult:
        """Check for exact and canonical URL duplicates."""

        exact = self._article_repository.get_by_source_url(source_url)
        if exact is not None:
            return DeduplicationResult(
                is_duplicate=True,
                reason=DeduplicationReason.EXACT_URL,
                article_id=exact.id,
            )

        canonical = self._article_repository.get_by_canonical_url(canonical_url)
        if canonical is not None:
            return DeduplicationResult(
                is_duplicate=True,
                reason=DeduplicationReason.CANONICAL_URL,
                article_id=canonical.id,
            )

        return DeduplicationResult(is_duplicate=False)

    def check_content_hash(self, content_hash: str) -> DeduplicationResult:
        """Check for content hash duplicates after extraction."""

        candidate = self._article_repository.get_by_content_hash(content_hash)
        if candidate is None:
            return DeduplicationResult(is_duplicate=False)
        return DeduplicationResult(
            is_duplicate=True,
            reason=DeduplicationReason.CONTENT_HASH,
            article_id=candidate.id,
        )
