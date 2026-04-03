"""Unit tests for deduplication service."""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from superbrain.app.application.ingestion.deduplication import (
    DeduplicationReason,
    DeduplicationService,
)
from superbrain.app.domain.models import Article


@dataclass
class FakeArticleRepository:
    """In-memory article repository for deduplication tests."""

    by_source: dict[str, Article]
    by_canonical: dict[str, Article]
    by_hash: dict[str, Article]

    def save(self, article: Article) -> Article:
        return article

    def save_chunks(self, chunks: list[object]) -> list[object]:
        return chunks

    def save_raw_snapshot(self, article_id: UUID, raw_html: str) -> None:
        _ = (article_id, raw_html)

    def get(self, article_id: UUID) -> Article | None:
        _ = article_id
        return None

    def get_by_source_url(self, source_url: str) -> Article | None:
        return self.by_source.get(source_url)

    def get_by_canonical_url(self, canonical_url: str) -> Article | None:
        return self.by_canonical.get(canonical_url)

    def get_by_content_hash(self, content_hash: str) -> Article | None:
        return self.by_hash.get(content_hash)


def _article(source_url: str, canonical_url: str, content_hash: str) -> Article:
    return Article(
        id=uuid4(),
        source_url=source_url,
        canonical_url=canonical_url,
        domain="example.com",
        title="Title",
        author=None,
        published_at=None,
        content="Body",
        content_hash=content_hash,
        extraction_quality_score=0.8,
        extraction_notes="note",
        created_at=datetime.now(UTC),
    )


def test_deduplication_detects_exact_url() -> None:
    """Service should detect existing exact source URL match."""

    existing = _article("https://x.com/a", "https://x.com/a", "hash1")
    repository = FakeArticleRepository(
        by_source={existing.source_url: existing},
        by_canonical={},
        by_hash={},
    )

    result = DeduplicationService(repository).check_url(existing.source_url, existing.canonical_url)
    assert result.is_duplicate is True
    assert result.reason == DeduplicationReason.EXACT_URL


def test_deduplication_detects_content_hash() -> None:
    """Service should detect content hash duplicates."""

    existing = _article("https://x.com/a", "https://x.com/a", "hash1")
    repository = FakeArticleRepository(by_source={}, by_canonical={}, by_hash={"hash1": existing})

    result = DeduplicationService(repository).check_content_hash("hash1")
    assert result.is_duplicate is True
    assert result.reason == DeduplicationReason.CONTENT_HASH
