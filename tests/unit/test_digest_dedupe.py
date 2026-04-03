"""Unit tests for digest dedupe policy."""

from datetime import UTC, datetime
from uuid import uuid4

from superbrain.app.application.digest.deduplication import CanonicalUrlDigestDeduper
from superbrain.app.application.digest.models import DigestSourceArticle
from superbrain.app.domain.models import Article


def _source(canonical_url: str, title: str) -> DigestSourceArticle:
    now = datetime.now(UTC)
    article = Article(
        id=uuid4(),
        source_url=canonical_url,
        canonical_url=canonical_url,
        domain="example.com",
        title=title,
        author=None,
        published_at=None,
        content="content",
        content_hash=str(uuid4()),
        extraction_quality_score=0.8,
        extraction_notes="test",
        created_at=now,
    )
    return DigestSourceArticle(article=article, topic_id=None, topic_name="Uncategorized")


def test_canonical_url_deduper_removes_duplicates() -> None:
    """Deduper should keep only first article per canonical URL."""

    sources = [
        _source("https://example.com/x", "One"),
        _source("https://example.com/x", "Two"),
        _source("https://example.com/y", "Three"),
    ]
    deduped = CanonicalUrlDigestDeduper().dedupe(sources)

    assert len(deduped) == 2
    assert deduped[0].article.title == "One"
