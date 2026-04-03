"""Unit tests for digest section grouping logic."""

from datetime import UTC, datetime
from uuid import uuid4

from superbrain.app.application.digest.generator import DigestGenerator
from superbrain.app.application.digest.models import DigestSourceArticle
from superbrain.app.domain.models import Article


def _article(title: str, url: str) -> Article:
    now = datetime.now(UTC)
    return Article(
        id=uuid4(),
        source_url=url,
        canonical_url=url,
        domain="example.com",
        title=title,
        author=None,
        published_at=None,
        content="content",
        content_hash=str(uuid4()),
        extraction_quality_score=0.9,
        extraction_notes="test",
        created_at=now,
    )


def test_digest_generator_groups_by_topic() -> None:
    """Digest generator should build one section per topic."""

    work_topic_id = uuid4()
    sources = [
        DigestSourceArticle(
            article=_article("A", "https://e.com/a"),
            topic_id=work_topic_id,
            topic_name="Work",
        ),
        DigestSourceArticle(
            article=_article("B", "https://e.com/b"),
            topic_id=uuid4(),
            topic_name="Personal",
        ),
        DigestSourceArticle(
            article=_article("C", "https://e.com/c"),
            topic_id=work_topic_id,
            topic_name="Work",
        ),
    ]

    sections = DigestGenerator().generate_sections(sources)

    assert len(sections) == 2
    work = next(section for section in sections if section.topic_name == "Work")
    assert len(work.source_urls) == 2
