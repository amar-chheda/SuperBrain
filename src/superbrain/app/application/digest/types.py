"""Intermediate data structures used only within the digest pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from superbrain.app.domain.entities import Article, ArticleTopicMatch, Topic


@dataclass
class ArticleWithTopics:
    """An article joined with its topic matches — the unit the digest works with."""

    id: UUID
    url: str
    canonical_url: str
    raw_text: str
    title: str | None
    ingested_at: datetime
    status: str
    topic_matches: list[ArticleTopicMatch] = field(default_factory=list)

    @classmethod
    def from_article(
        cls, article: Article, matches: list[ArticleTopicMatch]
    ) -> "ArticleWithTopics":
        return cls(
            id=article.id,
            url=article.url,
            canonical_url=article.canonical_url,
            raw_text=article.raw_text,
            title=article.title,
            ingested_at=article.ingested_at,
            status=article.status,
            topic_matches=matches,
        )


@dataclass
class TopicGroup:
    """A topic paired with all articles that matched it."""

    topic: Topic
    articles: list[ArticleWithTopics] = field(default_factory=list)
