"""Digest workflow data models."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from superbrain.app.domain.models import Article


@dataclass(slots=True, frozen=True)
class DigestSourceArticle:
    """Article with topic metadata used for digest section generation."""

    article: Article
    topic_id: UUID | None
    topic_name: str


@dataclass(slots=True, frozen=True)
class DigestSectionDraft:
    """Generated digest section for one topic."""

    topic_id: UUID | None
    topic_name: str
    summary: str
    source_urls: tuple[str, ...]
    citation_article_ids: tuple[UUID, ...]


@dataclass(slots=True, frozen=True)
class GenerateDigestInput:
    """Input payload for digest generation."""

    run_date: datetime
    notify: bool = False
