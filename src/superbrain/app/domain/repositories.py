"""Repository interfaces for domain persistence boundaries."""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from superbrain.app.domain.models import (
    Article,
    ArticleChunk,
    Digest,
    IngestionJob,
    IngestionStatus,
    QueryLogEntry,
    StoredChunk,
    TopicDefinition,
    TopicMatch,
    TopicVersion,
)


class ArticleRepository(Protocol):
    """Persistence contract for article records."""

    def save(self, article: Article) -> Article:
        """Persist and return an article."""

    def save_chunks(self, chunks: list[ArticleChunk]) -> list[ArticleChunk]:
        """Persist and return article chunks."""

    def save_raw_snapshot(self, article_id: UUID, raw_html: str) -> None:
        """Persist a raw snapshot for an article."""

    def get(self, article_id: UUID) -> Article | None:
        """Fetch an article by ID."""

    def list_articles(
        self,
        limit: int = 100,
        article_ids: list[UUID] | None = None,
    ) -> list[Article]:
        """List recent articles or scoped article IDs."""

    def list_between(self, start: datetime, end: datetime) -> list[Article]:
        """List articles whose created time falls within [start, end)."""

    def get_by_source_url(self, source_url: str) -> Article | None:
        """Fetch an article by source URL."""

    def get_by_canonical_url(self, canonical_url: str) -> Article | None:
        """Fetch an article by canonical URL."""

    def get_by_content_hash(self, content_hash: str) -> Article | None:
        """Fetch an article by content hash."""


class ArticleTopicMatchRepository(Protocol):
    """Persistence contract for article-topic classification matches."""

    def replace_for_article(self, article_id: UUID, matches: list[TopicMatch]) -> list[TopicMatch]:
        """Replace existing matches for an article and return saved matches."""

    def list_for_article(self, article_id: UUID) -> list[TopicMatch]:
        """List saved matches for an article."""

    def list_for_articles(self, article_ids: list[UUID]) -> list[TopicMatch]:
        """List saved matches across multiple article IDs."""


class RetrievalRepository(Protocol):
    """Persistence contract for retrieval candidate sources."""

    def list_chunks(self, limit: int = 1000) -> list[StoredChunk]:
        """Return chunk records joined with article metadata."""

    def lexical_scores(self, query: str, limit: int = 200) -> dict[str, float]:
        """Return lexical relevance scores keyed by chunk ID."""


class TopicRepository(Protocol):
    """Persistence contract for topic definitions and versions."""

    def create(self, topic: TopicDefinition, version: TopicVersion) -> TopicDefinition:
        """Create a topic and its initial version."""

    def update(self, topic: TopicDefinition, version: TopicVersion) -> TopicDefinition:
        """Persist new current version for a topic."""

    def set_inactive(self, topic_id: UUID) -> TopicDefinition:
        """Mark topic as inactive and return updated record."""

    def get(self, topic_id: UUID) -> TopicDefinition | None:
        """Fetch topic by ID."""

    def list_all(self, active_only: bool = False) -> list[TopicDefinition]:
        """List topics, optionally filtering active only."""

    def get_latest_version(self, topic_id: UUID) -> TopicVersion | None:
        """Fetch latest version for a topic."""

    def list_active_with_latest_versions(self) -> list[tuple[TopicDefinition, TopicVersion]]:
        """List active topics paired with latest versions."""


class IngestionJobRepository(Protocol):
    """Persistence contract for ingestion jobs."""

    def create(self, job: IngestionJob) -> IngestionJob:
        """Persist and return an ingestion job."""

    def update_status(
        self,
        job_id: UUID,
        status: IngestionStatus,
        *,
        error_message: str | None = None,
        article_id: UUID | None = None,
    ) -> IngestionJob:
        """Update status and optional metadata for an ingestion job."""

    def get(self, job_id: UUID) -> IngestionJob | None:
        """Fetch an ingestion job by ID."""

    def list_failed(self, limit: int = 50) -> list[IngestionJob]:
        """List failed ingestion jobs for retry workflows."""


class QueryLogRepository(Protocol):
    """Persistence contract for query logs."""

    def record(self, entry: QueryLogEntry) -> None:
        """Persist query execution metadata."""


class ModelCallLogRepository(Protocol):
    """Persistence contract for model and embedding call logs."""

    def record(
        self,
        *,
        provider: str,
        model_name: str,
        request_type: str,
        prompt_template: str | None,
        started_at: datetime,
        finished_at: datetime,
        duration_ms: float,
        status: str,
        retries: int,
        error_metadata: str | None,
        related_entity_id: str | None,
    ) -> None:
        """Persist model-call execution details."""


class DigestRepository(Protocol):
    """Persistence contract for digest runs."""

    def create_run(self, run_date: datetime) -> Digest:
        """Create a running digest run for a target date."""

    def complete_run(self, digest: Digest) -> Digest:
        """Persist completed digest payload and return saved run."""

    def fail_run(self, digest_id: UUID, error_message: str) -> Digest:
        """Mark a digest run as failed and return resulting run."""

    def get_latest(self) -> Digest | None:
        """Return most recent digest run if present."""

    def list_recent(self, limit: int = 20) -> list[Digest]:
        """List most recent digest runs."""
