"""Abstract repository contracts for domain entities.

Defines the interface the infrastructure layer must implement.
No concrete implementations live here — only the contracts that
decouple the application layer from database specifics.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Literal
from uuid import UUID

from superbrain.app.domain.entities import (
    Article,
    ArticleTopicMatch,
    Chunk,
    DigestItem,
    DigestRun,
    IngestionJob,
    ModelCallLog,
    QueryLog,
    Topic,
)


class ArticleRepository(ABC):
    """Contract for persisting and retrieving Article entities."""

    @abstractmethod
    async def save(self, article: Article) -> None:
        """Persist an article, inserting or updating as needed.

        Args:
            article: The article to save.
        """

    @abstractmethod
    async def find_by_hash(self, content_hash: str) -> Article | None:
        """Find an article by its content hash for deduplication.

        Args:
            content_hash: SHA-256 hash of the article's raw text.

        Returns:
            The matching article, or None if not found.
        """

    @abstractmethod
    async def find_by_id(self, article_id: UUID) -> Article | None:
        """Find an article by its primary key.

        Args:
            article_id: UUID of the article.

        Returns:
            The article, or None if not found.
        """

    @abstractmethod
    async def update_status(
        self,
        article_id: UUID,
        status: Literal["pending", "processing", "succeeded", "failed"],
    ) -> None:
        """Update an article's processing status.

        Args:
            article_id: UUID of the article to update.
            status: The new status value.
        """

    @abstractmethod
    async def list_all_active(self) -> list[Article]:
        """List all articles with status='succeeded'.

        Returns:
            All successfully ingested articles.
        """

    @abstractmethod
    async def list_by_date(self, target_date: date) -> list[Article]:
        """List all articles ingested on a given date.

        Args:
            target_date: The date to filter by (uses ingested_at).

        Returns:
            List of articles ingested on that date, possibly empty.
        """


class ChunkRepository(ABC):
    """Contract for persisting and retrieving Chunk entities."""

    @abstractmethod
    async def save_many(self, chunks: list[Chunk]) -> None:
        """Persist a batch of chunks.

        Args:
            chunks: The chunks to save. Must all belong to the same article.
        """

    @abstractmethod
    async def find_by_article(self, article_id: UUID) -> list[Chunk]:
        """Find all chunks belonging to an article.

        Args:
            article_id: UUID of the parent article.

        Returns:
            Ordered list of chunks for the article, possibly empty.
        """


class TopicRepository(ABC):
    """Contract for persisting and retrieving Topic entities."""

    @abstractmethod
    async def save(self, topic: Topic) -> None:
        """Persist a topic, inserting or updating as needed.

        Args:
            topic: The topic to save.
        """

    @abstractmethod
    async def list_active(self) -> list[Topic]:
        """List all topics with status='active'.

        Returns:
            List of active topics ordered by priority descending.
        """

    @abstractmethod
    async def list_all(self, include_archived: bool = False) -> list[Topic]:
        """List all topics, optionally including archived ones.

        Args:
            include_archived: If True, include archived topics.

        Returns:
            Topics ordered by priority descending.
        """

    @abstractmethod
    async def set_status(
        self, topic_id: UUID, status: Literal["active", "archived"]
    ) -> None:
        """Set a topic's status.

        Args:
            topic_id: UUID of the topic to update.
            status: New status value.
        """

    @abstractmethod
    async def find_by_id(self, topic_id: UUID) -> Topic | None:
        """Find a topic by its primary key.

        Args:
            topic_id: UUID of the topic.

        Returns:
            The topic, or None if not found.
        """


class ModelCallLogRepository(ABC):
    """Contract for persisting model call audit logs."""

    @abstractmethod
    async def save(self, log: ModelCallLog) -> None:
        """Persist a model call log entry.

        Args:
            log: The model call log to save.
        """


class QueryLogRepository(ABC):
    """Contract for persisting QA query logs."""

    @abstractmethod
    async def save(self, log: QueryLog) -> None:
        """Persist a query log entry.

        Args:
            log: The query log to save.
        """


class IngestionJobRepository(ABC):
    """Contract for persisting and retrieving IngestionJob entities."""

    @abstractmethod
    async def save(self, job: IngestionJob) -> None:
        """Persist a new ingestion job.

        Args:
            job: The job to save.
        """

    @abstractmethod
    async def find_by_id(self, job_id: UUID) -> IngestionJob | None:
        """Find a job by its primary key.

        Args:
            job_id: UUID of the job.

        Returns:
            The job, or None if not found.
        """

    @abstractmethod
    async def update_status(
        self,
        job_id: UUID,
        status: Literal["pending", "processing", "succeeded", "failed"],
        error_message: str | None = None,
    ) -> None:
        """Update the status (and optionally error_message) of a job.

        Args:
            job_id: UUID of the job to update.
            status: The new status value.
            error_message: Optional error detail when status is 'failed'.
        """


class ArticleTopicMatchRepository(ABC):
    """Contract for persisting and retrieving ArticleTopicMatch entities."""

    @abstractmethod
    async def save_many(self, matches: list[ArticleTopicMatch]) -> None:
        """Persist a batch of topic matches.

        Args:
            matches: The matches to save.
        """

    @abstractmethod
    async def delete_by_article(self, article_id: UUID) -> None:
        """Delete all matches for a given article (used before reclassification).

        Args:
            article_id: UUID of the article whose matches should be deleted.
        """

    @abstractmethod
    async def upsert_for_topic(
        self,
        article_id: UUID,
        topic_id: UUID,
        matches: list[ArticleTopicMatch],
    ) -> None:
        """Replace the match for a specific article-topic pair.

        Args:
            article_id: UUID of the article.
            topic_id: UUID of the topic being reclassified.
            matches: New matches (may be empty if article no longer matches).
        """

    @abstractmethod
    async def find_by_article(self, article_id: UUID) -> list[ArticleTopicMatch]:
        """Return all matches for a given article.

        Args:
            article_id: UUID of the article.

        Returns:
            All topic matches for the article, possibly empty.
        """

    @abstractmethod
    async def list_by_article_ids(
        self, article_ids: list[UUID]
    ) -> list[ArticleTopicMatch]:
        """Return all matches for a batch of article IDs.

        Args:
            article_ids: UUIDs of the articles to fetch matches for.

        Returns:
            All topic matches for the given articles, possibly empty.
        """


class DigestRepository(ABC):
    """Contract for persisting and retrieving digest runs and items."""

    @abstractmethod
    async def save_run(self, run: DigestRun) -> None:
        """Persist a new digest run record."""

    @abstractmethod
    async def update_run(
        self,
        run_id: UUID,
        *,
        status: str,
        article_count: int = 0,
        section_count: int = 0,
        finished_at: object = None,
        error_message: str | None = None,
    ) -> None:
        """Update mutable fields of an existing digest run."""

    @abstractmethod
    async def save_items(self, items: list[DigestItem]) -> None:
        """Persist a batch of digest items."""

    @abstractmethod
    async def list_runs(self, limit: int = 30) -> list[DigestRun]:
        """Return the most recent digest runs, newest first."""

    @abstractmethod
    async def find_run_by_id(self, run_id: UUID) -> DigestRun | None:
        """Return a digest run by ID, or None if not found."""

    @abstractmethod
    async def find_items_by_run(self, run_id: UUID) -> list[DigestItem]:
        """Return all items for a given digest run, ordered by position."""
