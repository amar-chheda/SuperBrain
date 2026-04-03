"""Core domain models for Superbrain."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4


class IngestionStatus(StrEnum):
    """Lifecycle states for ingestion jobs."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TopicStatus(StrEnum):
    """Lifecycle status for topic definitions."""

    ACTIVE = "active"
    INACTIVE = "inactive"


class DigestStatus(StrEnum):
    """Lifecycle status for digest run execution."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class Citation:
    """Citation reference used to ground generated answers."""

    article_id: UUID
    article_title: str
    article_url: str
    chunk_id: UUID
    snippet: str
    rank: int
    score: float


@dataclass(slots=True, frozen=True)
class ArticleChunk:
    """Chunk of normalized article content."""

    id: UUID
    article_id: UUID
    index: int
    text: str
    token_count: int
    embedding: list[float]
    char_start: int
    char_end: int


@dataclass(slots=True, frozen=True)
class StoredChunk:
    """Chunk with article metadata used by retrieval services."""

    chunk_id: UUID
    article_id: UUID
    article_title: str
    article_url: str
    chunk_text: str
    embedding: list[float]


@dataclass(slots=True, frozen=True)
class Article:
    """Normalized article entity with provenance and extraction metadata."""

    id: UUID
    source_url: str
    canonical_url: str
    domain: str
    title: str
    author: str | None
    published_at: datetime | None
    content: str
    content_hash: str
    extraction_quality_score: float
    extraction_notes: str
    created_at: datetime


@dataclass(slots=True, frozen=True)
class TopicVersion:
    """Versioned classification definition for a topic."""

    id: UUID
    topic_id: UUID
    version: int
    description: str
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    created_at: datetime


@dataclass(slots=True, frozen=True)
class TopicDefinition:
    """User-defined topic with current version pointer and metadata."""

    id: UUID
    name: str
    status: TopicStatus
    priority: int
    current_version_id: UUID
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True, frozen=True)
class TopicMatch:
    """Persisted topic-classification decision for an article."""

    article_id: UUID
    topic_id: UUID
    topic_version_id: UUID
    score: float
    rationale: str
    disqualifiers: tuple[str, ...]
    classified_at: datetime


@dataclass(slots=True, frozen=True)
class IngestionJob:
    """Work item representing asynchronous article ingestion."""

    id: UUID
    source_url: str
    canonical_url: str
    status: IngestionStatus
    requested_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None = None
    article_id: UUID | None = None


@dataclass(slots=True, frozen=True)
class QueryRequest:
    """Query request from a user."""

    id: UUID
    question: str
    requested_at: datetime


@dataclass(slots=True, frozen=True)
class QueryResponse:
    """Grounded answer with supporting citations."""

    request_id: UUID
    answer: str
    citations: tuple[Citation, ...]
    supported: bool
    created_at: datetime


@dataclass(slots=True, frozen=True)
class QueryLogEntry:
    """Persisted query execution metadata for observability and evaluation."""

    query_request: QueryRequest
    query_response: QueryResponse
    retrieval_ms: float
    generation_ms: float
    evidence_chunk_ids: tuple[UUID, ...]


@dataclass(slots=True, frozen=True)
class DigestItem:
    """Single topic section included in a digest run."""

    topic_id: UUID | None
    topic_name: str
    summary: str
    source_urls: tuple[str, ...]
    citation_article_ids: tuple[UUID, ...]


@dataclass(slots=True, frozen=True)
class Digest:
    """Digest aggregate containing selected items."""

    id: UUID
    run_date: datetime
    status: DigestStatus
    created_at: datetime
    items: tuple[DigestItem, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class NewIngestionJob:
    """Factory input for creating an ingestion job."""

    source_url: str
    canonical_url: str

    def to_job(self) -> IngestionJob:
        """Create a new pending ingestion job instance."""

        now = datetime.now(UTC)
        return IngestionJob(
            id=uuid4(),
            source_url=self.source_url,
            canonical_url=self.canonical_url,
            status=IngestionStatus.PENDING,
            requested_at=now,
            started_at=None,
            finished_at=None,
        )
