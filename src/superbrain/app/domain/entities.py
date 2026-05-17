"""Core domain entities for Superbrain.

Pure Python dataclasses with no SQLAlchemy or FastAPI dependencies.
These are the objects the system reasons about at the business logic level.
All infrastructure layers translate to/from these types.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal
from uuid import UUID


@dataclass
class Article:
    """A web page that has been ingested into the knowledge base."""

    id: UUID
    url: str
    canonical_url: str
    content_hash: str
    raw_text: str
    title: str | None
    author: str | None
    published_at: datetime | None
    ingested_at: datetime
    status: Literal["pending", "processing", "succeeded", "failed"]


@dataclass
class Chunk:
    """A text segment derived from an Article, ready for embedding."""

    id: UUID
    article_id: UUID
    content: str
    chunk_index: int
    strategy: Literal["semantic", "recursive", "fixed"]
    token_count: int
    embedding: list[float] | None = None


@dataclass
class Topic:
    """A user-defined classification topic for organising articles."""

    id: UUID
    name: str
    version: int
    description: str
    examples: list[str] = field(default_factory=list)
    priority: int = 0
    status: Literal["active", "archived"] = "active"


@dataclass
class QueryLog:
    """A record of a question-answer exchange."""

    id: UUID
    question: str
    answer: str | None
    evidence_chunk_ids: list[UUID]
    retrieval_latency_ms: int
    answer_latency_ms: int
    aborted: bool
    abort_reason: str | None
    created_at: datetime


@dataclass
class IngestionJob:
    """A background job that tracks the ingestion of a piece of content."""

    id: UUID
    input_type: Literal["url", "pdf", "text"]
    input_value: str
    status: Literal["pending", "processing", "succeeded", "failed"]
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    source: Literal["api", "telegram", "cli", "scheduler"] = "api"
    raw_text: str | None = None


@dataclass
class ArticleTopicMatch:
    """A record linking an article to a topic that it was classified into."""

    id: UUID
    article_id: UUID
    topic_id: UUID
    topic_version: int
    confidence: Literal["high", "medium", "low"]
    reason: str
    classified_at: datetime


@dataclass
class DigestRun:
    """A record of one daily digest generation run."""

    id: UUID
    date_label: date
    status: Literal["running", "succeeded", "failed"]
    triggered_by: Literal["scheduler", "manual", "api"]
    started_at: datetime
    article_count: int = 0
    section_count: int = 0
    finished_at: datetime | None = None
    error_message: str | None = None


@dataclass
class DigestItem:
    """One topic section within a digest run."""

    id: UUID
    run_id: UUID
    topic_id: UUID
    topic_name: str
    summary: str
    article_ids: list[UUID]
    article_urls: list[str]
    article_titles: list[str]
    position: int
    created_at: datetime


@dataclass
class ModelCallLog:
    """An audit record for every call made to a local LLM or embedding model."""

    id: UUID
    provider: str
    model_name: str
    request_type: Literal["embedding", "extraction", "classification", "qa", "digest"]
    prompt_template: str
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    status: Literal["success", "failed"]
    retries: int
    error_metadata: dict | None
    related_entity_id: UUID | None
