"""SQLAlchemy ORM model declarations.

Defines the declarative Base and all table mappings.
Alembic reads Base.metadata to auto-generate migrations.
"""

from datetime import date, datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class IngestionJobModel(Base):
    """ORM mapping for the ingestion_jobs table."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    input_type: Mapped[str] = mapped_column(String(10), nullable=False)
    input_value: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="api")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ArticleModel(Base):
    """ORM mapping for the articles table."""

    __tablename__ = "articles"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")


class ChunkModel(Base):
    """ORM mapping for the chunks table."""

    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    article_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy: Mapped[str] = mapped_column(String(20), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ModelCallLogModel(Base):
    """ORM mapping for the model_call_logs table."""

    __tablename__ = "model_call_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    request_type: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_template: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    related_entity_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True
    )
    prompt_input: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TopicModel(Base):
    """ORM mapping for the topics table."""

    __tablename__ = "topics"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_topics_name_version"),)

    id: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    examples: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ArticleTopicMatchModel(Base):
    """ORM mapping for the article_topic_matches table."""

    __tablename__ = "article_topic_matches"
    __table_args__ = (
        UniqueConstraint("article_id", "topic_id", name="uq_match_article_topic"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    article_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("topics.id"),
        nullable=False,
        index=True,
    )
    topic_version: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[str] = mapped_column(String(10), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DigestRunModel(Base):
    """ORM mapping for the digest_runs table."""

    __tablename__ = "digest_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    date_label: Mapped[date] = mapped_column(sa.Date(), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    section_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    triggered_by: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class DigestItemModel(Base):
    """ORM mapping for the digest_items table."""

    __tablename__ = "digest_items"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("digest_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("topics.id"),
        nullable=False,
    )
    topic_name: Mapped[str] = mapped_column(String(100), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    article_ids: Mapped[list] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False, default=list
    )
    article_urls: Mapped[list] = mapped_column(
        ARRAY(sa.Text()), nullable=False, default=list
    )
    article_titles: Mapped[list] = mapped_column(
        ARRAY(sa.Text()), nullable=False, default=list
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class QueryLogModel(Base):
    """ORM mapping for the query_logs table."""

    __tablename__ = "query_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_chunk_ids: Mapped[list] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False, default=list
    )
    retrieval_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    answer_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    aborted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    abort_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_trace: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
