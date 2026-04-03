"""Create pgvector extension and ingestion foundation tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_core_articles"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Apply schema upgrades."""

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "articles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False, unique=True),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("extraction_quality_score", sa.Float(), nullable=False),
        sa.Column("extraction_notes", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_articles_canonical_url", "articles", ["canonical_url"], unique=False)
    op.create_index("ix_articles_content_hash", "articles", ["content_hash"], unique=False)

    op.create_table(
        "article_raw_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("raw_html", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_article_raw_snapshots_article_id",
        "article_raw_snapshots",
        ["article_id"],
        unique=False,
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="SET NULL"),
    )
    op.create_index(
        "ix_ingestion_jobs_canonical_url",
        "ingestion_jobs",
        ["canonical_url"],
        unique=False,
    )
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"], unique=False)
    op.create_index("ix_ingestion_jobs_article_id", "ingestion_jobs", ["article_id"], unique=False)

    op.create_table(
        "article_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(16), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("article_id", "chunk_index", name="uq_article_chunk_order"),
    )
    op.create_index("ix_article_chunks_article_id", "article_chunks", ["article_id"], unique=False)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_article_chunks_text_fts "
        "ON article_chunks USING GIN (to_tsvector('english', text))"
    )

    op.create_table(
        "topics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("current_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_topics_status", "topics", ["status"], unique=False)

    op.create_table(
        "topic_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("positive_examples", sa.JSON(), nullable=False),
        sa.Column("negative_examples", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("topic_id", "version_number", name="uq_topic_version"),
    )
    op.create_index("ix_topic_versions_topic_id", "topic_versions", ["topic_id"], unique=False)

    op.create_table(
        "article_topic_matches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topic_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("disqualifiers", sa.JSON(), nullable=False),
        sa.Column("classified_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_version_id"], ["topic_versions.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("article_id", "topic_id", name="uq_article_topic_match"),
    )
    op.create_index(
        "ix_article_topic_matches_article_id",
        "article_topic_matches",
        ["article_id"],
        unique=False,
    )
    op.create_index(
        "ix_article_topic_matches_topic_id",
        "article_topic_matches",
        ["topic_id"],
        unique=False,
    )

    op.create_table(
        "query_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("supported", sa.Boolean(), nullable=False),
        sa.Column("retrieval_ms", sa.Float(), nullable=False),
        sa.Column("generation_ms", sa.Float(), nullable=False),
        sa.Column("evidence_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_query_logs_request_id", "query_logs", ["request_id"], unique=False)

    op.create_table(
        "model_call_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("request_type", sa.String(length=64), nullable=False),
        sa.Column("prompt_template", sa.String(length=255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("retries", sa.Integer(), nullable=False),
        sa.Column("error_metadata", sa.Text(), nullable=True),
        sa.Column("related_entity_id", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_model_call_logs_provider", "model_call_logs", ["provider"], unique=False)
    op.create_index(
        "ix_model_call_logs_request_type",
        "model_call_logs",
        ["request_type"],
        unique=False,
    )
    op.create_index("ix_model_call_logs_status", "model_call_logs", ["status"], unique=False)

    op.create_table(
        "digest_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("run_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index("ix_digest_runs_run_date", "digest_runs", ["run_date"], unique=False)
    op.create_index("ix_digest_runs_status", "digest_runs", ["status"], unique=False)

    op.create_table(
        "digest_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("digest_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("topic_name", sa.String(length=255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("source_urls", sa.JSON(), nullable=False),
        sa.Column("citation_article_ids", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["digest_run_id"], ["digest_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="SET NULL"),
    )
    op.create_index(
        "ix_digest_items_digest_run_id",
        "digest_items",
        ["digest_run_id"],
        unique=False,
    )

    op.create_table(
        "scheduled_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False, unique=True),
        sa.Column("cron_expression", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "scheduled_job_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("job_name", sa.String(length=128), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_scheduled_job_runs_job_name",
        "scheduled_job_runs",
        ["job_name"],
        unique=False,
    )
    op.create_index("ix_scheduled_job_runs_status", "scheduled_job_runs", ["status"], unique=False)


def downgrade() -> None:
    """Apply schema downgrades."""

    op.drop_index("ix_scheduled_job_runs_status", table_name="scheduled_job_runs")
    op.drop_index("ix_scheduled_job_runs_job_name", table_name="scheduled_job_runs")
    op.drop_table("scheduled_job_runs")

    op.drop_table("scheduled_jobs")

    op.drop_index("ix_digest_items_digest_run_id", table_name="digest_items")
    op.drop_table("digest_items")

    op.drop_index("ix_digest_runs_status", table_name="digest_runs")
    op.drop_index("ix_digest_runs_run_date", table_name="digest_runs")
    op.drop_table("digest_runs")

    op.drop_index("ix_article_chunks_article_id", table_name="article_chunks")
    op.execute("DROP INDEX IF EXISTS ix_article_chunks_text_fts")
    op.drop_table("article_chunks")

    op.drop_index("ix_query_logs_request_id", table_name="query_logs")
    op.drop_table("query_logs")

    op.drop_index("ix_model_call_logs_status", table_name="model_call_logs")
    op.drop_index("ix_model_call_logs_request_type", table_name="model_call_logs")
    op.drop_index("ix_model_call_logs_provider", table_name="model_call_logs")
    op.drop_table("model_call_logs")

    op.drop_index("ix_article_topic_matches_topic_id", table_name="article_topic_matches")
    op.drop_index("ix_article_topic_matches_article_id", table_name="article_topic_matches")
    op.drop_table("article_topic_matches")

    op.drop_index("ix_topic_versions_topic_id", table_name="topic_versions")
    op.drop_table("topic_versions")

    op.drop_index("ix_topics_status", table_name="topics")
    op.drop_table("topics")

    op.drop_index("ix_ingestion_jobs_article_id", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_canonical_url", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")

    op.drop_index("ix_article_raw_snapshots_article_id", table_name="article_raw_snapshots")
    op.drop_table("article_raw_snapshots")

    op.drop_index("ix_articles_content_hash", table_name="articles")
    op.drop_index("ix_articles_canonical_url", table_name="articles")
    op.drop_table("articles")
