"""Add query_logs table and FTS index on chunks

Revision ID: 005
Revises: b2d65e7d401c
Create Date: 2026-05-09 20:45:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005"
down_revision: Union[str, None] = "b2d65e7d401c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "query_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column(
            "evidence_chunk_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("retrieval_latency_ms", sa.Integer(), nullable=True),
        sa.Column("answer_latency_ms", sa.Integer(), nullable=True),
        sa.Column("aborted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("abort_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Add generated tsvector column for full-text search on chunks
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin(content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_fts")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
    op.drop_table("query_logs")
