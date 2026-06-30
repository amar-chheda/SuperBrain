"""Add raw_text column to ingestion_jobs.

Revision ID: 003
Revises: 002
Create Date: 2026-05-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add raw_text column to store crawled article text on the job."""
    op.add_column("ingestion_jobs", sa.Column("raw_text", sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove raw_text column from ingestion_jobs."""
    op.drop_column("ingestion_jobs", "raw_text")
