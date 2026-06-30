"""Enable pgvector extension.

Revision ID: 001
Revises:
Create Date: 2026-05-05
"""

from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable the pgvector extension for vector similarity search."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Remove the pgvector extension."""
    op.execute("DROP EXTENSION IF EXISTS vector")
