"""Add prompt_input/response_output to model_call_logs and retrieval_trace to query_logs.

Revision ID: 007
Revises: 006
Create Date: 2026-05-28
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_call_logs", sa.Column("prompt_input", sa.Text(), nullable=True))
    op.add_column("model_call_logs", sa.Column("response_output", sa.Text(), nullable=True))
    op.add_column("query_logs", sa.Column("retrieval_trace", sa.dialects.postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("query_logs", "retrieval_trace")
    op.drop_column("model_call_logs", "response_output")
    op.drop_column("model_call_logs", "prompt_input")
