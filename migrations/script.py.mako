"""${message}"""

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    """Apply schema upgrades."""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Apply schema downgrades."""
    ${downgrades if downgrades else "pass"}
