"""Record cache writes per lifetime on ``llm_usage_logs``.

Anthropic splits ``cache_creation_input_tokens`` into
``usage.cache_creation.ephemeral_5m_input_tokens`` and
``ephemeral_1h_input_tokens``. A 1-hour write bills 2x base input and a
5-minute write 1.25x, so the split is what shows which lifetime a call was
billed at and whether a gateway honoured the requested 1h.

Both columns are nullable with no default. Existing rows and providers that
do not report the split stay NULL, which is not the same as zero.

Revision ID: 049
Revises: 048
Create Date: 2026-09-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "049"
down_revision: str | None = "048"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "llm_usage_logs",
        sa.Column("cache_creation_5m_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "llm_usage_logs",
        sa.Column("cache_creation_1h_input_tokens", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_usage_logs", "cache_creation_1h_input_tokens")
    op.drop_column("llm_usage_logs", "cache_creation_5m_input_tokens")
