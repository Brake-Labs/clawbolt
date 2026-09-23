"""Record how a model comparison run rendered each turn's history.

A run can now replay turns with the cold-start history rebuild on or off
(``services.model_comparison.types.HistoryMode``), so the operator can read
what the rebuild changes. Existing rows replayed every row verbatim, which is
what the ``full`` server default says.

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
        "model_comparison_runs",
        sa.Column("history_mode", sa.String(length=32), nullable=False, server_default="full"),
    )


def downgrade() -> None:
    op.drop_column("model_comparison_runs", "history_mode")
