"""Remember where a mid-session trim stubbed old read results.

A trim that sheds stale read results instead of dropping whole turns has to
leave the next turn rendering the same bytes, or the history cache it wrote
is never read. ``sessions.history_stub_seq`` records the first row whose
tool results stay verbatim; every read result above the trim watermark and
below it renders as a stub (``prompt_epoch.build_history_view``). NULL means
no trim has stubbed anything.

Revision ID: 051
Revises: 050
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "051"
down_revision: str | None = "050"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("history_stub_seq", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("sessions", "history_stub_seq")
