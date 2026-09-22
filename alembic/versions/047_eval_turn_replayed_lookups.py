"""Record the lookups a model-swap replay fed back before each side's decision.

A replay now continues through a read-only call when the live turn made the
same call, by feeding back the result the turn recorded, so a model that
looks something up before it writes is scored on the write rather than on the
lookup. These columns keep what each side read on the way, so the report can
show it.

Envelope-encrypted at the application layer (``EncryptedString``) like every
other text column on this table, so plain ``TEXT`` here. The empty default
covers every row written before the replay continued past a first decision,
which the report renders as "decided on its first round".

Revision ID: 047
Revises: 046
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "047"
down_revision: str | None = "046"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for side in ("baseline", "candidate"):
        op.add_column(
            "llm_eval_turn_results",
            sa.Column(f"{side}_replayed_lookups", sa.Text(), server_default="", nullable=False),
        )


def downgrade() -> None:
    for side in ("baseline", "candidate"):
        op.drop_column("llm_eval_turn_results", f"{side}_replayed_lookups")
