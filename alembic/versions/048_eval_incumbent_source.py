"""Let a model-swap run read the incumbent's decisions out of the transcript.

A run used to call both models on every sampled turn, so a hundred-turn run
made two hundred live calls and every new candidate re-bought the incumbent's
answers to the same turns. ``incumbent_source`` picks where they come from
instead; ``llm_eval.types.IncumbentSource`` is what each mode does and does
not measure.

The two counters are what keeps the new default honest:
``baseline_turns_unavailable`` is how many sampled turns had no decision to
read and left every comparison, and ``historic_other_config_calls`` how many
agent calls in this user's usage log, over the window the sampled turns fall
in, ran on a model the run does not name.

Server defaults describe what old rows did: every existing run replayed, with
nothing unavailable, and every existing turn was a live call.

Revision ID: 048
Revises: 047
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "048"
down_revision: str | None = "047"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "llm_eval_runs",
        sa.Column("incumbent_source", sa.String(16), server_default="replay", nullable=False),
    )
    for column in ("baseline_turns_unavailable", "historic_other_config_calls"):
        op.add_column(
            "llm_eval_runs",
            sa.Column(column, sa.Integer(), server_default="0", nullable=False),
        )
    op.add_column(
        "llm_eval_turn_results",
        sa.Column("baseline_source", sa.String(16), server_default="live", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("llm_eval_turn_results", "baseline_source")
    op.drop_column("llm_eval_runs", "historic_other_config_calls")
    op.drop_column("llm_eval_runs", "baseline_turns_unavailable")
    op.drop_column("llm_eval_runs", "incumbent_source")
