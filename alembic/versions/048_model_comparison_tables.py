"""Replace the model-swap evaluator's tables with the comparison report's.

The evaluator replayed every turn through two models and converted the diff
into a recommendation. What replaces it replays one model, the candidate, and
lays its decision beside what production actually did, which is already in
the transcript. That removes the second live call, the judge, and the verdict,
so most of the old schema has nothing left to hold: the ``baseline_*`` per-turn
columns, ``judge_verdict`` and ``judge_rationale``, ``agreement``, the run's
``recommendation`` and its judge target.

Dropped and recreated rather than altered. There are no rows in production
(the operator deleted every run before this landed), the column overlap is
under half, and the rename from evaluation to comparison would have left two
tables whose names described the thing they no longer are. A compatibility
shim for runs nobody has would be dead code on the day it shipped.

Text columns on the turn table are envelope-encrypted at the application
layer (``EncryptedString``), so they are plain ``TEXT`` here and carry
ciphertext at rest. Token counts and latencies stay plaintext: they are
measurements, not user content.

``downgrade`` recreates the evaluator's tables empty, which is the whole of
what it can restore: it does not recover rows the upgrade dropped, and the
code that read them is gone.

Revision ID: 048
Revises: 047
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "048"
down_revision: str | None = "047"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _drop_eval_tables() -> None:
    op.drop_index("ix_llm_eval_turn_results_run_id", table_name="llm_eval_turn_results")
    op.drop_table("llm_eval_turn_results")
    op.drop_index("ix_llm_eval_runs_public_id", table_name="llm_eval_runs")
    op.drop_index("ix_llm_eval_runs_created_at", table_name="llm_eval_runs")
    op.drop_index("ix_llm_eval_runs_status", table_name="llm_eval_runs")
    op.drop_index("ix_llm_eval_runs_user_id", table_name="llm_eval_runs")
    op.drop_table("llm_eval_runs")


def _create_eval_tables() -> None:
    """The evaluator's schema as migrations 042 through 047 left it."""
    op.create_table(
        "llm_eval_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("created_by_admin_id", sa.String(), nullable=True),
        sa.Column("baseline_endpoint", sa.String(length=64), server_default="", nullable=False),
        sa.Column("baseline_provider", sa.String(length=64), server_default="", nullable=False),
        sa.Column("baseline_model", sa.String(length=128), server_default="", nullable=False),
        sa.Column(
            "baseline_reasoning_effort", sa.String(length=16), server_default="", nullable=False
        ),
        sa.Column("candidate_endpoint", sa.String(length=64), server_default="", nullable=False),
        sa.Column("candidate_provider", sa.String(length=64), server_default="", nullable=False),
        sa.Column("candidate_model", sa.String(length=128), server_default="", nullable=False),
        sa.Column(
            "candidate_reasoning_effort", sa.String(length=16), server_default="", nullable=False
        ),
        sa.Column("judge_provider", sa.String(length=64), server_default="", nullable=False),
        sa.Column("judge_model", sa.String(length=128), server_default="", nullable=False),
        sa.Column("requested_samples", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("progress_completed", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("progress_total", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("recommendation", sa.String(length=32), server_default="", nullable=False),
        sa.Column("summary_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by_admin_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_eval_runs_user_id", "llm_eval_runs", ["user_id"])
    op.create_index("ix_llm_eval_runs_status", "llm_eval_runs", ["status"])
    op.create_index("ix_llm_eval_runs_created_at", "llm_eval_runs", ["created_at"])
    op.create_index("ix_llm_eval_runs_public_id", "llm_eval_runs", ["public_id"], unique=True)

    def side_columns(side: str) -> list[sa.Column]:
        return [
            sa.Column(f"{side}_text", sa.Text(), server_default="", nullable=False),
            sa.Column(f"{side}_tool_calls", sa.Text(), server_default="", nullable=False),
            sa.Column(f"{side}_replayed_lookups", sa.Text(), server_default="", nullable=False),
            sa.Column(
                f"{side}_stop_reason", sa.String(length=64), server_default="", nullable=False
            ),
            sa.Column(
                f"{side}_input_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
            ),
            sa.Column(
                f"{side}_output_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
            ),
            sa.Column(
                f"{side}_cache_read_tokens",
                sa.Integer(),
                server_default=sa.text("0"),
                nullable=False,
            ),
            sa.Column(
                f"{side}_cache_creation_tokens",
                sa.Integer(),
                server_default=sa.text("0"),
                nullable=False,
            ),
            sa.Column(
                f"{side}_latency_ms", sa.Float(), server_default=sa.text("0"), nullable=False
            ),
            sa.Column(f"{side}_error", sa.Text(), server_default="", nullable=False),
        ]

    op.create_table(
        "llm_eval_turn_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("message_seq", sa.Integer(), nullable=False),
        sa.Column("message_timestamp", sa.String(), server_default="", nullable=False),
        sa.Column("user_message", sa.Text(), server_default="", nullable=False),
        sa.Column("historic_reply", sa.Text(), server_default="", nullable=False),
        sa.Column("historic_tool_names", sa.Text(), server_default="", nullable=False),
        *side_columns("baseline"),
        *side_columns("candidate"),
        sa.Column("agreement", sa.String(length=48), server_default="", nullable=False),
        sa.Column("safety_issues", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "judge_verdict", sa.String(length=32), server_default="not_judged", nullable=False
        ),
        sa.Column("judge_rationale", sa.Text(), server_default="", nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["llm_eval_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "message_seq", name="uq_eval_turn_run_seq"),
    )
    op.create_index("ix_llm_eval_turn_results_run_id", "llm_eval_turn_results", ["run_id"])


def _create_comparison_tables() -> None:
    op.create_table(
        "model_comparison_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("created_by_admin_id", sa.String(), nullable=True),
        # A label for what the user runs on today. No call is sent here: the
        # baseline this report compares against is the recorded turn itself.
        sa.Column("incumbent_endpoint", sa.String(length=64), server_default="", nullable=False),
        sa.Column("incumbent_provider", sa.String(length=64), server_default="", nullable=False),
        sa.Column("incumbent_model", sa.String(length=128), server_default="", nullable=False),
        sa.Column("candidate_endpoint", sa.String(length=64), server_default="", nullable=False),
        sa.Column("candidate_provider", sa.String(length=64), server_default="", nullable=False),
        sa.Column("candidate_model", sa.String(length=128), server_default="", nullable=False),
        sa.Column(
            "candidate_reasoning_effort", sa.String(length=16), server_default="", nullable=False
        ),
        sa.Column("requested_samples", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("progress_completed", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("progress_total", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("summary_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by_admin_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_comparison_runs_user_id", "model_comparison_runs", ["user_id"])
    op.create_index("ix_model_comparison_runs_status", "model_comparison_runs", ["status"])
    op.create_index("ix_model_comparison_runs_created_at", "model_comparison_runs", ["created_at"])
    op.create_index(
        "ix_model_comparison_runs_public_id", "model_comparison_runs", ["public_id"], unique=True
    )

    op.create_table(
        "model_comparison_turns",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("message_seq", sa.Integer(), nullable=False),
        sa.Column("message_timestamp", sa.String(), server_default="", nullable=False),
        sa.Column("user_message", sa.Text(), server_default="", nullable=False),
        sa.Column("production_reply", sa.Text(), server_default="", nullable=False),
        sa.Column("production_tool_calls", sa.Text(), server_default="", nullable=False),
        sa.Column("candidate_text", sa.Text(), server_default="", nullable=False),
        sa.Column("candidate_tool_calls", sa.Text(), server_default="", nullable=False),
        sa.Column("candidate_replayed_lookups", sa.Text(), server_default="", nullable=False),
        sa.Column("candidate_stop_reason", sa.String(length=64), server_default="", nullable=False),
        sa.Column(
            "candidate_input_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "candidate_output_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "candidate_cache_read_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "candidate_cache_creation_tokens",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("candidate_latency_ms", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("candidate_error", sa.Text(), server_default="", nullable=False),
        sa.Column("outcome", sa.String(length=32), server_default="", nullable=False),
        sa.Column("write_results", sa.Text(), server_default="", nullable=False),
        sa.Column("findings", sa.Text(), server_default="", nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["model_comparison_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "message_seq", name="uq_comparison_turn_run_seq"),
    )
    op.create_index("ix_model_comparison_turns_run_id", "model_comparison_turns", ["run_id"])


def _drop_comparison_tables() -> None:
    op.drop_index("ix_model_comparison_turns_run_id", table_name="model_comparison_turns")
    op.drop_table("model_comparison_turns")
    op.drop_index("ix_model_comparison_runs_public_id", table_name="model_comparison_runs")
    op.drop_index("ix_model_comparison_runs_created_at", table_name="model_comparison_runs")
    op.drop_index("ix_model_comparison_runs_status", table_name="model_comparison_runs")
    op.drop_index("ix_model_comparison_runs_user_id", table_name="model_comparison_runs")
    op.drop_table("model_comparison_runs")


def upgrade() -> None:
    _drop_eval_tables()
    _create_comparison_tables()


def downgrade() -> None:
    _drop_comparison_tables()
    _create_eval_tables()
