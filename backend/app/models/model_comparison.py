"""Model comparison runs and their per-turn reports."""

from __future__ import annotations

import uuid as _uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base
from backend.app.models.types import EncryptedString


class ComparisonRun(Base):
    """One model comparison run for one user, started from the admin console.

    A run replays the user's most recent turns through a candidate model and
    records what it decided beside what production actually did. It exists to
    give an operator something to read before moving a user to a different
    model. It does not decide anything, and there is deliberately no verdict
    column: the evaluator this replaced had one, and it was wrong in both
    directions often enough that the operator stopped believing it.

    ``summary_json`` holds the serialized summary (outcome buckets, finding
    counts per side, write-match counts, token and latency totals, notes). It
    is denormalized on purpose: the per-turn rows are the evidence, and
    recomputing a summary from them would silently restate a run the operator
    already read whenever a check changes.

    Nothing here is charged to the user. Replay calls bypass the quota
    pipeline and are not written to ``llm_usage_logs``, so a comparison run
    never shows up as the user's own spend.
    """

    __tablename__ = "model_comparison_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # The id the API and the report URL use. A run's report is a page an
    # operator bookmarks, revisits, and pastes to someone else, so its address
    # must not be a guessable row counter that also leaks how many runs exist.
    public_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, nullable=False, default=lambda: str(_uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # ``SET NULL`` rather than CASCADE: deleting an admin must not delete the
    # evidence behind a switching decision they made.
    created_by_admin_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # What the user's loop runs on today. A label, not a measurement: no call
    # is sent here, and the recorded turns may predate this configuration. It
    # is on the row because the report is read weeks later, by which time the
    # subscription may point somewhere else and there would be nothing left
    # saying what the candidate was being considered against.
    incumbent_endpoint: Mapped[str] = mapped_column(String(64), default="")
    incumbent_provider: Mapped[str] = mapped_column(String(64), default="")
    incumbent_model: Mapped[str] = mapped_column(String(128), default="")

    # Where the replay's calls went and at what effort. Recorded because
    # neither is recoverable afterwards: the settings they defaulted from are
    # mutable, and a run whose effort is unknown cannot be reproduced.
    candidate_endpoint: Mapped[str] = mapped_column(String(64), default="")
    candidate_provider: Mapped[str] = mapped_column(String(64), default="")
    candidate_model: Mapped[str] = mapped_column(String(128), default="")
    candidate_reasoning_effort: Mapped[str] = mapped_column(String(16), default="")

    requested_samples: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    progress_completed: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Touched by the worker as each turn lands. The sweep uses it to tell a
    # run abandoned by a dead process from one still advancing in another:
    # during a rolling deploy the new instance boots while the old one is
    # still draining, and a sweep with no liveness signal marks a running
    # comparison interrupted underneath itself.
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    turns: Mapped[list[ComparisonTurn]] = relationship(
        "ComparisonTurn",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="raise",
    )


class ComparisonTurn(Base):
    """One replayed turn: what production did, and what the candidate did.

    Every text column here is reconstructed from, or generated in response
    to, the user's real conversation, so all of them carry the same
    ``EncryptedString`` treatment as ``messages``. ``user_message`` is the
    user's own words, and the tool-call payloads routinely embed customer
    names, addresses and phone numbers.

    Token and latency columns stay plaintext: they are measurements, not
    content, and the report aggregates them on read.
    """

    __tablename__ = "model_comparison_turns"
    __table_args__ = (UniqueConstraint("run_id", "message_seq", name="uq_comparison_turn_run_seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("model_comparison_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    message_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    message_timestamp: Mapped[str] = mapped_column(String, default="")

    user_message: Mapped[str] = mapped_column(
        EncryptedString(table="model_comparison_turns", column="user_message"), default=""
    )
    # What the live turn did, read back from the transcript. The baseline the
    # candidate is shown against, and the standard the write comparison and
    # the unrequested-write check are measured against.
    production_reply: Mapped[str] = mapped_column(
        EncryptedString(table="model_comparison_turns", column="production_reply"), default=""
    )
    production_tool_calls: Mapped[str] = mapped_column(
        EncryptedString(table="model_comparison_turns", column="production_tool_calls"), default=""
    )

    candidate_text: Mapped[str] = mapped_column(
        EncryptedString(table="model_comparison_turns", column="candidate_text"), default=""
    )
    candidate_tool_calls: Mapped[str] = mapped_column(
        EncryptedString(table="model_comparison_turns", column="candidate_tool_calls"), default=""
    )
    # Lookups the replay fed back from the live turn before the decision
    # above, with their recorded results. See ``model_comparison.execution``.
    candidate_replayed_lookups: Mapped[str] = mapped_column(
        EncryptedString(table="model_comparison_turns", column="candidate_replayed_lookups"),
        default="",
    )
    candidate_stop_reason: Mapped[str] = mapped_column(String(64), default="")
    candidate_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    candidate_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    candidate_cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    candidate_cache_creation_tokens: Mapped[int] = mapped_column(Integer, default=0)
    candidate_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    candidate_error: Mapped[str] = mapped_column(Text, default="")

    # ``types.TurnOutcome``: how the candidate's decision reads against the
    # writes production made on this turn.
    outcome: Mapped[str] = mapped_column(String(32), default="")
    # One entry per write the live turn made, with what the candidate did
    # about it. Encrypted: the compared arguments are record IDs and, for a
    # write with none, the whole payload.
    write_results: Mapped[str] = mapped_column(
        EncryptedString(table="model_comparison_turns", column="write_results"), default=""
    )
    findings: Mapped[str] = mapped_column(
        EncryptedString(table="model_comparison_turns", column="findings"), default=""
    )

    run: Mapped[ComparisonRun] = relationship("ComparisonRun", back_populates="turns", lazy="raise")
