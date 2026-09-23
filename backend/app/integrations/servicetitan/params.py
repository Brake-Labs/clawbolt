"""Pydantic parameter models for ServiceTitan tools.

Centralized so tool builders import the whole set with one line and the
agent's schema stays consistent across the integration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class StSearchCustomersParams(BaseModel):
    """Inputs for ``st_search_customers``.

    The agent passes a single free-form query string. The tool
    decides whether to filter ServiceTitan by name or phone based on
    whether the query looks numeric. The optional ``limit`` caps the
    response so chat output stays compact; ServiceTitan's own page
    size is independent and may return more.
    """

    query: str = Field(
        description=(
            "Name fragment (e.g. 'Acme', 'Jane Doe') or, if mostly digits, a partial"
            " phone number (e.g. '5550101')."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum matches to return.",
    )


class StGetCustomerParams(BaseModel):
    """Inputs for ``st_get_customer``."""

    customer_id: int = Field(
        description="Customer ID from st_search_customers.",
    )


class StListAppointmentsParams(BaseModel):
    """Inputs for ``st_list_appointments``.

    All fields are optional. When ``from_date`` and ``to_date`` are
    both omitted the tool defaults to today's appointments in UTC,
    which matches the "today's dispatch view" use case in the issue.
    """

    from_date: str | None = Field(
        default=None,
        description=(
            "Inclusive start-time lower bound, ISO 8601 (e.g. 2026-05-11 or"
            " 2026-05-11T08:00:00Z). Omit for the start of today (UTC)."
        ),
    )
    to_date: str | None = Field(
        default=None,
        description=(
            "Exclusive start-time upper bound, ISO 8601. Omit for the start of tomorrow (UTC)."
        ),
    )
    status: str | None = Field(
        default=None,
        description=(
            "Status to filter to: Scheduled, Dispatched, Working, Done, or Hold. Omit for all."
        ),
    )


class StAddJobNoteParams(BaseModel):
    """Inputs for ``st_add_job_note``.

    Posts a free-form note to a ServiceTitan job. The note is visible
    to anyone in the tenant with access to the job, so the tool gates
    on approval before sending. ``pin_to_top`` mirrors the API's
    ``pinToTop`` flag and surfaces the note above other notes in the
    job's note feed.
    """

    job_id: int = Field(
        description=("Job ID from an appointment lookup. Confirm it with the user before calling."),
    )
    text: str = Field(
        min_length=1,
        description="Plain-text note body; must not be blank.",
    )
    pin_to_top: bool = Field(
        default=False,
        description=(
            "Pin the note above the job's other notes. Only when the user explicitly"
            " asks for a pinned note."
        ),
    )

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value
