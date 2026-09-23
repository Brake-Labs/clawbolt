"""Pydantic parameter models for CompanyCam tools.

Extracted from ``factory.py`` to keep that entrypoint module
focused on registration and factory wiring. Imported by
``projects``, ``photos``, and ``checklists``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _coerce_tags_to_list(value: Any) -> Any:
    """Parse JSON-encoded strings into lists so the LLM can pass either shape.

    Mirrors the ``_coerce_data_to_dict`` helper in the QuickBooks factory:
    Sonnet occasionally over-quotes the ``tags`` argument and emits it as
    ``"[]"`` or ``"[\\"kitchen\\"]"`` instead of a real JSON array. Accept
    both forms on the first round to avoid a wasted retry.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "tags must be a JSON array or a JSON-encoded array string; "
                f"could not parse string as JSON: {exc.msg}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(
                f"tags must be a JSON array; got a JSON-encoded {type(parsed).__name__}"
            )
        return parsed
    return value


class CompanyCamSearchParams(BaseModel):
    query: str = Field(description="Client name, address, or keyword.")


class CompanyCamCreateProjectParams(BaseModel):
    name: str = Field(description="Project name: the client name and address.")
    address: str = Field(default="", description="Street address.")


class CompanyCamUpdateProjectParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")
    name: str = Field(default="", description="New project name. Omit to keep.")
    address: str = Field(default="", description="New street address. Omit to keep.")


class CompanyCamUploadPhotoParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")
    original_url: str = Field(
        description="Media handle of the photo, e.g. 'media_ab12cd'. Never blank.",
    )
    description: str = Field(default="", description="Photo description.")
    tags: list[str] = Field(
        default_factory=list, description="Tags, e.g. 'kitchen', 'demo', 'before'."
    )

    _coerce_tags = field_validator("tags", mode="before")(_coerce_tags_to_list)


class CompanyCamGetProjectParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")


class CompanyCamArchiveProjectParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")


class CompanyCamDeleteProjectParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")


class CompanyCamUpdateNotepadParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")
    notepad: str = Field(description="New notepad content.")


class CompanyCamListDocumentsParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")
    page: int = Field(default=1, description="Page number.")


class CompanyCamAddCommentParams(BaseModel):
    target_type: str = Field(description="'project' or 'photo'.")
    target_id: str = Field(description="Project or photo ID.")
    content: str = Field(description="Comment text.")


class CompanyCamListCommentsParams(BaseModel):
    target_type: str = Field(description="'project' or 'photo'.")
    target_id: str = Field(description="Project or photo ID.")
    page: int = Field(default=1, description="Page number.")


class CompanyCamTagPhotoParams(BaseModel):
    photo_id: str = Field(description="CompanyCam photo ID.")
    tags: list[str] = Field(description="Tags, e.g. 'before', 'kitchen', 'damage'.")

    _coerce_tags = field_validator("tags", mode="before")(_coerce_tags_to_list)


class CompanyCamDeletePhotoParams(BaseModel):
    photo_id: str = Field(description="CompanyCam photo ID.")


class CompanyCamSearchPhotosParams(BaseModel):
    project_id: str = Field(
        default="",
        description="Project ID to filter to.",
    )
    start_date: str = Field(
        default="",
        description="Start date, ISO (e.g. 2024-01-15).",
    )
    end_date: str = Field(
        default="",
        description="End date, ISO (e.g. 2024-01-31).",
    )
    page: int = Field(default=1, description="Page number.")


class CompanyCamListChecklistsParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")


class CompanyCamGetChecklistParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")
    checklist_id: str = Field(description="Checklist ID.")


class CompanyCamCreateChecklistParams(BaseModel):
    project_id: str = Field(description="CompanyCam project ID.")
    template_id: str = Field(description="Checklist template ID.")
