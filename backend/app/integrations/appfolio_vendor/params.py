"""Pydantic parameter models for AppFolio Vendor Portal tools.

Kept in one module so tool builders can import the full set with one
line and so the agent's schema is centralized.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AppFolioListWorkOrdersParams(BaseModel):
    include_in_progress: bool = Field(
        default=True,
        description="Include in-progress work orders.",
    )
    include_completed: bool = Field(
        default=False,
        description="Include completed (closed) work orders.",
    )
    include_estimates: bool = Field(
        default=True,
        description="Include work orders where the property manager wants an estimate.",
    )
    customer_id: str = Field(
        default="",
        description="Customer (property manager) ID to filter by. Omit for all customers.",
    )


class AppFolioSearchWorkOrdersParams(BaseModel):
    search_term: str = Field(
        description="Work order number, address, unit, tenant name, or other free text.",
    )


class AppFolioGetWorkOrderParams(BaseModel):
    customer_id: str = Field(
        description="Customer (property manager) ID from list or search output.",
    )
    work_order_id: str = Field(description="AppFolio work order ID.")


class AppFolioUpdateWorkOrderStatusParams(BaseModel):
    work_order_id: str = Field(description="AppFolio work order ID.")
    status_code: int = Field(
        description=(
            "Status code. Common: 0=new, 4=in progress, 8=completed. Confirm with the"
            " user when uncertain rather than guessing."
        ),
    )


class AppFolioUndoWorkOrderStatusParams(BaseModel):
    work_order_id: str = Field(description="AppFolio work order ID.")
    previous_status: str = Field(
        description="Prior status code or label, as returned by appfolio_get_work_order.",
    )


class AppFolioListNotesParams(BaseModel):
    work_order_id: str = Field(description="AppFolio work order ID.")


class AppFolioAddNoteParams(BaseModel):
    work_order_id: str = Field(description="AppFolio work order ID.")
    body: str = Field(description="Note text.")
    media_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Photos from the conversation, each an image's original_url or a media"
            " handle (e.g. 'media_ab12cd'). Uploaded inline with the note."
        ),
    )


class AppFolioUpdateNoteParams(BaseModel):
    work_order_id: str = Field(description="AppFolio work order ID.")
    note_id: str = Field(description="Note ID.")
    body: str = Field(description="Replacement note text.")
    media_refs: list[str] = Field(
        default_factory=list,
        description="More photos, as in appfolio_add_note. Existing attachments are kept.",
    )


class AppFolioInvoiceLineItem(BaseModel):
    description: str = Field(description="Line description, e.g. 'Labor: 4hr'.")
    quantity: float = Field(default=1.0, description="Quantity; decimals allowed.")
    amount: float = Field(
        description=(
            "Per-unit price in dollars. AppFolio stores quantity x amount, so 5 hours"
            " at $55/hr is quantity=5, amount=55, not quantity=1, amount=275."
        ),
    )


class AppFolioCreateInvoiceParams(BaseModel):
    customer_id: str = Field(
        description="Customer (property manager) ID.",
    )
    work_order_id: str = Field(description="Work order ID to bill.")
    line_items: list[AppFolioInvoiceLineItem] = Field(
        description="Invoice lines.",
    )
    reference_number: str = Field(
        default="",
        description=(
            "Vendor reference number to print. Omit to let AppFolio generate one"
            " ('<workOrderNumber>-<sequence>')."
        ),
    )


class AppFolioUploadInvoicePdfParams(BaseModel):
    customer_id: str = Field(
        description="Customer (property manager) ID.",
    )
    work_order_id: str = Field(description="Work order ID to bill.")
    media_refs: list[str] = Field(
        description=(
            "PDFs or photos from the conversation, as original_url or media handle;"
            " uploaded as one invoice document."
        ),
    )
    reference_number: str = Field(
        default="",
        description="Vendor reference number to print.",
    )
