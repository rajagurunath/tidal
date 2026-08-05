"""Pydantic wire models for the OpenAI-compatible Batch API.

Shapes are dictated by the official ``openai`` Python SDK: field names,
``object`` discriminators, and unix-integer timestamps all have to match or
the SDK's response models come back empty.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from tidal.store.interfaces import BatchRecord, FileRecord

#: Internal purposes → the purposes the OpenAI SDK knows about. Batch error
#: files are not a distinct purpose on the wire; OpenAI serves them as
#: ``batch_output``.
WIRE_PURPOSE = {
    "batch": "batch",
    "batch_output": "batch_output",
    "batch_error": "batch_output",
}

SUPPORTED_ENDPOINTS = ("/v1/chat/completions", "/v1/completions")
COMPLETION_WINDOW = "24h"


def unix(value: datetime | None) -> int | None:
    """Unix seconds, treating naive datetimes as UTC (SQLite gives naive)."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp())


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Literal["uploaded", "processed", "error"] = "processed"

    @classmethod
    def from_record(cls, rec: FileRecord) -> FileObject:
        return cls(
            id=rec.id,
            bytes=rec.size_bytes,
            created_at=unix(rec.created_at) or 0,
            filename=rec.filename,
            purpose=WIRE_PURPOSE.get(rec.purpose, rec.purpose),
        )


class FileListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject]
    has_more: bool = False


class RequestCounts(BaseModel):
    total: int
    completed: int
    failed: int


class BatchError(BaseModel):
    code: str
    message: str
    param: str | None = None
    line: int | None = None


class BatchErrors(BaseModel):
    object: Literal["list"] = "list"
    data: list[BatchError]


class BatchObject(BaseModel):
    id: str
    object: Literal["batch"] = "batch"
    endpoint: str
    errors: BatchErrors | None = None
    input_file_id: str
    completion_window: str = COMPLETION_WINDOW
    status: str
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: RequestCounts
    metadata: dict[str, str] | None = None

    @classmethod
    def from_record(
        cls,
        rec: BatchRecord,
        counts: tuple[int, int, int] | None = None,
        errors: BatchErrors | None = None,
    ) -> BatchObject:
        total, completed, failed = counts or (
            rec.counts_total,
            rec.counts_completed,
            rec.counts_failed,
        )
        status = rec.status.value
        finished = unix(rec.completed_at)
        cancelled = unix(rec.cancelled_at)
        return cls(
            id=rec.id,
            endpoint=rec.endpoint,
            errors=errors,
            input_file_id=rec.input_file_id,
            status=status,
            output_file_id=rec.output_file_id,
            error_file_id=rec.error_file_id,
            created_at=unix(rec.created_at) or 0,
            expires_at=unix(rec.expires_at),
            completed_at=finished if status == "completed" else None,
            failed_at=finished if status == "failed" else None,
            expired_at=finished if status == "expired" else None,
            cancelling_at=(cancelled or unix(rec.created_at)) if status == "cancelling" else None,
            cancelled_at=cancelled if status == "cancelled" else None,
            request_counts=RequestCounts(total=total, completed=completed, failed=failed),
            metadata=rec.metadata or {},
        )


class BatchListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[BatchObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class BatchCreateRequest(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str = COMPLETION_WINDOW
    metadata: dict[str, str] | None = Field(default=None)
