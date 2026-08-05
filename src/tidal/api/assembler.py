"""Result assembly: turn a finished batch's items into OpenAI result files.

The dispatcher calls :func:`finalize_if_done` whenever a batch might have
drained. It is deliberately idempotent and cheap to call: if any item is still
PENDING or INFLIGHT, or the batch already carries result files, it does
nothing and returns ``None``.

Wire shapes (what the ``openai`` SDK expects to read back):

* output line — ``{"id", "custom_id", "response": {"status_code", "request_id",
  "body"}, "error": null}``
* error line  — ``{"id", "custom_id", "response": null, "error": {"code",
  "message"}}``

Cancelled items appear in neither file: OpenAI reports them only through
``request_counts``, where — like expired items — they count as neither
completed nor failed.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from tidal.store.interfaces import BatchRecord, BatchStatus, ItemRecord, ItemState, Repository

__all__ = ["finalize_if_done"]

OUTPUT_PURPOSE = "batch_output"
ERROR_PURPOSE = "batch_error"

#: Error codes used in the error file.
FAILED_CODE = "request_failed"
EXPIRED_CODE = "batch_expired"

_DEFAULT_MESSAGES = {
    FAILED_CODE: "The request failed and exhausted its retries.",
    EXPIRED_CODE: "The request expired before the batch completion window closed.",
}

#: Statuses that mean "this batch is done"; combined with result files already
#: being attached, they make finalization a no-op.
_TERMINAL = frozenset(
    {BatchStatus.COMPLETED, BatchStatus.EXPIRED, BatchStatus.CANCELLED, BatchStatus.FAILED}
)


def finalize_if_done(
    repo: Repository, batch_id: str, *, now: datetime | None = None
) -> BatchRecord | None:
    """Write the result files and close the batch, if it has actually drained.

    Args:
        repo: the store.
        batch_id: batch to consider; unknown ids are skipped.
        now: injected clock (UTC-aware) for deterministic tests.

    Returns:
        The finalized :class:`BatchRecord`, or ``None`` when the batch still
        has outstanding items, has already been finalized, or does not exist.

    Final status: a cancelling batch becomes ``cancelled``; a batch past its
    ``expires_at`` that has expired items becomes ``expired``; anything else
    becomes ``completed`` — including a batch whose every item failed, which
    matches OpenAI (a completed batch with an error file).
    """
    timestamp = now or datetime.now(UTC)
    batch = repo.get_batch(batch_id)
    if batch is None:
        return None
    if batch.status in _TERMINAL and (batch.output_file_id or batch.error_file_id):
        return None
    if any(repo.pending_and_inflight(batch_id)):
        return None

    succeeded = repo.list_items(batch_id, [ItemState.SUCCEEDED])
    unfulfilled = repo.list_items(batch_id, [ItemState.FAILED, ItemState.EXPIRED])

    output_file_id = _store(
        repo,
        OUTPUT_PURPOSE,
        f"{batch_id}_output.jsonl",
        [_output_line(item) for item in succeeded],
        timestamp,
    )
    error_file_id = _store(
        repo,
        ERROR_PURPOSE,
        f"{batch_id}_error.jsonl",
        [_error_line(item) for item in unfulfilled],
        timestamp,
    )

    status = _final_status(batch, unfulfilled, timestamp)
    if status is BatchStatus.COMPLETED and batch.status is BatchStatus.VALIDATING:
        # The dispatcher normally flips a batch to in_progress when it claims
        # the first item; a batch finalized before that still has to walk the
        # state machine rather than jump validating -> completed.
        repo.set_batch_status(batch_id, BatchStatus.IN_PROGRESS, now=timestamp)
    return repo.finalize_batch(batch_id, status, output_file_id, error_file_id, now=timestamp)


def _final_status(batch: BatchRecord, unfulfilled: list[ItemRecord], now: datetime) -> BatchStatus:
    if batch.status in (BatchStatus.CANCELLING, BatchStatus.CANCELLED):
        return BatchStatus.CANCELLED
    expired = any(item.state is ItemState.EXPIRED for item in unfulfilled)
    if batch.status is BatchStatus.EXPIRED or (now >= batch.expires_at and expired):
        return BatchStatus.EXPIRED
    return BatchStatus.COMPLETED


def _store(
    repo: Repository, purpose: str, filename: str, lines: list[dict], now: datetime
) -> str | None:
    """Persist JSONL ``lines``, or return ``None`` when there are none."""
    if not lines:
        return None
    content = "".join(json.dumps(line) + "\n" for line in lines).encode()
    return repo.create_file(purpose, filename, content, now=now).id


def _output_line(item: ItemRecord) -> dict:
    return {
        "id": _result_id(),
        "custom_id": item.custom_id,
        "response": {
            "status_code": 200,
            # vLLM's request id when we have it, else the item id: the field
            # only has to be stable and unique per result line.
            "request_id": item.vllm_request_id or item.id,
            "body": item.result_json or {},
        },
        "error": None,
    }


def _error_line(item: ItemRecord) -> dict:
    code = EXPIRED_CODE if item.state is ItemState.EXPIRED else FAILED_CODE
    return {
        "id": _result_id(),
        "custom_id": item.custom_id,
        "response": None,
        "error": {"code": code, "message": item.last_error or _DEFAULT_MESSAGES[code]},
    }


def _result_id() -> str:
    return f"batch_req_{uuid.uuid4().hex}"
