"""In-memory ``Repository`` implementation used by the API unit tests.

Deliberately independent of ``tidal.store.repo`` — the API layer is only ever
allowed to see the ``tidal.store.interfaces.Repository`` protocol, so the tests
prove that by talking to a fake that implements nothing else.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

from tidal.store.interfaces import (
    BatchRecord,
    BatchStatus,
    FileRecord,
    IllegalTransition,
    ItemRecord,
    ItemState,
)

MAX_ITEM_ATTEMPTS = 3

_LEGAL_BATCH: dict[BatchStatus, set[BatchStatus]] = {
    BatchStatus.VALIDATING: {
        BatchStatus.IN_PROGRESS,
        BatchStatus.FAILED,
        BatchStatus.CANCELLING,
        BatchStatus.EXPIRED,
    },
    BatchStatus.IN_PROGRESS: {
        BatchStatus.COMPLETED,
        BatchStatus.EXPIRED,
        BatchStatus.CANCELLING,
        BatchStatus.FAILED,
    },
    BatchStatus.CANCELLING: {BatchStatus.CANCELLED, BatchStatus.COMPLETED},
    BatchStatus.COMPLETED: set(),
    BatchStatus.FAILED: set(),
    BatchStatus.EXPIRED: set(),
    BatchStatus.CANCELLED: set(),
}

_TERMINAL_ITEM = {
    ItemState.SUCCEEDED,
    ItemState.FAILED,
    ItemState.EXPIRED,
    ItemState.CANCELLED,
}


def _now() -> datetime:
    return datetime.now(UTC)


class FakeRepository:
    """Everything lives in dicts; blobs live in ``self.blobs``."""

    def __init__(self) -> None:
        self.files: dict[str, FileRecord] = {}
        self.blobs: dict[str, bytes] = {}
        self.batches: dict[str, BatchRecord] = {}
        self.items: dict[str, list[ItemRecord]] = {}
        self.usage: list[dict] = []

    # -- files --------------------------------------------------------------

    def create_file(self, purpose: str, filename: str, content: bytes) -> FileRecord:
        file_id = f"file-{uuid.uuid4().hex}"
        rec = FileRecord(
            id=file_id,
            purpose=purpose,
            filename=filename,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            created_at=_now(),
            blob_path=f"memory://{file_id}",
        )
        self.files[file_id] = rec
        self.blobs[file_id] = content
        return rec

    def get_file(self, file_id: str) -> FileRecord | None:
        return self.files.get(file_id)

    def read_file_content(self, file_id: str) -> bytes:
        if file_id not in self.blobs:
            raise KeyError(file_id)
        return self.blobs[file_id]

    # -- batches ------------------------------------------------------------

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        window_hours: int,
        metadata: dict[str, str],
        items: list[tuple[str, int, int, dict]],
    ) -> BatchRecord:
        created = _now()
        batch_id = f"batch-{uuid.uuid4().hex}"
        rec = BatchRecord(
            id=batch_id,
            input_file_id=input_file_id,
            endpoint=endpoint,
            status=BatchStatus.VALIDATING,
            created_at=created,
            expires_at=created + timedelta(hours=window_hours),
            counts_total=len(items),
            counts_completed=0,
            counts_failed=0,
            metadata=dict(metadata or {}),
        )
        self.batches[batch_id] = rec
        self.items[batch_id] = [
            ItemRecord(
                id=f"item-{uuid.uuid4().hex}",
                batch_id=batch_id,
                custom_id=custom_id,
                line_no=line_no,
                prefix_group=prefix_group,
                request_json=body,
                state=ItemState.PENDING,
            )
            for custom_id, line_no, prefix_group, body in items
        ]
        return rec

    def get_batch(self, batch_id: str) -> BatchRecord | None:
        return self.batches.get(batch_id)

    def list_batches(self, limit: int, after: str | None) -> list[BatchRecord]:
        ordered = sorted(self.batches.values(), key=lambda b: b.created_at, reverse=True)
        if after is not None:
            ids = [b.id for b in ordered]
            ordered = ordered[ids.index(after) + 1 :] if after in ids else []
        return ordered[:limit]

    def find_batches_by_metadata(self, key: str, value: str) -> list[BatchRecord]:
        return [b for b in self.batches.values() if b.metadata.get(key) == value]

    def set_batch_status(self, batch_id: str, status: BatchStatus) -> BatchRecord:
        rec = self._batch(batch_id)
        self._guard(rec.status, status)
        rec.status = status
        if status is BatchStatus.CANCELLED:
            rec.cancelled_at = _now()
        if status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED):
            rec.completed_at = _now()
        return rec

    def finalize_batch(
        self,
        batch_id: str,
        status: BatchStatus,
        output_file_id: str | None,
        error_file_id: str | None,
    ) -> BatchRecord:
        rec = self.set_batch_status(batch_id, status)
        rec.output_file_id = output_file_id
        rec.error_file_id = error_file_id
        return rec

    # -- items --------------------------------------------------------------

    def claim_pending_items(self, limit: int, order: str = "edf_prefix") -> list[ItemRecord]:
        candidates = [
            (self.batches[bid], item)
            for bid, items in self.items.items()
            for item in items
            if item.state is ItemState.PENDING
        ]
        if order == "fifo":
            candidates.sort(key=lambda pair: pair[1].line_no)
        else:
            candidates.sort(
                key=lambda pair: (pair[0].expires_at, pair[1].prefix_group, pair[1].line_no)
            )
        claimed = []
        for _batch, item in candidates[:limit]:
            item.state = ItemState.INFLIGHT
            item.submitted_at = _now()
            claimed.append(item)
        return claimed

    def record_item_success(
        self,
        item_id: str,
        result_json: dict,
        prompt_tokens: int,
        completion_tokens: int,
        vllm_request_id: str | None,
    ) -> ItemRecord:
        item = self._item(item_id)
        item.state = ItemState.SUCCEEDED
        item.result_json = result_json
        item.usage_prompt_tokens = prompt_tokens
        item.usage_completion_tokens = completion_tokens
        item.vllm_request_id = vllm_request_id
        item.finished_at = _now()
        self._recount(item.batch_id)
        return item

    def record_item_failure(self, item_id: str, error: str, *, retryable: bool) -> ItemRecord:
        item = self._item(item_id)
        item.attempts += 1
        item.last_error = error
        if retryable and item.attempts < MAX_ITEM_ATTEMPTS:
            item.state = ItemState.PENDING
        else:
            item.state = ItemState.FAILED
            item.finished_at = _now()
        self._recount(item.batch_id)
        return item

    def requeue_inflight(self) -> int:
        n = 0
        for items in self.items.values():
            for item in items:
                if item.state is ItemState.INFLIGHT and item.result_json is None:
                    item.state = ItemState.PENDING
                    n += 1
        return n

    def expire_batch(self, batch_id: str) -> BatchRecord:
        self._sweep(batch_id, ItemState.EXPIRED)
        return self.set_batch_status(batch_id, BatchStatus.EXPIRED)

    def cancel_batch(self, batch_id: str) -> BatchRecord:
        self._sweep(batch_id, ItemState.CANCELLED)
        return self.set_batch_status(batch_id, BatchStatus.CANCELLING)

    def batch_progress(self, batch_id: str) -> tuple[int, int, int]:
        items = self.items.get(batch_id, [])
        completed = sum(1 for i in items if i.state is ItemState.SUCCEEDED)
        failed = sum(
            1 for i in items if i.state in (ItemState.FAILED, ItemState.EXPIRED)
        )
        return len(items), completed, failed

    def pending_and_inflight(self, batch_id: str) -> tuple[int, int]:
        items = self.items.get(batch_id, [])
        return (
            sum(1 for i in items if i.state is ItemState.PENDING),
            sum(1 for i in items if i.state is ItemState.INFLIGHT),
        )

    # -- metering -----------------------------------------------------------

    def record_usage(
        self,
        item_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        self.usage.append(
            {
                "item_id": item_id,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "priced_at": _now(),
            }
        )

    def usage_summary(self, since: datetime | None) -> list[dict]:
        rows: dict[str, dict] = {}
        for entry in self.usage:
            if since is not None and entry["priced_at"] < since:
                continue
            row = rows.setdefault(
                entry["model"],
                {
                    "model": entry["model"],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "items": 0,
                },
            )
            row["prompt_tokens"] += entry["prompt_tokens"]
            row["completion_tokens"] += entry["completion_tokens"]
            row["cost_usd"] += entry["cost_usd"]
            row["items"] += 1
        return list(rows.values())

    # -- helpers (test-side only) ------------------------------------------

    def _batch(self, batch_id: str) -> BatchRecord:
        rec = self.batches.get(batch_id)
        if rec is None:
            raise KeyError(batch_id)
        return rec

    def _item(self, item_id: str) -> ItemRecord:
        for items in self.items.values():
            for item in items:
                if item.id == item_id:
                    return item
        raise KeyError(item_id)

    def _guard(self, current: BatchStatus, target: BatchStatus) -> None:
        if target not in _LEGAL_BATCH[current]:
            raise IllegalTransition(f"{current.value} → {target.value}")

    def _sweep(self, batch_id: str, state: ItemState) -> None:
        for item in self.items.get(batch_id, []):
            if item.state in (ItemState.PENDING, ItemState.INFLIGHT):
                item.state = state
                item.finished_at = _now()
        self._recount(batch_id)

    def _recount(self, batch_id: str) -> None:
        total, completed, failed = self.batch_progress(batch_id)
        rec = self.batches[batch_id]
        rec.counts_total, rec.counts_completed, rec.counts_failed = total, completed, failed

    def all_items(self, batch_id: str) -> list[ItemRecord]:
        return list(self.items.get(batch_id, []))

    def terminal_items(self, batch_id: str) -> list[ItemRecord]:
        return [i for i in self.items.get(batch_id, []) if i.state in _TERMINAL_ITEM]
