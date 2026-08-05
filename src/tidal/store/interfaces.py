"""Contract layer for Tidal's persistent store.

Every other component (API, dispatcher, metering) programs against these
types and the ``Repository`` protocol — never against SQLAlchemy models
directly. The SQLite→Postgres swap is a DSN change because nothing outside
``tidal.store`` sees the ORM.

State machines (mirrors OpenAI Batch API semantics):

    batch:  validating → in_progress → completed | expired | cancelling → cancelled
            validating → failed            (input file fails validation)
    item:   pending → inflight → succeeded | failed
            pending | inflight → expired   (batch hit its 24h window)
            pending | inflight → cancelled (batch cancelled)

Illegal transitions must raise ``IllegalTransition``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


class BatchStatus(enum.StrEnum):
    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class ItemState(enum.StrEnum):
    PENDING = "pending"
    INFLIGHT = "inflight"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class IllegalTransition(Exception):
    """Raised on any state change not permitted by the state machines above."""


@dataclass
class FileRecord:
    id: str                      # "file-<uuid>"
    purpose: str                 # "batch" | "batch_output" | "batch_error"
    filename: str
    size_bytes: int
    sha256: str
    created_at: datetime
    blob_path: str               # filesystem path of the raw JSONL bytes


@dataclass
class BatchRecord:
    id: str                      # "batch-<uuid>"
    input_file_id: str
    endpoint: str                # "/v1/chat/completions" | "/v1/completions"
    status: BatchStatus
    created_at: datetime
    expires_at: datetime         # created_at + completion window (24h)
    counts_total: int
    counts_completed: int
    counts_failed: int
    metadata: dict[str, str] = field(default_factory=dict)
    output_file_id: str | None = None
    error_file_id: str | None = None
    cancelled_at: datetime | None = None
    completed_at: datetime | None = None
    in_progress_at: datetime | None = None   # set on first transition to IN_PROGRESS


@dataclass
class ItemRecord:
    id: str                      # "item-<uuid>"
    batch_id: str
    custom_id: str               # unique within batch
    line_no: int
    prefix_group: int            # bucket for prefix-affinity ordering
    request_json: dict           # the OpenAI request body from the JSONL line
    state: ItemState
    attempts: int = 0
    last_error: str | None = None
    result_json: dict | None = None
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    submitted_at: datetime | None = None
    finished_at: datetime | None = None
    vllm_request_id: str | None = None


class Repository(Protocol):
    """Single data-access surface. All methods are synchronous; the API and
    dispatcher call them via ``asyncio.to_thread`` (SQLite is fast enough,
    and this keeps the ORM session model trivial).

    Conventions (pinned after A1 implementation review):
    - Every mutating method accepts keyword-only ``now: datetime | None = None``
      (UTC-aware) for deterministic tests; implementations default to
      ``datetime.now(UTC)`` and must reject naive datetimes.
    - Unknown ids: getters return None; mutators raise KeyError;
      ``read_file_content`` raises FileNotFoundError.
    - ``list_batches`` with an unknown ``after`` cursor returns an empty page.
    - ``batch_progress`` counts completed=SUCCEEDED, failed=FAILED only;
      EXPIRED/CANCELLED items count as neither, so completed+failed may be
      < total for expired/cancelled batches (OpenAI request_counts semantics).
    """

    # -- files --
    def create_file(self, purpose: str, filename: str, content: bytes) -> FileRecord: ...
    def get_file(self, file_id: str) -> FileRecord | None: ...
    def read_file_content(self, file_id: str) -> bytes: ...

    # -- batches --
    def create_batch(
        self, input_file_id: str, endpoint: str, window_hours: int,
        metadata: dict[str, str], items: list[tuple[str, int, int, dict]],
        # items: (custom_id, line_no, prefix_group, request_json)
    ) -> BatchRecord: ...
    def get_batch(self, batch_id: str) -> BatchRecord | None: ...
    def list_batches(self, limit: int, after: str | None) -> list[BatchRecord]: ...
    def find_batches_by_metadata(self, key: str, value: str) -> list[BatchRecord]: ...
    def set_batch_status(self, batch_id: str, status: BatchStatus) -> BatchRecord: ...
    def finalize_batch(
        self, batch_id: str, status: BatchStatus,
        output_file_id: str | None, error_file_id: str | None,
    ) -> BatchRecord: ...

    # -- items --
    def claim_pending_items(self, limit: int, order: str = "edf_prefix") -> list[ItemRecord]:
        """Atomically move up to ``limit`` PENDING items → INFLIGHT and return
        them. Ordering 'edf_prefix': earliest batch expires_at, then
        prefix_group, then line_no. Ordering 'fifo': line_no only."""
        ...
    def record_item_success(
        self, item_id: str, result_json: dict,
        prompt_tokens: int, completion_tokens: int, vllm_request_id: str | None,
    ) -> ItemRecord: ...
    def record_item_failure(self, item_id: str, error: str, *, retryable: bool) -> ItemRecord:
        """retryable=True and attempts < max → back to PENDING (attempts+1);
        otherwise → FAILED."""
        ...
    def requeue_inflight(self) -> int:
        """Crash recovery: all INFLIGHT items without result → PENDING.
        Returns count requeued."""
        ...
    def expire_batch(self, batch_id: str) -> BatchRecord:
        """PENDING/INFLIGHT items → EXPIRED; batch → EXPIRED."""
        ...
    def cancel_batch(self, batch_id: str) -> BatchRecord:
        """PENDING/INFLIGHT → CANCELLED; batch → CANCELLING (caller flips to
        CANCELLED after in-flight aborts complete)."""
        ...
    def batch_progress(self, batch_id: str) -> tuple[int, int, int]:
        """(total, completed, failed) live counts."""
        ...
    def pending_and_inflight(self, batch_id: str) -> tuple[int, int]: ...
    # added A7 — result assembly needs to enumerate a batch's items by state.
    def list_items(
        self, batch_id: str, states: list[ItemState] | None = None
    ) -> list[ItemRecord]:
        """Items of a batch in ``line_no`` order, optionally filtered by state."""
        ...

    # -- metering --
    def record_usage(
        self, item_id: str, model: str,
        prompt_tokens: int, completion_tokens: int, cost_usd: float,
    ) -> None: ...
    def usage_summary(self, since: datetime | None) -> list[dict]:
        """Rows: {model, prompt_tokens, completion_tokens, cost_usd, items}."""
        ...
    def set_price(self, model: str, input_per_mtok: float, output_per_mtok: float) -> None: ...
    def price_table(self) -> dict[str, tuple[float, float]]:
        """{model: (input_per_mtok, output_per_mtok)} — online list prices."""
        ...


def make_repository(dsn: str, blob_dir: str) -> Repository:
    """Factory implemented in tidal.store.repo (SQLAlchemy).

    dsn examples: 'sqlite:///tidal.db' (WAL enabled automatically),
    'postgresql+psycopg://...'.
    """
    from tidal.store.repo import SqlRepository
    return SqlRepository(dsn, blob_dir)
