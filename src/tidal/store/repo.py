"""SQLAlchemy 2.0 implementation of :class:`tidal.store.interfaces.Repository`.

Nothing outside this module sees the ORM: every public method takes and
returns the plain dataclasses declared in ``interfaces.py``. Swapping SQLite
for Postgres is therefore a DSN change.

Design notes:

* **WAL** — SQLite DSNs get ``PRAGMA journal_mode=WAL`` on every connect so
  the API process and the dispatcher can read while the other writes.
* **Atomic claim** — :meth:`SqlRepository.claim_pending_items` is a single
  ``UPDATE ... WHERE state='pending' ... RETURNING id`` statement, so two
  dispatchers racing for the same item can never both win.
* **Time** — every method that stamps a timestamp accepts ``now=`` so callers
  (and tests) can be deterministic. All datetimes are timezone-aware UTC;
  they are stored naive-UTC and re-tagged on the way out.
* **Blobs** — file bytes live at ``<blob_dir>/<file_id>.jsonl``; the database
  keeps only the metadata and the path.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from sqlalchemy import (
    JSON,
    CursorResult,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
    update,
)
from sqlalchemy import Enum as SaEnum
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import TypeDecorator

from tidal.store.interfaces import (
    BatchRecord,
    BatchStatus,
    FileRecord,
    IllegalTransition,
    ItemRecord,
    ItemState,
)

__all__ = ["SqlRepository"]

# -- state machines ------------------------------------------------------

_BATCH_TRANSITIONS: dict[BatchStatus, frozenset[BatchStatus]] = {
    BatchStatus.VALIDATING: frozenset(
        {BatchStatus.IN_PROGRESS, BatchStatus.FAILED, BatchStatus.CANCELLING, BatchStatus.EXPIRED}
    ),
    BatchStatus.IN_PROGRESS: frozenset(
        {BatchStatus.COMPLETED, BatchStatus.EXPIRED, BatchStatus.CANCELLING}
    ),
    BatchStatus.CANCELLING: frozenset({BatchStatus.CANCELLED}),
    BatchStatus.COMPLETED: frozenset(),
    BatchStatus.FAILED: frozenset(),
    BatchStatus.EXPIRED: frozenset(),
    BatchStatus.CANCELLED: frozenset(),
}

_ITEM_TRANSITIONS: dict[ItemState, frozenset[ItemState]] = {
    ItemState.PENDING: frozenset({ItemState.INFLIGHT, ItemState.EXPIRED, ItemState.CANCELLED}),
    ItemState.INFLIGHT: frozenset(
        {
            ItemState.SUCCEEDED,
            ItemState.FAILED,
            ItemState.PENDING,
            ItemState.EXPIRED,
            ItemState.CANCELLED,
        }
    ),
    ItemState.SUCCEEDED: frozenset(),
    ItemState.FAILED: frozenset(),
    ItemState.EXPIRED: frozenset(),
    ItemState.CANCELLED: frozenset(),
}

_TERMINAL_ITEM_STATES = frozenset(
    {ItemState.SUCCEEDED, ItemState.FAILED, ItemState.EXPIRED, ItemState.CANCELLED}
)


def _check(kind: str, ident: str, src, dst, table) -> None:
    """Raise :class:`IllegalTransition` unless ``src -> dst`` is permitted.

    Self-transitions are always allowed so callers can be idempotent.
    """
    if src is dst:
        return
    if dst not in table[src]:
        raise IllegalTransition(f"{kind} {ident}: {src.value} -> {dst.value} is not allowed")


def _now(now: datetime | None) -> datetime:
    """Normalise an optional caller-supplied timestamp to aware UTC."""
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return now.astimezone(UTC)


# -- ORM -----------------------------------------------------------------


class _UtcDateTime(TypeDecorator):
    """Stores aware datetimes as naive UTC and returns them aware again."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime rejected; use timezone-aware UTC")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base for the Tidal schema."""


def _status_col(enum_cls):
    return SaEnum(
        enum_cls,
        native_enum=False,
        length=16,
        values_callable=lambda e: [m.value for m in e],
    )


class FileRow(Base):
    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    purpose: Mapped[str] = mapped_column(String(32))
    filename: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(_UtcDateTime)
    blob_path: Mapped[str] = mapped_column(Text)


class BatchRow(Base):
    __tablename__ = "batches"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    input_file_id: Mapped[str] = mapped_column(String(64), ForeignKey("files.id"))
    endpoint: Mapped[str] = mapped_column(String(64))
    status: Mapped[BatchStatus] = mapped_column(_status_col(BatchStatus), index=True)
    created_at: Mapped[datetime] = mapped_column(_UtcDateTime)
    expires_at: Mapped[datetime] = mapped_column(_UtcDateTime, index=True)
    counts_total: Mapped[int] = mapped_column(Integer, default=0)
    counts_completed: Mapped[int] = mapped_column(Integer, default=0)
    counts_failed: Mapped[int] = mapped_column(Integer, default=0)
    # "metadata" is reserved on the declarative base, hence the attribute rename.
    batch_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    output_file_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_file_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    in_progress_at: Mapped[datetime | None] = mapped_column(_UtcDateTime, nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(_UtcDateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(_UtcDateTime, nullable=True)


class ItemRow(Base):
    __tablename__ = "batch_items"
    __table_args__ = (UniqueConstraint("batch_id", "custom_id", name="uq_item_custom_id"),)

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    batch_id: Mapped[str] = mapped_column(String(64), ForeignKey("batches.id"), index=True)
    custom_id: Mapped[str] = mapped_column(String(255))
    line_no: Mapped[int] = mapped_column(Integer)
    prefix_group: Mapped[int] = mapped_column(Integer)
    request_json: Mapped[dict] = mapped_column(JSON)
    state: Mapped[ItemState] = mapped_column(_status_col(ItemState), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    usage_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(_UtcDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(_UtcDateTime, nullable=True)
    vllm_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class UsageRow(Base):
    __tablename__ = "usage_ledger"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[str] = mapped_column(String(64), ForeignKey("batch_items.id"), index=True)
    model: Mapped[str] = mapped_column(String(255), index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(_UtcDateTime, index=True)


class PriceRow(Base):
    """Per-model list price, in USD per million tokens (metering, Task A6)."""

    __tablename__ = "price_table"

    model: Mapped[str] = mapped_column(String(255), primary_key=True)
    input_usd_per_mtok: Mapped[float] = mapped_column(Float)
    output_usd_per_mtok: Mapped[float] = mapped_column(Float)


# -- row -> record -------------------------------------------------------


def _file_record(row: FileRow) -> FileRecord:
    return FileRecord(
        id=row.id,
        purpose=row.purpose,
        filename=row.filename,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        created_at=row.created_at,
        blob_path=row.blob_path,
    )


def _batch_record(row: BatchRow) -> BatchRecord:
    return BatchRecord(
        id=row.id,
        input_file_id=row.input_file_id,
        endpoint=row.endpoint,
        status=row.status,
        created_at=row.created_at,
        expires_at=row.expires_at,
        counts_total=row.counts_total,
        counts_completed=row.counts_completed,
        counts_failed=row.counts_failed,
        metadata=dict(row.batch_metadata or {}),
        output_file_id=row.output_file_id,
        error_file_id=row.error_file_id,
        cancelled_at=row.cancelled_at,
        completed_at=row.completed_at,
        in_progress_at=row.in_progress_at,
    )


def _item_record(row: ItemRow) -> ItemRecord:
    return ItemRecord(
        id=row.id,
        batch_id=row.batch_id,
        custom_id=row.custom_id,
        line_no=row.line_no,
        prefix_group=row.prefix_group,
        request_json=row.request_json,
        state=row.state,
        attempts=row.attempts,
        last_error=row.last_error,
        result_json=row.result_json,
        usage_prompt_tokens=row.usage_prompt_tokens,
        usage_completion_tokens=row.usage_completion_tokens,
        submitted_at=row.submitted_at,
        finished_at=row.finished_at,
        vllm_request_id=row.vllm_request_id,
    )


def _enable_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _record):  # pragma: no cover - driver callback
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# -- repository ----------------------------------------------------------


class SqlRepository:
    """SQLAlchemy-backed :class:`~tidal.store.interfaces.Repository`.

    Args:
        dsn: SQLAlchemy URL, e.g. ``sqlite:///tidal.db`` or
            ``postgresql+psycopg://...``. SQLite URLs get WAL journalling.
        blob_dir: directory for raw JSONL blobs; created if absent.
        max_attempts: retry budget per item, mirroring
            ``TidalConfig.max_item_attempts``. The config object is not
            imported here so the store stays dependency-free.
    """

    def __init__(self, dsn: str, blob_dir: str, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.blob_dir = Path(blob_dir)
        self.blob_dir.mkdir(parents=True, exist_ok=True)

        connect_args = {"check_same_thread": False} if dsn.startswith("sqlite") else {}
        self._engine = create_engine(dsn, future=True, connect_args=connect_args)
        if self._engine.dialect.name == "sqlite":
            _enable_sqlite_pragmas(self._engine)
        Base.metadata.create_all(self._engine)
        self._session = sessionmaker(self._engine, expire_on_commit=False)

    # -- files --

    def create_file(
        self, purpose: str, filename: str, content: bytes, *, now: datetime | None = None
    ) -> FileRecord:
        """Persist ``content`` under ``blob_dir`` and record its metadata."""
        file_id = f"file-{uuid.uuid4().hex}"
        blob_path = self.blob_dir / f"{file_id}.jsonl"
        blob_path.write_bytes(content)
        row = FileRow(
            id=file_id,
            purpose=purpose,
            filename=filename,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            created_at=_now(now),
            blob_path=str(blob_path),
        )
        with self._session.begin() as session:
            session.add(row)
        return _file_record(row)

    def get_file(self, file_id: str) -> FileRecord | None:
        """Return the file's metadata, or ``None`` if it does not exist."""
        with self._session() as session:
            row = session.get(FileRow, file_id)
            return _file_record(row) if row else None

    def read_file_content(self, file_id: str) -> bytes:
        """Return the raw bytes of a stored file.

        Raises:
            FileNotFoundError: if the id is unknown or the blob is missing.
        """
        with self._session() as session:
            row = session.get(FileRow, file_id)
            if row is None:
                raise FileNotFoundError(f"unknown file {file_id}")
            return Path(row.blob_path).read_bytes()

    # -- batches --

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        window_hours: int,
        metadata: dict[str, str],
        items: list[tuple[str, int, int, dict]],
        *,
        now: datetime | None = None,
    ) -> BatchRecord:
        """Create a VALIDATING batch and its PENDING items in one transaction.

        Args:
            items: ``(custom_id, line_no, prefix_group, request_json)`` tuples.
        """
        created_at = _now(now)
        batch = BatchRow(
            id=f"batch-{uuid.uuid4().hex}",
            input_file_id=input_file_id,
            endpoint=endpoint,
            status=BatchStatus.VALIDATING,
            created_at=created_at,
            expires_at=created_at + timedelta(hours=window_hours),
            counts_total=len(items),
            counts_completed=0,
            counts_failed=0,
            batch_metadata=dict(metadata or {}),
        )
        with self._session.begin() as session:
            session.add(batch)
            session.flush()
            session.add_all(
                ItemRow(
                    id=f"item-{uuid.uuid4().hex}",
                    batch_id=batch.id,
                    custom_id=custom_id,
                    line_no=line_no,
                    prefix_group=prefix_group,
                    request_json=request_json,
                    state=ItemState.PENDING,
                    attempts=0,
                )
                for custom_id, line_no, prefix_group, request_json in items
            )
        return _batch_record(batch)

    def get_batch(self, batch_id: str) -> BatchRecord | None:
        """Return the batch, or ``None`` if it does not exist."""
        with self._session() as session:
            row = self._find_batch(session, batch_id)
            return _batch_record(row) if row else None

    def list_batches(self, limit: int, after: str | None) -> list[BatchRecord]:
        """List batches newest-first, cursor-paginated by batch id.

        An unknown ``after`` cursor yields an empty page.
        """
        with self._session() as session:
            stmt = select(BatchRow).order_by(BatchRow.seq.desc()).limit(limit)
            if after is not None:
                cursor = self._find_batch(session, after)
                if cursor is None:
                    return []
                stmt = stmt.where(BatchRow.seq < cursor.seq)
            return [_batch_record(r) for r in session.execute(stmt).scalars()]

    def find_batches_by_metadata(self, key: str, value: str) -> list[BatchRecord]:
        """Return batches whose ``metadata[key]`` equals ``value``, newest first."""
        with self._session() as session:
            stmt = (
                select(BatchRow)
                .where(BatchRow.batch_metadata[key].as_string() == value)
                .order_by(BatchRow.seq.desc())
            )
            return [_batch_record(r) for r in session.execute(stmt).scalars()]

    def set_batch_status(
        self, batch_id: str, status: BatchStatus, *, now: datetime | None = None
    ) -> BatchRecord:
        """Move a batch to ``status``, enforcing the batch state machine."""
        with self._session.begin() as session:
            row = self._batch_row(session, batch_id)
            self._apply_batch_status(row, status, _now(now))
            return _batch_record(row)

    def finalize_batch(
        self,
        batch_id: str,
        status: BatchStatus,
        output_file_id: str | None,
        error_file_id: str | None,
        *,
        now: datetime | None = None,
    ) -> BatchRecord:
        """Attach the result files and move the batch to its terminal status.

        Attaching the files is a *claim*, and it is made inside the transaction:
        a batch that already carries an ``output_file_id`` or an
        ``error_file_id`` is returned exactly as it is. Assembly is idempotent
        by design and can be entered twice (the dispatcher's finalize hook and
        the API's cancel route both call it), so the second writer must not be
        able to replace the result files a client may already have downloaded,
        nor re-stamp ``completed_at``.
        """
        timestamp = _now(now)
        with self._session.begin() as session:
            row = self._batch_row(session, batch_id)
            if row.output_file_id is not None or row.error_file_id is not None:
                return _batch_record(row)
            if output_file_id is not None or error_file_id is not None:
                # Conditional UPDATE, so the claim is atomic on any backend
                # rather than a read-then-write with a window in the middle.
                claim = cast(
                    "CursorResult",
                    session.execute(
                        update(BatchRow)
                        .where(
                            BatchRow.id == batch_id,
                            BatchRow.output_file_id.is_(None),
                            BatchRow.error_file_id.is_(None),
                        )
                        .values(output_file_id=output_file_id, error_file_id=error_file_id)
                        .execution_options(synchronize_session=False)
                    ),
                )
                claimed = claim.rowcount
                session.expire(row)  # re-read what actually landed
                if not claimed:
                    return _batch_record(row)
            self._apply_batch_status(row, status, timestamp)
            return _batch_record(row)

    # -- items --

    def claim_pending_items(
        self, limit: int, order: str = "edf_prefix", *, now: datetime | None = None
    ) -> list[ItemRecord]:
        """Atomically move up to ``limit`` PENDING items to INFLIGHT.

        Ordering ``'edf_prefix'``: earliest batch ``expires_at``, then
        ``prefix_group``, then ``line_no``. Ordering ``'fifo'``: ``line_no``
        only (ties broken by insertion order). The claim is one
        ``UPDATE ... WHERE state='pending' ... RETURNING`` statement, so
        concurrent dispatchers never claim the same item twice.
        """
        order_by = self._claim_order(order)
        if limit <= 0:
            return []
        timestamp = _now(now)
        with self._session.begin() as session:
            candidates = (
                select(ItemRow.id)
                .join(BatchRow, ItemRow.batch_id == BatchRow.id)
                .where(ItemRow.state == ItemState.PENDING)
                .order_by(*order_by)
                .limit(limit)
            )
            claimed_ids = (
                session.execute(
                    update(ItemRow)
                    .where(ItemRow.id.in_(candidates), ItemRow.state == ItemState.PENDING)
                    .values(state=ItemState.INFLIGHT, submitted_at=timestamp)
                    .returning(ItemRow.id)
                    .execution_options(synchronize_session=False)
                )
                .scalars()
                .all()
            )
            if not claimed_ids:
                return []
            rows = session.execute(
                select(ItemRow)
                .join(BatchRow, ItemRow.batch_id == BatchRow.id)
                .where(ItemRow.id.in_(claimed_ids))
                .order_by(*order_by)
            ).scalars()
            return [_item_record(r) for r in rows]

    def record_item_success(
        self,
        item_id: str,
        result_json: dict,
        prompt_tokens: int,
        completion_tokens: int,
        vllm_request_id: str | None,
        *,
        now: datetime | None = None,
    ) -> ItemRecord:
        """Mark an INFLIGHT item SUCCEEDED and bump the batch's completed count.

        Only an INFLIGHT item is mutated. An item that is already terminal —
        because the expiry/cancel sweep closed it while the request was still
        in the engine, or because this result is a duplicate — is returned
        unchanged, so ``counts_completed`` can never drift above the number of
        items that actually succeeded.
        """
        with self._session.begin() as session:
            row = self._item_row(session, item_id)
            if row.state in _TERMINAL_ITEM_STATES:
                return _item_record(row)
            _check("item", item_id, row.state, ItemState.SUCCEEDED, _ITEM_TRANSITIONS)
            row.state = ItemState.SUCCEEDED
            row.result_json = result_json
            row.usage_prompt_tokens = prompt_tokens
            row.usage_completion_tokens = completion_tokens
            row.vllm_request_id = vllm_request_id
            row.finished_at = _now(now)
            self._batch_row(session, row.batch_id).counts_completed += 1
            return _item_record(row)

    def record_item_failure(
        self, item_id: str, error: str, *, retryable: bool, now: datetime | None = None
    ) -> ItemRecord:
        """Record an attempt failure.

        ``retryable=True`` and ``attempts < max_attempts`` sends the item back
        to PENDING; otherwise it becomes FAILED and the batch's failed count
        is incremented.

        Only PENDING/INFLIGHT items are mutated: a late failure for an item the
        sweep already expired or cancelled is returned unchanged rather than
        re-stating it and bumping ``counts_failed`` a second time.
        """
        with self._session.begin() as session:
            row = self._item_row(session, item_id)
            if row.state in _TERMINAL_ITEM_STATES:
                return _item_record(row)
            attempts = row.attempts + 1
            target = (
                ItemState.PENDING
                if retryable and attempts < self.max_attempts
                else ItemState.FAILED
            )
            _check("item", item_id, row.state, target, _ITEM_TRANSITIONS)
            row.attempts = attempts
            row.last_error = error
            row.state = target
            if target is ItemState.PENDING:
                row.submitted_at = None
            else:
                row.finished_at = _now(now)
                self._batch_row(session, row.batch_id).counts_failed += 1
            return _item_record(row)

    def requeue_inflight(self) -> int:
        """Crash recovery: return every result-less INFLIGHT item to PENDING."""
        with self._session.begin() as session:
            requeued = (
                session.execute(
                    update(ItemRow)
                    .where(ItemRow.state == ItemState.INFLIGHT, ItemRow.result_json.is_(None))
                    .values(state=ItemState.PENDING, submitted_at=None)
                    .returning(ItemRow.id)
                    .execution_options(synchronize_session=False)
                )
                .scalars()
                .all()
            )
            return len(requeued)

    def expire_batch(self, batch_id: str, *, now: datetime | None = None) -> BatchRecord:
        """Mark unfinished items EXPIRED and the batch EXPIRED."""
        timestamp = _now(now)
        with self._session.begin() as session:
            row = self._batch_row(session, batch_id)
            self._terminate_open_items(session, batch_id, ItemState.EXPIRED, timestamp)
            self._apply_batch_status(row, BatchStatus.EXPIRED, timestamp)
            return _batch_record(row)

    def cancel_batch(self, batch_id: str, *, now: datetime | None = None) -> BatchRecord:
        """Mark unfinished items CANCELLED and the batch CANCELLING.

        The caller flips the batch to CANCELLED once in-flight requests have
        actually been aborted.
        """
        timestamp = _now(now)
        with self._session.begin() as session:
            row = self._batch_row(session, batch_id)
            self._terminate_open_items(session, batch_id, ItemState.CANCELLED, timestamp)
            self._apply_batch_status(row, BatchStatus.CANCELLING, timestamp)
            return _batch_record(row)

    def batch_progress(self, batch_id: str) -> tuple[int, int, int]:
        """Live ``(total, completed, failed)`` counts computed from the items.

        Expired and cancelled items count towards neither completed nor
        failed, matching OpenAI's ``request_counts``.
        """
        with self._session() as session:
            self._batch_row(session, batch_id)
            return (
                self._count_items(session, batch_id),
                self._count_items(session, batch_id, ItemState.SUCCEEDED),
                self._count_items(session, batch_id, ItemState.FAILED),
            )

    def pending_and_inflight(self, batch_id: str) -> tuple[int, int]:
        """Return the batch's ``(pending, inflight)`` item counts."""
        with self._session() as session:
            self._batch_row(session, batch_id)
            return (
                self._count_items(session, batch_id, ItemState.PENDING),
                self._count_items(session, batch_id, ItemState.INFLIGHT),
            )

    def list_items(self, batch_id: str, states: list[ItemState] | None = None) -> list[ItemRecord]:
        """Return a batch's items in ``line_no`` order (added for A7).

        Args:
            states: keep only items in these states; ``None`` returns all of
                them. An unknown ``batch_id`` yields an empty list — result
                assembly polls this and must not care about races.
        """
        with self._session() as session:
            stmt = select(ItemRow).where(ItemRow.batch_id == batch_id)
            if states:
                stmt = stmt.where(ItemRow.state.in_(list(states)))
            stmt = stmt.order_by(ItemRow.line_no, ItemRow.seq)
            return [_item_record(r) for r in session.execute(stmt).scalars()]

    # -- metering --

    def record_usage(
        self,
        item_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        *,
        now: datetime | None = None,
    ) -> None:
        """Append a priced row to the usage ledger."""
        with self._session.begin() as session:
            session.add(
                UsageRow(
                    item_id=item_id,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                    created_at=_now(now),
                )
            )

    def usage_summary(self, since: datetime | None) -> list[dict]:
        """Aggregate the ledger per model, optionally from ``since`` onwards."""
        stmt = (
            select(
                UsageRow.model,
                func.sum(UsageRow.prompt_tokens),
                func.sum(UsageRow.completion_tokens),
                func.sum(UsageRow.cost_usd),
                func.count(UsageRow.seq),
            )
            .group_by(UsageRow.model)
            .order_by(UsageRow.model)
        )
        if since is not None:
            stmt = stmt.where(UsageRow.created_at >= _now(since))
        with self._session() as session:
            return [
                {
                    "model": model,
                    "prompt_tokens": int(ptoks or 0),
                    "completion_tokens": int(ctoks or 0),
                    "cost_usd": float(cost or 0.0),
                    "items": int(items or 0),
                }
                for model, ptoks, ctoks, cost, items in session.execute(stmt)
            ]

    def set_price(self, model: str, input_usd_per_mtok: float, output_usd_per_mtok: float) -> None:
        """Upsert a model's list price (USD per million tokens).

        Not part of the ``Repository`` protocol; the metering layer uses it to
        seed the price table without importing the ORM.
        """
        with self._session.begin() as session:
            row = session.get(PriceRow, model)
            if row is None:
                session.add(
                    PriceRow(
                        model=model,
                        input_usd_per_mtok=input_usd_per_mtok,
                        output_usd_per_mtok=output_usd_per_mtok,
                    )
                )
            else:
                row.input_usd_per_mtok = input_usd_per_mtok
                row.output_usd_per_mtok = output_usd_per_mtok

    def price_table(self) -> dict[str, tuple[float, float]]:
        """Return ``{model: (input_usd_per_mtok, output_usd_per_mtok)}``."""
        with self._session() as session:
            return {
                r.model: (r.input_usd_per_mtok, r.output_usd_per_mtok)
                for r in session.execute(select(PriceRow)).scalars()
            }

    # -- internals --

    @staticmethod
    def _claim_order(order: str) -> tuple:
        if order == "edf_prefix":
            return (BatchRow.expires_at, ItemRow.prefix_group, ItemRow.line_no)
        if order == "fifo":
            return (ItemRow.line_no, ItemRow.seq)
        raise ValueError(f"unknown claim order {order!r}; expected 'edf_prefix' or 'fifo'")

    @staticmethod
    def _find_batch(session, batch_id: str) -> BatchRow | None:
        return session.execute(select(BatchRow).where(BatchRow.id == batch_id)).scalar_one_or_none()

    @classmethod
    def _batch_row(cls, session, batch_id: str) -> BatchRow:
        row = cls._find_batch(session, batch_id)
        if row is None:
            raise KeyError(f"unknown batch {batch_id}")
        return row

    @staticmethod
    def _item_row(session, item_id: str) -> ItemRow:
        row = session.execute(select(ItemRow).where(ItemRow.id == item_id)).scalar_one_or_none()
        if row is None:
            raise KeyError(f"unknown item {item_id}")
        return row

    @staticmethod
    def _count_items(session, batch_id: str, state: ItemState | None = None) -> int:
        stmt = select(func.count(ItemRow.seq)).where(ItemRow.batch_id == batch_id)
        if state is not None:
            stmt = stmt.where(ItemRow.state == state)
        return int(session.execute(stmt).scalar_one())

    @staticmethod
    def _apply_batch_status(row: BatchRow, status: BatchStatus, timestamp: datetime) -> None:
        _check("batch", row.id, row.status, status, _BATCH_TRANSITIONS)
        row.status = status
        if status is BatchStatus.IN_PROGRESS and row.in_progress_at is None:
            # Stamped on the *first* transition only: the dispatcher re-asserts
            # IN_PROGRESS on later ticks and OpenAI's in_progress_at means "when
            # the batch started", not "when it was last touched".
            row.in_progress_at = timestamp
        elif status is BatchStatus.COMPLETED:
            row.completed_at = timestamp
        elif status is BatchStatus.CANCELLED:
            row.cancelled_at = timestamp

    @staticmethod
    def _terminate_open_items(
        session, batch_id: str, state: ItemState, timestamp: datetime
    ) -> None:
        session.execute(
            update(ItemRow)
            .where(
                ItemRow.batch_id == batch_id,
                ItemRow.state.in_([ItemState.PENDING, ItemState.INFLIGHT]),
            )
            .values(state=state, finished_at=timestamp)
            .execution_options(synchronize_session=False)
        )
