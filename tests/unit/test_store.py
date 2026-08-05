"""Unit tests for the SQLAlchemy store (Task A1).

Every test is deterministic: no sleeps, and anything time-sensitive is driven
through the explicit ``now=`` parameters rather than the wall clock.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest

from tidal.store.interfaces import BatchStatus, IllegalTransition, ItemState
from tidal.store.repo import SqlRepository

T0 = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def repo(tmp_path):
    """A fresh file-backed SQLite repository per test."""
    return SqlRepository(f"sqlite:///{tmp_path / 'tidal.db'}", str(tmp_path / "blobs"))


def _body(text: str = "hi") -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": text}]}


def _batch(repo, *, window_hours: int = 24, items=None, now: datetime = T0):
    lines = b"".join(json.dumps({"custom_id": c}).encode() + b"\n" for c, *_ in (items or []))
    f = repo.create_file("batch", "in.jsonl", lines or b"{}\n", now=now)
    return repo.create_batch(
        f.id,
        "/v1/chat/completions",
        window_hours,
        {},
        items=items or [("a", 1, 0, _body())],
        now=now,
    )


# -- files ---------------------------------------------------------------


def test_create_file_hashes_and_persists_blob(repo):
    content = b'{"custom_id":"a"}\n'
    f = repo.create_file("batch", "in.jsonl", content, now=T0)

    assert f.id.startswith("file-")
    assert f.purpose == "batch"
    assert f.size_bytes == len(content)
    assert f.sha256 == hashlib.sha256(content).hexdigest()
    assert repo.read_file_content(f.id) == content
    assert repo.get_file(f.id) == f
    assert repo.get_file("file-nope") is None
    assert f.created_at.tzinfo is not None


def test_sqlite_uses_wal_journal(repo, tmp_path):
    repo.create_file("batch", "in.jsonl", b"x", now=T0)
    with sqlite3.connect(tmp_path / "tidal.db") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


# -- batches -------------------------------------------------------------


def test_create_batch_with_items_roundtrip(repo):
    f = repo.create_file("batch", "in.jsonl", b'{"custom_id":"a"}\n', now=T0)
    b = repo.create_batch(
        f.id,
        "/v1/chat/completions",
        24,
        {"k": "v"},
        items=[("a", 1, 0, {"model": "m", "messages": []})],
        now=T0,
    )

    assert b.status is BatchStatus.VALIDATING
    assert b.counts_total == 1
    assert b.counts_completed == 0 and b.counts_failed == 0
    assert b.created_at == T0
    assert b.expires_at == T0 + timedelta(hours=24)

    got = repo.get_batch(b.id)
    assert got.metadata == {"k": "v"}
    assert got.input_file_id == f.id
    assert got.endpoint == "/v1/chat/completions"
    assert repo.get_batch("batch-nope") is None

    (item,) = repo.claim_pending_items(limit=10, now=T0)
    assert item.custom_id == "a"
    assert item.request_json == {"model": "m", "messages": []}
    assert item.attempts == 0


def test_list_batches_paginates_newest_first(repo):
    ids = [_batch(repo, now=T0 + timedelta(minutes=i)).id for i in range(3)]

    page1 = repo.list_batches(limit=2, after=None)
    assert [b.id for b in page1] == ids[::-1][:2]

    page2 = repo.list_batches(limit=2, after=page1[-1].id)
    assert [b.id for b in page2] == ids[::-1][2:]


def test_find_batches_by_metadata(repo):
    f = repo.create_file("batch", "in.jsonl", b"{}\n", now=T0)
    a = repo.create_batch(
        f.id, "/v1/chat/completions", 24, {"tenant": "x"}, items=[("a", 1, 0, _body())], now=T0
    )
    repo.create_batch(
        f.id, "/v1/chat/completions", 24, {"tenant": "y"}, items=[("b", 1, 0, _body())], now=T0
    )

    assert [b.id for b in repo.find_batches_by_metadata("tenant", "x")] == [a.id]
    assert repo.find_batches_by_metadata("tenant", "zzz") == []


# -- claiming ------------------------------------------------------------


def test_claim_orders_by_edf_then_prefix_then_line(repo):
    b_late = _batch(
        repo,
        window_hours=24,
        items=[("late-1", 1, 1, _body()), ("late-2", 2, 0, _body())],
        now=T0,
    )
    b_soon = _batch(repo, window_hours=1, items=[("soon", 1, 0, _body())], now=T0)

    claimed = repo.claim_pending_items(limit=3, now=T0)

    assert claimed[0].batch_id == b_soon.id
    assert [i.prefix_group for i in claimed[1:]] == [0, 1]
    assert [i.custom_id for i in claimed[1:]] == ["late-2", "late-1"]
    assert all(i.batch_id == b_late.id for i in claimed[1:])
    assert all(i.state is ItemState.INFLIGHT for i in claimed)
    assert all(i.submitted_at == T0 for i in claimed)


def test_claim_is_atomic_and_respects_limit(repo):
    _batch(repo, items=[(f"c{i}", i, 0, _body()) for i in range(5)], now=T0)

    first = repo.claim_pending_items(limit=2, now=T0)
    second = repo.claim_pending_items(limit=10, now=T0)

    assert len(first) == 2
    assert len(second) == 3
    assert not {i.id for i in first} & {i.id for i in second}
    assert repo.claim_pending_items(limit=10, now=T0) == []


def test_claim_fifo_order_uses_line_no_only(repo):
    _batch(repo, window_hours=24, items=[("x", 1, 9, _body()), ("y", 2, 0, _body())], now=T0)

    claimed = repo.claim_pending_items(limit=2, order="fifo", now=T0)
    assert [i.custom_id for i in claimed] == ["x", "y"]


def test_claim_rejects_unknown_order(repo):
    with pytest.raises(ValueError):
        repo.claim_pending_items(limit=1, order="random")


# -- item outcomes -------------------------------------------------------


def test_record_item_success_stores_result_and_bumps_counts(repo):
    b = _batch(repo, items=[("a", 1, 0, _body()), ("b", 2, 0, _body())], now=T0)
    a, _ = repo.claim_pending_items(limit=2, now=T0)

    done = repo.record_item_success(
        a.id, {"choices": []}, 11, 7, "req-1", now=T0 + timedelta(seconds=5)
    )

    assert done.state is ItemState.SUCCEEDED
    assert done.result_json == {"choices": []}
    assert (done.usage_prompt_tokens, done.usage_completion_tokens) == (11, 7)
    assert done.vllm_request_id == "req-1"
    assert done.finished_at == T0 + timedelta(seconds=5)
    assert repo.get_batch(b.id).counts_completed == 1


def test_record_item_success_twice_does_not_drift_counts(repo):
    """A second success for the same item is a no-op, not a double count.

    There is no ownership lease on the dispatcher, so a duplicate submission
    (or a retried hook) must not be able to inflate ``counts_completed`` past
    the number of items that actually succeeded.
    """
    b = _batch(repo, items=[("a", 1, 0, _body())], now=T0)
    (item,) = repo.claim_pending_items(limit=1, now=T0)
    first = repo.record_item_success(item.id, {"choices": ["one"]}, 11, 7, "req-1", now=T0)

    again = repo.record_item_success(
        item.id, {"choices": ["two"]}, 999, 999, "req-2", now=T0 + timedelta(minutes=1)
    )

    assert again.state is ItemState.SUCCEEDED
    assert again.result_json == {"choices": ["one"]}  # the first result stands
    assert (again.usage_prompt_tokens, again.usage_completion_tokens) == (11, 7)
    assert again.vllm_request_id == "req-1"
    assert again.finished_at == first.finished_at
    assert repo.get_batch(b.id).counts_completed == 1
    assert repo.batch_progress(b.id) == (1, 1, 0)


def test_late_success_for_an_expired_item_is_ignored(repo):
    """The sweep already closed this item; the engine's answer arrives after."""
    b = _batch(repo, window_hours=1, items=[("a", 1, 0, _body())], now=T0)
    (item,) = repo.claim_pending_items(limit=1, now=T0)
    repo.set_batch_status(b.id, BatchStatus.IN_PROGRESS, now=T0)
    repo.expire_batch(b.id, now=T0 + timedelta(hours=2))

    late = repo.record_item_success(
        item.id, {"choices": []}, 5, 5, "req-late", now=T0 + timedelta(hours=2)
    )

    assert late.state is ItemState.EXPIRED
    assert late.result_json is None
    assert repo.get_batch(b.id).counts_completed == 0
    assert repo.batch_progress(b.id) == (1, 0, 0)


def test_record_item_failure_on_a_terminal_item_is_ignored(repo):
    b = _batch(repo, items=[("a", 1, 0, _body()), ("b", 2, 0, _body())], now=T0)
    failed, cancelled = repo.claim_pending_items(limit=2, now=T0)
    repo.record_item_failure(failed.id, "boom", retryable=False, now=T0)
    assert repo.get_batch(b.id).counts_failed == 1

    again = repo.record_item_failure(failed.id, "boom again", retryable=False, now=T0)
    assert again.state is ItemState.FAILED
    assert again.attempts == 1
    assert again.last_error == "boom"
    assert repo.get_batch(b.id).counts_failed == 1

    # A cancelled item is terminal too: a late failure must not turn it into a
    # FAILED item and must not count towards request_counts.failed.
    repo.set_batch_status(b.id, BatchStatus.IN_PROGRESS, now=T0)
    repo.cancel_batch(b.id, now=T0)
    late = repo.record_item_failure(cancelled.id, "engine 500", retryable=True, now=T0)
    assert late.state is ItemState.CANCELLED
    assert late.attempts == 0
    assert repo.get_batch(b.id).counts_failed == 1


def test_retryable_failure_requeues_until_max_attempts(repo):
    _batch(repo, items=[("a", 1, 0, _body())], now=T0)

    for expected_attempts in (1, 2):
        (item,) = repo.claim_pending_items(limit=1, now=T0)
        out = repo.record_item_failure(item.id, "boom", retryable=True, now=T0)
        assert out.state is ItemState.PENDING
        assert out.attempts == expected_attempts
        assert out.last_error == "boom"
        assert out.submitted_at is None

    (item,) = repo.claim_pending_items(limit=1, now=T0)
    out = repo.record_item_failure(item.id, "boom", retryable=True, now=T0)
    assert out.state is ItemState.FAILED
    assert out.attempts == 3
    assert repo.claim_pending_items(limit=1, now=T0) == []


def test_non_retryable_failure_fails_immediately(repo):
    b = _batch(repo, items=[("a", 1, 0, _body())], now=T0)
    (item,) = repo.claim_pending_items(limit=1, now=T0)

    out = repo.record_item_failure(item.id, "400 bad request", retryable=False, now=T0)

    assert out.state is ItemState.FAILED
    assert out.attempts == 1
    assert repo.get_batch(b.id).counts_failed == 1


def test_max_attempts_is_configurable(tmp_path):
    repo = SqlRepository(
        f"sqlite:///{tmp_path / 'tidal.db'}", str(tmp_path / "blobs"), max_attempts=1
    )
    _batch(repo, items=[("a", 1, 0, _body())], now=T0)
    (item,) = repo.claim_pending_items(limit=1, now=T0)

    out = repo.record_item_failure(item.id, "boom", retryable=True, now=T0)
    assert out.state is ItemState.FAILED


def test_requeue_inflight_recovers_crash(repo):
    _batch(repo, items=[("a", 1, 0, _body()), ("b", 2, 0, _body())], now=T0)
    a, _ = repo.claim_pending_items(limit=2, now=T0)
    repo.record_item_success(a.id, {"choices": []}, 1, 1, None, now=T0)

    assert repo.requeue_inflight() == 1
    assert repo.requeue_inflight() == 0

    (again,) = repo.claim_pending_items(limit=5, now=T0)
    assert again.custom_id == "b"
    assert again.attempts == 0


# -- batch lifecycle -----------------------------------------------------


def test_expire_batch_marks_remaining_and_batch(repo):
    b = _batch(
        repo,
        window_hours=1,
        items=[("a", 1, 0, _body()), ("b", 2, 0, _body()), ("c", 3, 0, _body())],
        now=T0,
    )
    a, _b2 = repo.claim_pending_items(limit=2, now=T0)
    repo.record_item_success(a.id, {"choices": []}, 1, 1, None, now=T0)
    repo.set_batch_status(b.id, BatchStatus.IN_PROGRESS, now=T0)

    after = T0 + timedelta(hours=2)
    out = repo.expire_batch(b.id, now=after)

    assert out.status is BatchStatus.EXPIRED
    assert repo.pending_and_inflight(b.id) == (0, 0)
    assert repo.batch_progress(b.id) == (3, 1, 0)
    assert repo.claim_pending_items(limit=5, now=after) == []


def test_cancel_flow_and_illegal_transition_raises(repo):
    b = _batch(repo, items=[("a", 1, 0, _body()), ("b", 2, 0, _body())], now=T0)
    (a,) = repo.claim_pending_items(limit=1, now=T0)
    repo.set_batch_status(b.id, BatchStatus.IN_PROGRESS, now=T0)

    cancelling = repo.cancel_batch(b.id, now=T0)
    assert cancelling.status is BatchStatus.CANCELLING
    assert repo.pending_and_inflight(b.id) == (0, 0)

    cancelled = repo.set_batch_status(b.id, BatchStatus.CANCELLED, now=T0 + timedelta(seconds=1))
    assert cancelled.status is BatchStatus.CANCELLED
    assert cancelled.cancelled_at == T0 + timedelta(seconds=1)

    with pytest.raises(IllegalTransition):
        repo.set_batch_status(b.id, BatchStatus.IN_PROGRESS)

    # A result that lands after the cancel sweep is dropped, not an error: the
    # item is already terminal and its counts are already final.
    stale = repo.record_item_success(a.id, {"choices": []}, 1, 1, None, now=T0)
    assert stale.state is ItemState.CANCELLED
    assert repo.get_batch(b.id).counts_completed == 0

    completed = _batch(repo, items=[("z", 1, 0, _body())], now=T0)
    repo.set_batch_status(completed.id, BatchStatus.IN_PROGRESS, now=T0)
    repo.set_batch_status(completed.id, BatchStatus.COMPLETED, now=T0)
    with pytest.raises(IllegalTransition):
        repo.set_batch_status(completed.id, BatchStatus.IN_PROGRESS)


def test_finalize_batch_attaches_output_files(repo):
    b = _batch(repo, items=[("a", 1, 0, _body())], now=T0)
    out_f = repo.create_file("batch_output", "out.jsonl", b"{}\n", now=T0)
    err_f = repo.create_file("batch_error", "err.jsonl", b"{}\n", now=T0)
    repo.set_batch_status(b.id, BatchStatus.IN_PROGRESS, now=T0)

    done = repo.finalize_batch(
        b.id, BatchStatus.COMPLETED, out_f.id, err_f.id, now=T0 + timedelta(minutes=5)
    )

    assert done.status is BatchStatus.COMPLETED
    assert done.output_file_id == out_f.id
    assert done.error_file_id == err_f.id
    assert done.completed_at == T0 + timedelta(minutes=5)


def test_finalize_batch_does_not_overwrite_an_existing_claim(repo):
    """Attaching result files is a claim: first writer wins, rest are no-ops."""
    b = _batch(repo, items=[("a", 1, 0, _body())], now=T0)
    first_out = repo.create_file("batch_output", "out.jsonl", b"{}\n", now=T0)
    second_out = repo.create_file("batch_output", "out2.jsonl", b"{}\n", now=T0)
    second_err = repo.create_file("batch_error", "err2.jsonl", b"{}\n", now=T0)
    repo.set_batch_status(b.id, BatchStatus.IN_PROGRESS, now=T0)

    won = repo.finalize_batch(b.id, BatchStatus.COMPLETED, first_out.id, None, now=T0)
    lost = repo.finalize_batch(
        b.id,
        BatchStatus.COMPLETED,
        second_out.id,
        second_err.id,
        now=T0 + timedelta(minutes=1),
    )

    assert won.output_file_id == first_out.id
    assert lost.output_file_id == first_out.id
    assert lost.error_file_id is None
    assert lost.completed_at == won.completed_at
    stored = repo.get_batch(b.id)
    assert (stored.output_file_id, stored.error_file_id) == (first_out.id, None)


def test_concurrent_finalize_writes_one_set_of_file_ids(repo):
    """Two threads racing to finalize must not interleave their file ids."""
    b = _batch(repo, items=[("a", 1, 0, _body())], now=T0)
    repo.set_batch_status(b.id, BatchStatus.IN_PROGRESS, now=T0)
    pairs = [
        (
            repo.create_file("batch_output", f"out{i}.jsonl", b"{}\n", now=T0).id,
            repo.create_file("batch_error", f"err{i}.jsonl", b"{}\n", now=T0).id,
        )
        for i in range(2)
    ]

    barrier = threading.Barrier(len(pairs))
    winners: list[tuple[str | None, str | None]] = []
    errors: list[Exception] = []

    def finalize(out_id: str, err_id: str) -> None:
        barrier.wait(timeout=5)
        try:
            rec = repo.finalize_batch(b.id, BatchStatus.COMPLETED, out_id, err_id, now=T0)
            winners.append((rec.output_file_id, rec.error_file_id))
        except Exception as exc:  # a losing writer may hit the store's lock
            errors.append(exc)

    threads = [threading.Thread(target=finalize, args=pair) for pair in pairs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    stored = repo.get_batch(b.id)
    assert (stored.output_file_id, stored.error_file_id) in pairs
    assert winners, f"both writers failed: {errors}"
    # Whoever returned did so with the *stored* pair — never a mix of the two.
    assert set(winners) == {(stored.output_file_id, stored.error_file_id)}
    assert stored.status is BatchStatus.COMPLETED


def test_progress_counts_live(repo):
    b = _batch(repo, items=[(f"c{i}", i, 0, _body()) for i in range(4)], now=T0)

    assert repo.batch_progress(b.id) == (4, 0, 0)
    assert repo.pending_and_inflight(b.id) == (4, 0)

    claimed = repo.claim_pending_items(limit=3, now=T0)
    assert repo.pending_and_inflight(b.id) == (1, 3)

    repo.record_item_success(claimed[0].id, {"choices": []}, 1, 1, None, now=T0)
    repo.record_item_failure(claimed[1].id, "boom", retryable=False, now=T0)

    assert repo.batch_progress(b.id) == (4, 1, 1)
    assert repo.pending_and_inflight(b.id) == (1, 1)

    stored = repo.get_batch(b.id)
    assert (stored.counts_total, stored.counts_completed, stored.counts_failed) == (4, 1, 1)


# -- metering ------------------------------------------------------------


def test_usage_summary_aggregates_by_model(repo):
    _batch(repo, items=[("a", 1, 0, _body()), ("b", 2, 0, _body())], now=T0)
    a, b = repo.claim_pending_items(limit=2, now=T0)
    repo.record_usage(a.id, "m", 100, 50, 0.001, now=T0)
    repo.record_usage(b.id, "m", 200, 25, 0.002, now=T0 + timedelta(hours=1))

    (row,) = repo.usage_summary(None)
    assert row["model"] == "m"
    assert row["prompt_tokens"] == 300
    assert row["completion_tokens"] == 75
    assert row["items"] == 2
    assert row["cost_usd"] == pytest.approx(0.003)

    (recent,) = repo.usage_summary(T0 + timedelta(minutes=30))
    assert recent["items"] == 1 and recent["prompt_tokens"] == 200
