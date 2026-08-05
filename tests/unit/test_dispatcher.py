"""Dispatcher loop + vLLM client unit tests.

Every test drives ``Dispatcher.tick(now)`` with an injected clock and fake
collaborators: no sleeping, no wall clock, no engine. The fakes live here (not
in ``tests/unit/fakes.py``) because they model dispatcher-specific behaviour —
notably the store's transition guards, which is how these tests prove the
dispatcher flips a batch VALIDATING → IN_PROGRESS before anything else can
finalize it.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from tidal.config import TidalConfig
from tidal.dispatcher.loop import Dispatcher, TickReport
from tidal.dispatcher.vllm_client import (
    EngineDown,
    EngineMetrics,
    FatalUpstream,
    RetryableUpstream,
    VllmClient,
    parse_metrics,
)
from tidal.store.interfaces import (
    BatchRecord,
    BatchStatus,
    IllegalTransition,
    ItemRecord,
    ItemState,
)

T0 = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)

_LEGAL_BATCH: dict[BatchStatus, frozenset[BatchStatus]] = {
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

_OPEN_ITEM = (ItemState.PENDING, ItemState.INFLIGHT)
_TERMINAL_ITEM = frozenset(
    {ItemState.SUCCEEDED, ItemState.FAILED, ItemState.EXPIRED, ItemState.CANCELLED}
)


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeRepo:
    """In-memory subset of ``Repository`` — the methods the dispatcher calls."""

    def __init__(self, max_attempts: int = 3) -> None:
        self.batches: dict[str, BatchRecord] = {}
        self.items: dict[str, list[ItemRecord]] = {}
        self.max_attempts = max_attempts
        self.requeue_calls = 0
        self.page_sizes: list[int] = []

    # -- test helpers -------------------------------------------------------

    def add_batch(
        self,
        n_items: int,
        *,
        created_at: datetime = T0,
        window_hours: int = 24,
        endpoint: str = "/v1/chat/completions",
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ) -> BatchRecord:
        batch_id = f"batch-{uuid.uuid4().hex[:8]}"
        rec = BatchRecord(
            id=batch_id,
            input_file_id="file-in",
            endpoint=endpoint,
            status=BatchStatus.VALIDATING,
            created_at=created_at,
            expires_at=created_at + timedelta(hours=window_hours),
            counts_total=n_items,
            counts_completed=0,
            counts_failed=0,
        )
        self.batches[batch_id] = rec
        self.items[batch_id] = [
            ItemRecord(
                id=f"item-{uuid.uuid4().hex[:8]}",
                batch_id=batch_id,
                custom_id=f"c{line}",
                line_no=line,
                prefix_group=0,
                request_json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"q{line}"}],
                },
                state=ItemState.PENDING,
            )
            for line in range(n_items)
        ]
        return rec

    def item_states(self, batch_id: str) -> list[ItemState]:
        return [i.state for i in self.items[batch_id]]

    def _item(self, item_id: str) -> ItemRecord:
        for items in self.items.values():
            for item in items:
                if item.id == item_id:
                    return item
        raise KeyError(item_id)

    @staticmethod
    def _guard(current: BatchStatus, target: BatchStatus) -> None:
        if current is target:
            return
        if target not in _LEGAL_BATCH[current]:
            raise IllegalTransition(f"{current.value} -> {target.value}")

    # -- Repository surface -------------------------------------------------

    def list_batches(self, limit: int, after: str | None) -> list[BatchRecord]:
        self.page_sizes.append(limit)
        ordered = sorted(self.batches.values(), key=lambda b: (b.created_at, b.id), reverse=True)
        if after is not None:
            ids = [b.id for b in ordered]
            ordered = ordered[ids.index(after) + 1 :] if after in ids else []
        return ordered[:limit]

    def get_batch(self, batch_id: str) -> BatchRecord | None:
        return self.batches.get(batch_id)

    def set_batch_status(
        self, batch_id: str, status: BatchStatus, *, now: datetime | None = None
    ) -> BatchRecord:
        rec = self.batches[batch_id]
        self._guard(rec.status, status)
        if status is BatchStatus.IN_PROGRESS and rec.in_progress_at is None:
            rec.in_progress_at = now
        rec.status = status
        return rec

    def finalize_batch(
        self,
        batch_id: str,
        status: BatchStatus,
        output_file_id: str | None,
        error_file_id: str | None,
        *,
        now: datetime | None = None,
    ) -> BatchRecord:
        rec = self.set_batch_status(batch_id, status, now=now)
        rec.output_file_id = output_file_id
        rec.error_file_id = error_file_id
        return rec

    def claim_pending_items(
        self, limit: int, order: str = "edf_prefix", *, now: datetime | None = None
    ) -> list[ItemRecord]:
        if limit <= 0:
            return []
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
            item.submitted_at = now
            claimed.append(item)
        return claimed

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
        item = self._item(item_id)
        if item.state in _TERMINAL_ITEM:
            return item  # already closed by a sweep: no mutation, no counts
        if item.state is not ItemState.INFLIGHT:
            raise IllegalTransition(f"{item.state.value} -> succeeded")
        item.state = ItemState.SUCCEEDED
        item.result_json = result_json
        item.usage_prompt_tokens = prompt_tokens
        item.usage_completion_tokens = completion_tokens
        item.vllm_request_id = vllm_request_id
        item.finished_at = now
        self.batches[item.batch_id].counts_completed += 1
        return item

    def record_item_failure(
        self, item_id: str, error: str, *, retryable: bool, now: datetime | None = None
    ) -> ItemRecord:
        item = self._item(item_id)
        if item.state in _TERMINAL_ITEM:
            return item  # already closed by a sweep: no mutation, no counts
        if item.state is not ItemState.INFLIGHT:
            raise IllegalTransition(f"{item.state.value} -> failed")
        item.attempts += 1
        item.last_error = error
        if retryable and item.attempts < self.max_attempts:
            item.state = ItemState.PENDING
            item.submitted_at = None
        else:
            item.state = ItemState.FAILED
            item.finished_at = now
            self.batches[item.batch_id].counts_failed += 1
        return item

    def requeue_inflight(self) -> int:
        self.requeue_calls += 1
        n = 0
        for items in self.items.values():
            for item in items:
                if item.state is ItemState.INFLIGHT and item.result_json is None:
                    item.state = ItemState.PENDING
                    item.submitted_at = None
                    n += 1
        return n

    def expire_batch(self, batch_id: str, *, now: datetime | None = None) -> BatchRecord:
        for item in self.items[batch_id]:
            if item.state in _OPEN_ITEM:
                item.state = ItemState.EXPIRED
                item.finished_at = now
        return self.set_batch_status(batch_id, BatchStatus.EXPIRED, now=now)

    def cancel_batch(self, batch_id: str, *, now: datetime | None = None) -> BatchRecord:
        for item in self.items[batch_id]:
            if item.state in _OPEN_ITEM:
                item.state = ItemState.CANCELLED
                item.finished_at = now
        return self.set_batch_status(batch_id, BatchStatus.CANCELLING, now=now)

    def pending_and_inflight(self, batch_id: str) -> tuple[int, int]:
        items = self.items.get(batch_id, [])
        return (
            sum(1 for i in items if i.state is ItemState.PENDING),
            sum(1 for i in items if i.state is ItemState.INFLIGHT),
        )

    def batch_progress(self, batch_id: str) -> tuple[int, int, int]:
        items = self.items.get(batch_id, [])
        return (
            len(items),
            sum(1 for i in items if i.state is ItemState.SUCCEEDED),
            sum(1 for i in items if i.state is ItemState.FAILED),
        )


class FakeClient:
    """Scriptable stand-in for :class:`VllmClient`."""

    def __init__(self, *, running: int = 0, waiting: int = 0, kv_usage: float = 0.10) -> None:
        self.metrics = EngineMetrics(running=running, waiting=waiting, kv_usage=kv_usage)
        self.scrape_error: Exception | None = None
        self.chat_error: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.calls: list[tuple[dict, int, str]] = []
        self.scrapes = 0

    def load(self, *, running: int = 0, waiting: int = 0, kv_usage: float = 0.10) -> None:
        self.metrics = EngineMetrics(running=running, waiting=waiting, kv_usage=kv_usage)

    async def scrape(self) -> EngineMetrics:
        self.scrapes += 1
        if self.scrape_error is not None:
            raise self.scrape_error
        return self.metrics

    async def chat(
        self, body: dict, priority: int, *, endpoint: str = "/v1/chat/completions"
    ) -> tuple[dict, int, int]:
        self.calls.append((body, priority, endpoint))
        if self.gate is not None:
            await self.gate.wait()
        if self.chat_error is not None:
            raise self.chat_error
        n = len(self.calls)
        return {"id": f"cmpl-{n}", "choices": [{"index": 0, "message": {"content": "ok"}}]}, 11, 7

    @property
    def priorities(self) -> list[int]:
        return [priority for _body, priority, _ep in self.calls]


class FakeFinalizer:
    """Mimics A7's ``finalize_if_done``: a no-op until a batch is fully drained."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.finalized: list[str] = []

    def __call__(self, repo: FakeRepo, batch_id: str) -> BatchRecord | None:
        self.calls.append(batch_id)
        batch = repo.get_batch(batch_id)
        if batch is None or batch.status in (BatchStatus.COMPLETED, BatchStatus.CANCELLED):
            return None
        pending, inflight = repo.pending_and_inflight(batch_id)
        if pending or inflight:
            return None
        if batch.status is BatchStatus.CANCELLING:
            rec = repo.finalize_batch(batch_id, BatchStatus.CANCELLED, "file-out", None)
            self.finalized.append(batch_id)
            return rec
        if batch.status is BatchStatus.EXPIRED:
            batch.output_file_id = "file-out"
            self.finalized.append(batch_id)
            return batch
        rec = repo.finalize_batch(batch_id, BatchStatus.COMPLETED, "file-out", None)
        self.finalized.append(batch_id)
        return rec


def make_dispatcher(
    repo: FakeRepo,
    client: FakeClient,
    cfg: TidalConfig | None = None,
    **kwargs,
) -> Dispatcher:
    return Dispatcher(cfg or TidalConfig(), repo, client, **kwargs)


async def run_tick(disp: Dispatcher, now: datetime) -> TickReport:
    """One tick, then let every submitted item settle (fakes finish instantly)."""
    report = await disp.tick(now)
    await disp.drain()
    return report


# --------------------------------------------------------------------------
# startup / lifecycle
# --------------------------------------------------------------------------


async def test_startup_requeues_inflight_items():
    repo = FakeRepo()
    batch = repo.add_batch(2)
    repo.items[batch.id][0].state = ItemState.INFLIGHT  # simulated crash

    disp = make_dispatcher(repo, FakeClient())
    assert await disp.startup() == 1
    assert repo.item_states(batch.id) == [ItemState.PENDING, ItemState.PENDING]


async def test_run_survives_tick_exceptions():
    repo, client = FakeRepo(), FakeClient()
    disp = make_dispatcher(repo, client, TidalConfig(poll_interval_s=0.001))
    seen: list[datetime] = []

    async def flaky(now: datetime) -> TickReport:
        seen.append(now)
        if len(seen) == 1:
            raise RuntimeError("engine exploded mid-tick")
        return TickReport(target=0, submitted=0, inflight=0)

    disp.tick = flaky  # type: ignore[method-assign]
    task = asyncio.create_task(disp.run())
    for _ in range(500):
        await asyncio.sleep(0.001)
        if len(seen) >= 3:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(seen) >= 3, "run() must keep ticking after a tick raises"
    assert repo.requeue_calls == 1, "run() performs crash recovery once at startup"


# --------------------------------------------------------------------------
# fill / AIMD
# --------------------------------------------------------------------------


async def test_tick_fills_to_target_with_correct_priorities():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    batch = repo.add_batch(10)
    disp = make_dispatcher(repo, client)

    reports = [await run_tick(disp, T0 + timedelta(seconds=i)) for i in (1, 2, 3)]

    assert [r.target for r in reports] == [1, 2, 3]  # additive increase
    assert [r.submitted for r in reports] == [1, 2, 3]  # filled to target each tick
    assert len(client.calls) == 6
    assert set(client.priorities) == {100}  # on-track → zero interference
    assert reports[-1].escalated_priorities[batch.id] == 100
    assert repo.batches[batch.id].status is BatchStatus.IN_PROGRESS
    assert repo.batches[batch.id].in_progress_at == T0 + timedelta(seconds=1)
    assert repo.batch_progress(batch.id) == (10, 6, 0)


async def test_tick_halves_target_on_online_queueing_spike():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    disp = make_dispatcher(repo, client, TidalConfig(max_inflight=16))

    for i in range(4):
        await run_tick(disp, T0 + timedelta(seconds=i))
    assert disp.aimd.target == 4

    client.load(waiting=8, kv_usage=0.10)  # online requests queue up
    report = await run_tick(disp, T0 + timedelta(seconds=4))
    assert report.target == 2

    report = await run_tick(disp, T0 + timedelta(seconds=5))
    assert report.target == 1


async def test_ewma_smooths_a_single_kv_spike_but_not_sustained_load():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    disp = make_dispatcher(repo, client, TidalConfig(max_inflight=16))

    for i in range(4):
        await run_tick(disp, T0 + timedelta(seconds=i))
    assert disp.aimd.target == 4

    client.load(kv_usage=0.95)  # one-tick blip
    spike = await run_tick(disp, T0 + timedelta(seconds=4))
    assert spike.target == 5, "a single KV sample above kv_high must not collapse the target"
    assert disp.metrics is not None and 0.5 <= disp.metrics.kv_usage <= 0.55

    targets = [(await run_tick(disp, T0 + timedelta(seconds=5 + i))).target for i in range(6)]
    assert min(targets) < 5, "sustained high KV must eventually halve the target"


# --------------------------------------------------------------------------
# laxity escalation
# --------------------------------------------------------------------------


async def test_at_risk_batch_escalates_priority_and_on_track_batch_does_not():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    at_risk = repo.add_batch(500, created_at=T0, window_hours=24)
    disp = make_dispatcher(repo, client)

    report = await run_tick(disp, T0 + timedelta(hours=23))  # 1h of slack, 500 items left
    assert report.escalated_priorities[at_risk.id] <= 50
    assert client.priorities, "the at-risk batch must actually be submitted"
    assert set(client.priorities) == {report.escalated_priorities[at_risk.id]}

    fresh_repo, fresh_client = FakeRepo(), FakeClient(kv_usage=0.10)
    on_track = fresh_repo.add_batch(500, created_at=T0, window_hours=24)
    fresh = make_dispatcher(fresh_repo, fresh_client)
    early = await run_tick(fresh, T0 + timedelta(minutes=1))
    assert early.escalated_priorities[on_track.id] == 100
    assert fresh_client.priorities == [100]

    # Halfway through the 24h window with a healthy drain rate is still "on
    # track": 12h of slack is twice the escalation horizon, so the batch must
    # not escalate at all. (A window-wide urgency ramp would put it at ~51.)
    midway_repo, midway_client = FakeRepo(), FakeClient(kv_usage=0.10)
    midway_batch = midway_repo.add_batch(10, created_at=T0, window_hours=24)
    midway = make_dispatcher(midway_repo, midway_client)
    for i in range(3):
        report = await run_tick(midway, T0 + timedelta(hours=12, seconds=i))
    assert report.escalated_priorities[midway_batch.id] == 100
    assert set(midway_client.priorities) == {100}


async def test_escalation_floor_keeps_items_flowing_under_sustained_load():
    cfg = TidalConfig(max_inflight=4, floor_items_per_s=0.2, floor_urgency=0.9)
    repo, client = FakeRepo(), FakeClient(kv_usage=0.95, waiting=3)
    at_risk = repo.add_batch(20, created_at=T0, window_hours=24)
    disp = make_dispatcher(repo, client, cfg)

    # 30 minutes of slack against a 6h escalation horizon → urgency ~0.92,
    # above floor_urgency, so the token bucket must keep the batch draining.
    start = T0 + timedelta(hours=23, minutes=30)
    for i in range(10):
        await run_tick(disp, start + timedelta(seconds=i))

    assert len(client.calls) >= 2, "token-bucket floor must keep the batch draining"
    assert repo.batch_progress(at_risk.id)[1] == len(client.calls)

    # control: same crushing load, but the batch is nowhere near its deadline
    calm_repo, calm_client = FakeRepo(), FakeClient(kv_usage=0.95, waiting=3)
    calm_repo.add_batch(20, created_at=T0, window_hours=24)
    calm = make_dispatcher(calm_repo, calm_client, cfg)
    for i in range(10):
        await run_tick(calm, T0 + timedelta(seconds=i))
    assert calm_client.calls == [], "an on-track batch yields completely to online load"


# --------------------------------------------------------------------------
# completion hooks
# --------------------------------------------------------------------------


async def test_success_records_usage_and_finalizes_the_batch_once():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    batch = repo.add_batch(2)
    metered: list[ItemRecord] = []
    finalizer = FakeFinalizer()
    disp = make_dispatcher(repo, client, on_success=metered.append, finalize=finalizer)

    await run_tick(disp, T0 + timedelta(seconds=1))  # target 1 → one item
    await run_tick(disp, T0 + timedelta(seconds=2))  # target 2 → the last item

    assert repo.item_states(batch.id) == [ItemState.SUCCEEDED, ItemState.SUCCEEDED]
    assert [i.usage_prompt_tokens for i in metered] == [11, 11]
    assert [i.usage_completion_tokens for i in metered] == [7, 7]
    assert [i.vllm_request_id for i in metered] == ["cmpl-1", "cmpl-2"]
    assert finalizer.calls == [batch.id, batch.id]  # probed after every terminal item
    assert finalizer.finalized == [batch.id]  # finalized exactly once
    assert repo.batches[batch.id].status is BatchStatus.COMPLETED
    assert repo.batches[batch.id].output_file_id == "file-out"


async def test_finalize_hook_may_close_over_the_repository():
    """The CLI wires ``finalize(batch_id)``; the contract shape is
    ``finalize(repo, batch_id)``. Both must work — a hook that quietly raised
    TypeError on every completion would mean batches never finalize."""
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    batch = repo.add_batch(1)
    seen: list[str] = []
    disp = make_dispatcher(repo, client, finalize=lambda batch_id: seen.append(batch_id))

    await run_tick(disp, T0 + timedelta(seconds=1))

    assert seen == [batch.id]


async def test_late_result_for_a_cancelled_item_is_not_metered_and_unwedges_the_batch():
    """A result that lands after the cancel sweep must not be billed.

    The store keeps the item CANCELLED and does not count it, so the meter must
    not see it either — but finalize still has to run, or a batch cancelled
    while its last item was in the engine would sit in CANCELLING forever.
    """
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    batch = repo.add_batch(1)
    client.gate = asyncio.Event()  # the submission hangs inside the engine
    metered: list[ItemRecord] = []
    finalizer = FakeFinalizer()
    disp = make_dispatcher(repo, client, on_success=metered.append, finalize=finalizer)

    await disp.tick(T0 + timedelta(seconds=1))
    await asyncio.sleep(0)
    assert repo.item_states(batch.id) == [ItemState.INFLIGHT]

    repo.cancel_batch(batch.id, now=T0 + timedelta(seconds=2))  # user cancels
    client.gate.set()
    await disp.drain()

    assert repo.item_states(batch.id) == [ItemState.CANCELLED]
    assert metered == [], "a cancelled item is never billed"
    assert repo.batches[batch.id].counts_completed == 0
    assert finalizer.finalized == [batch.id]
    assert repo.batches[batch.id].status is BatchStatus.CANCELLED


async def test_retryable_upstream_requeues_and_fatal_upstream_fails_the_item():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    batch = repo.add_batch(1)
    finalizer = FakeFinalizer()
    disp = make_dispatcher(repo, client, finalize=finalizer)

    client.chat_error = RetryableUpstream(503, "engine busy")
    await run_tick(disp, T0 + timedelta(seconds=1))
    item = repo.items[batch.id][0]
    assert item.state is ItemState.PENDING and item.attempts == 1
    assert finalizer.calls == [], "a retried item has not reached a terminal state"

    client.chat_error = FatalUpstream(400, "bad request")
    await run_tick(disp, T0 + timedelta(seconds=2))
    assert item.state is ItemState.FAILED and item.attempts == 2
    assert "400" in (item.last_error or "")
    assert finalizer.calls == [batch.id]


# --------------------------------------------------------------------------
# circuit breaker + expiry
# --------------------------------------------------------------------------


async def test_engine_down_opens_circuit_and_requeues_inflight():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    batch = repo.add_batch(4)
    client.gate = asyncio.Event()  # submissions hang
    disp = make_dispatcher(repo, client)

    report = await disp.tick(T0 + timedelta(seconds=1))
    await asyncio.sleep(0)
    assert report.submitted == 1 and report.inflight == 1
    assert repo.item_states(batch.id)[0] is ItemState.INFLIGHT

    client.scrape_error = EngineDown("connection refused")
    down = await disp.tick(T0 + timedelta(seconds=2))

    assert down.circuit_open is True
    assert down.target == 0 and down.submitted == 0 and down.inflight == 0
    assert disp.aimd.target == 0
    assert repo.requeue_calls == 1
    assert repo.item_states(batch.id) == [ItemState.PENDING] * 4
    assert len(client.calls) == 1, "nothing is submitted while the circuit is open"

    client.gate.set()
    await disp.drain()
    assert repo.item_states(batch.id) == [ItemState.PENDING] * 4


async def test_expiry_sweep_expires_open_items_and_finalizes_partial_output():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    batch = repo.add_batch(3, created_at=T0, window_hours=1)
    finalizer = FakeFinalizer()
    disp = make_dispatcher(repo, client, finalize=finalizer)

    await run_tick(disp, T0 + timedelta(seconds=1))  # one item completes
    assert repo.batch_progress(batch.id) == (3, 1, 0)

    report = await run_tick(disp, T0 + timedelta(hours=2))  # past expires_at

    assert report.expired_batches == [batch.id]
    assert repo.batches[batch.id].status is BatchStatus.EXPIRED
    assert repo.item_states(batch.id) == [
        ItemState.SUCCEEDED,
        ItemState.EXPIRED,
        ItemState.EXPIRED,
    ]
    assert len(client.calls) == 1, "no work is submitted for a batch past its window"
    assert finalizer.finalized == [batch.id]
    assert repo.batch_progress(batch.id) == (3, 1, 0)  # partial output survives


# --------------------------------------------------------------------------
# vLLM client
# --------------------------------------------------------------------------

METRICS_TEXT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 3.0
vllm:num_requests_running{model_name="other"} 1.0
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 2.0
vllm:kv_cache_usage_perc{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.42
vllm:kv_cache_usage_perc{model_name="other"} 0.11
vllm:num_requests_running_total 999.0
vllm:num_requests_swapped 7.0
"""


def make_client(handler, cfg: TidalConfig | None = None) -> VllmClient:
    cfg = cfg or TidalConfig(
        vllm_base_url="http://engine:8000", vllm_metrics_url="http://engine:8000/metrics"
    )
    return VllmClient(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_parse_metrics_is_label_tolerant_and_ignores_similar_names():
    m = parse_metrics(METRICS_TEXT)
    assert m == EngineMetrics(running=4, waiting=2, kv_usage=0.42)
    assert parse_metrics("vllm:num_requests_waiting 5.0\n").waiting == 5
    assert parse_metrics("nothing here\n") == EngineMetrics(running=0, waiting=0, kv_usage=0.0)


async def test_scrape_reads_engine_metrics():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text=METRICS_TEXT)

    client = make_client(handler)
    assert await client.scrape() == EngineMetrics(running=4, waiting=2, kv_usage=0.42)
    assert seen == ["http://engine:8000/metrics"]
    await client.aclose()


async def test_scrape_raises_engine_down_on_transport_error_and_bad_status():
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(refused)
    with pytest.raises(EngineDown):
        await client.scrape()
    await client.aclose()

    client = make_client(lambda request: httpx.Response(503, text="unavailable"))
    with pytest.raises(EngineDown):
        await client.scrape()
    await client.aclose()


async def test_chat_sends_priority_non_streaming_and_returns_usage():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-7",
                "choices": [{"index": 0, "message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 34},
            },
        )

    client = make_client(handler)
    result, ptoks, ctoks = await client.chat({"model": "m", "messages": [], "stream": True}, 42)

    assert captured["url"] == "http://engine:8000/v1/chat/completions"
    assert captured["body"]["priority"] == 42
    assert captured["body"]["stream"] is False
    assert result["id"] == "cmpl-7"
    assert (ptoks, ctoks) == (12, 34)
    await client.aclose()


async def test_chat_maps_status_codes_to_retryable_and_fatal():
    client = make_client(lambda request: httpx.Response(503, text="overloaded"))
    with pytest.raises(RetryableUpstream) as retryable:
        await client.chat({"model": "m"}, 1)
    assert retryable.value.status == 503
    await client.aclose()

    client = make_client(lambda request: httpx.Response(400, text="bad model"))
    with pytest.raises(FatalUpstream) as fatal:
        await client.chat({"model": "m"}, 1)
    assert fatal.value.status == 400
    await client.aclose()

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = make_client(timeout)
    with pytest.raises(RetryableUpstream):
        await client.chat({"model": "m"}, 1)
    await client.aclose()


async def test_chat_uses_the_batch_endpoint():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "cmpl-1", "usage": {}})

    client = make_client(handler)
    _result, ptoks, ctoks = await client.chat(
        {"model": "m", "prompt": "hi"}, 3, endpoint="/v1/completions"
    )
    assert captured["url"] == "http://engine:8000/v1/completions"
    assert (ptoks, ctoks) == (0, 0)
    await client.aclose()
