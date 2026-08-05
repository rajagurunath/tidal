"""TidalScheduler: the work-conserving pre-gate around vLLM's stock schedule().

Runs in the **engine** venv (vLLM installed), not tidal's own::

    env PYTHONPATH=<tidal>/src TORCHDYNAMO_DISABLE=1 \\
        <vllm-experiments>/.venv/bin/python -m pytest tests/unit/test_tidal_scheduler.py

Schedulers are built by :mod:`tests.unit.engine_fixtures`, which mirrors vLLM's
own ``create_scheduler_with_priority`` harness — real ``Scheduler`` machinery,
real KV block accounting, no model weights.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm", reason="engine tests require vLLM in the environment")

from tests.unit.engine_fixtures import create_requests, create_scheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from tidal.engine.latency_model import NotReadyError
from tidal.engine.scheduler import TidalScheduler

ONLINE = 0
BATCH = 100


class _ModelStub:
    """Stand-in for ``StepLatencyModel`` with a caller-chosen cap."""

    def __init__(self, cap: int | None) -> None:
        self.cap = cap  # None -> behave as an unconverged fit
        self.observations: list[tuple[int, int, float]] = []

    def max_batch_tokens(self, p_on: int, c_total: int, budget_ms: float) -> int:
        if self.cap is None:
            raise NotReadyError("stub: not converged")
        return self.cap

    def observe(self, p: int, c: int, step_ms: float) -> None:
        self.observations.append((p, c, step_ms))

    def ready(self) -> bool:
        return self.cap is not None

    def mape(self) -> float:
        return float("nan")


class _GuardbandStub:
    """Stand-in for ``KvGuardband`` with pinned admit/evict verdicts."""

    def __init__(self, admit: bool = True, evict: bool = False) -> None:
        self.admit = admit
        self.evict = evict

    def admit_ok(self, kv_usage: float) -> bool:
        return self.admit

    def evict_needed(self, kv_usage: float) -> bool:
        return self.evict

    def h(self) -> float:
        return 0.3

    def observe_online_demand(self, blocks_allocated: int, pool_blocks: int) -> None:
        pass


def _batch_ids(scheduler) -> set[str]:
    return {rid for rid, r in scheduler.requests.items() if TidalScheduler.classify(r)}


def _waiting_ids(scheduler) -> set[str]:
    return {r.request_id for r in scheduler.waiting} | {
        r.request_id for r in scheduler.skipped_waiting
    }


# -- classification & construction ----------------------------------------


def test_classify_splits_at_priority_one():
    """Online is priority 0 and nothing else; every batch band is >= 1.

    The gateway emits 100..1 for batch work, so an *escalated* batch item sits
    at priority 1..9 — it must still be classified as batch (capped,
    guardbanded, counted in batch telemetry) rather than smuggled into the
    online partition by its own urgency.
    """
    online, escalated, mid, boundary, batch = create_requests(5, priorities=[0, 1, 9, 10, 100])
    assert not TidalScheduler.classify(online)
    assert TidalScheduler.classify(escalated)
    assert TidalScheduler.classify(mid)
    assert TidalScheduler.classify(boundary)
    assert TidalScheduler.classify(batch)


def test_escalated_batch_request_is_still_capped_as_batch():
    """A priority-1 item is subject to the batch token cap, not exempt from it."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    scheduler.latency_model = _ModelStub(cap=0)

    for request in create_requests(3, priorities=[ONLINE, 1, 1], num_tokens=64):
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"0": 64}  # the online request only
    assert scheduler.tidal_stats.held_count == 2
    assert scheduler.tidal_stats.batch_tokens == 0


def test_construction_requires_the_priority_policy():
    with pytest.raises(ValueError, match="priority scheduling policy"):
        create_scheduler(TidalScheduler, policy="fcfs")


# -- the six plan tests ----------------------------------------------------


def test_online_only_schedules_identically_to_stock():
    """With no batch traffic, TidalScheduler is a pass-through: same token map."""
    kwargs = dict(max_num_batched_tokens=128, max_num_seqs=8, max_model_len=1024)
    stock = create_scheduler(Scheduler, **kwargs)
    tidal = create_scheduler(TidalScheduler, **kwargs)

    for scheduler in (stock, tidal):
        for request in create_requests(6, priorities=[ONLINE] * 6, num_tokens=100):
            scheduler.add_request(request)

    # Three steps without model output: enough to cover admission, chunked
    # prefill continuation and the transition to decode.
    for _ in range(3):
        stock_output = stock.schedule()
        tidal_output = tidal.schedule()
        assert tidal_output.num_scheduled_tokens == stock_output.num_scheduled_tokens
        assert tidal_output.total_num_scheduled_tokens == stock_output.total_num_scheduled_tokens
    assert tidal.tidal_stats.batch_tokens == 0
    assert tidal.tidal_stats.held_count == 0


def test_batch_held_when_model_caps():
    """An over-budget T̂ hides every waiting batch request — and loses none."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    scheduler.latency_model = _ModelStub(cap=0)

    requests = create_requests(5, priorities=[ONLINE] + [BATCH] * 4, num_tokens=64)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    batch_ids = _batch_ids(scheduler)
    assert batch_ids.isdisjoint(output.num_scheduled_tokens)
    assert output.num_scheduled_tokens == {"0": 64}  # the online request only
    assert scheduler.tidal_stats.capped_x == 0
    assert scheduler.tidal_stats.held_count == 4
    assert scheduler.tidal_stats.batch_tokens == 0
    # Nothing is lost: every held request is back in the waiting queue.
    assert scheduler._tidal_held == []
    assert batch_ids <= _waiting_ids(scheduler)


def test_batch_admitted_up_to_the_cap():
    """The cap is enforced at chunk granularity: 2 x 64 fits under X=128."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=512)
    scheduler.latency_model = _ModelStub(cap=128)

    for request in create_requests(5, priorities=[ONLINE] + [BATCH] * 4, num_tokens=64):
        scheduler.add_request(request)

    output = scheduler.schedule()

    scheduled_batch = _batch_ids(scheduler) & set(output.num_scheduled_tokens)
    assert len(scheduled_batch) == 2
    assert scheduler.tidal_stats.batch_tokens == 128
    assert scheduler.tidal_stats.held_count == 2
    assert scheduler.tidal_stats.slack_unfilled == 0


def test_offline_mode_fills_budget_with_batch():
    """Zero online resident -> no cap at all; the whole budget is batch."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    for request in create_requests(8, priorities=[BATCH] * 8, num_tokens=64):
        scheduler.add_request(request)

    output = scheduler.schedule()

    stats = scheduler.tidal_stats
    assert stats.capped_x is None  # offline mode
    assert output.total_num_scheduled_tokens == 256
    assert stats.batch_tokens == 256
    assert stats.online_tokens == 0
    assert stats.slack_unfilled == 0


def test_guardband_blocks_batch_admission_at_high_kv():
    """No KV headroom -> batch admission stops even with budget to spare."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    scheduler.guardband = _GuardbandStub(admit=False, evict=False)

    for request in create_requests(4, priorities=[BATCH] * 4, num_tokens=32):
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert scheduler.tidal_stats.held_count == 4
    assert scheduler._tidal_held == []
    assert _batch_ids(scheduler) <= _waiting_ids(scheduler)


def test_proactive_eviction_picks_cheapest_batch_victim():
    """Above the evict watermark we drop the victim with the least lost work."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    cheap = create_requests(1, priorities=[BATCH], num_tokens=32)[0]
    expensive = create_requests(1, priorities=[BATCH], num_tokens=160, starting_idx=9)[0]
    for request in (cheap, expensive):
        scheduler.add_request(request)

    scheduler.schedule()  # offline mode admits both
    running_ids = {r.request_id for r in scheduler.running}
    assert running_ids == {cheap.request_id, expensive.request_id}
    assert cheap.num_computed_tokens < expensive.num_computed_tokens

    scheduler.guardband = _GuardbandStub(admit=False, evict=True)
    scheduler.schedule()

    assert scheduler.tidal_stats.evicted == 1
    assert cheap.status == RequestStatus.PREEMPTED
    assert cheap not in scheduler.running
    assert expensive in scheduler.running
    # The victim is parked in the waiting queue, not dropped.
    assert cheap.request_id in _waiting_ids(scheduler)


def test_failed_eviction_puts_the_victim_back_in_running(monkeypatch, caplog):
    """If ``_preempt_request`` raises, the victim must not fall out of the world.

    It has already been popped from ``running`` at that point; leaving it there
    would strand a request that still holds KV blocks and is still in
    ``self.requests`` — it would never be scheduled or freed again.
    """
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    cheap = create_requests(1, priorities=[BATCH], num_tokens=32)[0]
    expensive = create_requests(1, priorities=[BATCH], num_tokens=160, starting_idx=9)[0]
    for request in (cheap, expensive):
        scheduler.add_request(request)
    scheduler.schedule()  # offline mode admits both

    def boom(*args, **kwargs):
        raise RuntimeError("preemption exploded")

    monkeypatch.setattr(scheduler, "_preempt_request", boom)
    scheduler.guardband = _GuardbandStub(admit=False, evict=True)

    with caplog.at_level("ERROR", logger="tidal.engine.scheduler"):
        scheduler.schedule()

    assert scheduler.tidal_stats.evicted == 0
    assert cheap in scheduler.running
    assert expensive in scheduler.running
    assert cheap.status == RequestStatus.RUNNING
    assert "eviction" in caplog.text
    # ...and the next step still runs, with the victim intact and schedulable.
    scheduler.guardband = _GuardbandStub(admit=True, evict=False)
    scheduler.schedule()
    assert cheap in scheduler.running
    assert cheap.request_id in scheduler.requests


def test_exception_in_super_restores_held_requests(monkeypatch):
    """A blow-up inside stock schedule() must not swallow hidden requests."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    scheduler.latency_model = _ModelStub(cap=0)
    for request in create_requests(5, priorities=[ONLINE] + [BATCH] * 4, num_tokens=64):
        scheduler.add_request(request)

    def boom(self, *args, **kwargs):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(Scheduler, "schedule", boom)

    with pytest.raises(RuntimeError, match="upstream exploded"):
        scheduler.schedule()

    assert scheduler._tidal_held == []
    all_ids = set(scheduler.requests)
    assert all_ids == _waiting_ids(scheduler)


# -- calibration & robustness ---------------------------------------------


def test_step_latency_is_attributed_to_the_previous_step():
    """(P, C) of step N is observed with the wall time measured at step N+1."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    model = _ModelStub(cap=256)
    scheduler.latency_model = model
    for request in create_requests(2, priorities=[ONLINE, ONLINE], num_tokens=64):
        scheduler.add_request(request)

    first = scheduler.schedule()
    assert model.observations == []  # nothing to attribute yet

    scheduler.schedule()
    assert len(model.observations) == 1
    p, c, step_ms = model.observations[0]
    assert p == first.total_num_scheduled_tokens
    assert c == 0  # fresh prefills ran against no resident context
    assert step_ms > 0.0


def test_pre_gate_failure_falls_back_to_stock_scheduling(monkeypatch, caplog):
    """Any error in our hook degrades to stock behaviour, never to an outage."""
    scheduler = create_scheduler(TidalScheduler, max_num_batched_tokens=256)
    for request in create_requests(4, priorities=[BATCH] * 4, num_tokens=32):
        scheduler.add_request(request)

    def boom(self, *args, **kwargs):
        raise RuntimeError("pre-gate exploded")

    monkeypatch.setattr(TidalScheduler, "_pre_gate", boom)

    with caplog.at_level("ERROR", logger="tidal.engine.scheduler"):
        output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 128  # all four, stock behaviour
    assert scheduler._tidal_held == []
    assert "pre-gate failed" in caplog.text


def test_capability_probe_finds_the_upstream_surface():
    scheduler = create_scheduler(TidalScheduler)
    caps = scheduler._caps
    assert caps.kv_usage and caps.preempt and caps.estimate_cached_tokens
    assert caps.pool_blocks > 0
