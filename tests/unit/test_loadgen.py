"""Load generator: the schedule must be a *seeded, reproducible* input.

Every condition in the A/B matrix is compared against every other one, so the
arrival process has to be identical across runs — otherwise a difference in
p99 could just be a difference in the load. These tests pin that property, the
statistics of both arrival shapes, and the executor's bookkeeping.
"""

from __future__ import annotations

import math
from itertools import pairwise

import httpx
import pytest

from tidal.eval.loadgen import (
    OnlineLoadGen,
    diurnal_schedule,
    percentile,
    poisson_schedule,
    schedule,
    summarize_latencies,
)

# -- schedules ---------------------------------------------------------------


def test_poisson_schedule_is_deterministic_in_the_seed():
    a = poisson_schedule(2.0, 30.0, seed=42)
    b = poisson_schedule(2.0, 30.0, seed=42)
    c = poisson_schedule(2.0, 30.0, seed=43)
    assert a == b
    assert a != c


def test_poisson_schedule_is_sorted_and_inside_the_window():
    offsets = poisson_schedule(5.0, 20.0, seed=1)
    assert offsets == sorted(offsets)
    assert all(0.0 <= t < 20.0 for t in offsets)


def test_poisson_rate_matches_the_requested_rps():
    offsets = poisson_schedule(4.0, 2000.0, seed=3)
    assert 4.0 * 2000.0 * 0.9 < len(offsets) < 4.0 * 2000.0 * 1.1


def test_poisson_gaps_are_exponential():
    """Mean and standard deviation of an exponential both equal 1/rate."""
    offsets = poisson_schedule(10.0, 500.0, seed=5)
    gaps = [b - a for a, b in pairwise(offsets)]
    mean = sum(gaps) / len(gaps)
    sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
    assert 0.09 < mean < 0.11
    assert 0.9 < sd / mean < 1.1


def test_poisson_rejects_nonpositive_rate_and_returns_empty_for_no_window():
    with pytest.raises(ValueError):
        poisson_schedule(0.0, 10.0)
    assert poisson_schedule(1.0, 0.0) == []


def test_diurnal_schedule_is_deterministic_and_bounded_by_the_peak():
    a = diurnal_schedule(0.2, 2.0, 600.0, 300.0, seed=9)
    assert a == diurnal_schedule(0.2, 2.0, 600.0, 300.0, seed=9)
    assert a == sorted(a)
    # Thinning can only remove arrivals, so the count stays under the peak rate.
    assert len(a) < 2.0 * 600.0


def test_diurnal_peak_window_is_busier_than_the_trough_window():
    period = 400.0
    offsets = diurnal_schedule(0.1, 4.0, 4000.0, period, seed=11)
    # sin() peaks a quarter period in and troughs three quarters in.
    peak = sum(1 for t in offsets if 0.15 < (t % period) / period < 0.35)
    trough = sum(1 for t in offsets if 0.65 < (t % period) / period < 0.85)
    assert peak > 3 * trough


def test_diurnal_rejects_a_peak_below_the_base():
    with pytest.raises(ValueError):
        diurnal_schedule(2.0, 1.0, 100.0, 60.0)


def test_schedule_dispatches_by_name_and_rejects_unknown_shapes():
    assert schedule("poisson", 1.0, 50.0, 4) == poisson_schedule(1.0, 50.0, 4)
    assert schedule("diurnal", 2.0, 200.0, 4, period_s=100.0, trough_frac=0.25) == diurnal_schedule(
        0.5, 2.0, 200.0, 100.0, 4
    )
    with pytest.raises(ValueError):
        schedule("flat", 1.0, 10.0)


# -- percentiles -------------------------------------------------------------


def test_percentile_interpolates_like_numpy():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.25) == pytest.approx(1.75)


def test_percentile_handles_empty_and_single_values():
    assert math.isnan(percentile([], 0.5))
    assert percentile([7.0], 0.99) == 7.0


def test_percentile_rejects_out_of_range_quantiles():
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def test_summarize_latencies_reports_the_paper_percentiles():
    stats = summarize_latencies([float(i) for i in range(1, 101)])
    assert stats["count"] == 100
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["p50"] == pytest.approx(50.5)
    assert stats["p99"] == pytest.approx(99.01)
    assert stats["max"] == 100.0


def test_summarize_latencies_of_nothing_is_nan_not_a_crash():
    stats = summarize_latencies([])
    assert stats["count"] == 0
    assert math.isnan(stats["p99"])


# -- the executor ------------------------------------------------------------


def test_body_is_deterministic_in_the_index_and_marks_online_priority():
    gen = OnlineLoadGen(base_url="http://x", model="m", max_tokens=8)
    body = gen.body_for(3)
    assert body == gen.body_for(3)
    assert body["priority"] == 0
    assert body["stream"] is False
    assert body["max_tokens"] == 8
    assert gen.body_for(0)["messages"] != gen.body_for(1)["messages"]


async def test_run_records_one_result_per_arrival_with_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    gen = OnlineLoadGen(base_url="http://engine", model="m")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        records = await gen.run([0.0, 0.001, 0.002], client=client)

    assert [r.index for r in records] == [0, 1, 2]
    assert all(r.ok and r.status == 200 for r in records)
    assert all(r.completion_tokens == 3 and r.prompt_tokens == 5 for r in records)
    assert all(r.latency_s >= 0 for r in records)
    assert gen.summary()["count"] == 3
    assert gen.summary()["errors"] == 0


async def test_failures_are_recorded_not_raised_and_excluded_from_latencies():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    gen = OnlineLoadGen(base_url="http://engine", model="m")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        records = await gen.run([0.0, 0.001], client=client)

    assert len(records) == 2
    assert all(not r.ok for r in records)
    assert gen.latencies() == []
    assert gen.summary()["errors"] == 2


async def test_arrivals_are_open_loop_a_slow_response_does_not_delay_the_next():
    """The generator must not close the loop: request k+1 is issued on
    schedule even while k is still outstanding, or a slow engine would
    silently reduce the offered load and hide the queueing it caused."""
    started: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        import asyncio

        started.append(1)
        if len(started) == 1:
            await asyncio.sleep(0.25)
        return httpx.Response(200, json={"usage": {}})

    gen = OnlineLoadGen(base_url="http://engine", model="m")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        records = await gen.run([0.0, 0.01, 0.02], client=client)

    assert len(records) == 3
    # The two fast requests finished long before the slow first one.
    assert records[1].finished_at < records[0].finished_at
