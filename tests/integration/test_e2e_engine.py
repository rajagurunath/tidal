"""Task B4 — the TidalScheduler inside a live vLLM, end to end.

A8 proves the *gateway* works against a stock engine. This proves the other
technique: vLLM launched with
``--scheduler-cls tidal.engine.scheduler.TidalScheduler`` serves online and
batch traffic together, with no gateway in the path at all, and the scheduler
is demonstrably the thing doing the co-scheduling.

"Demonstrably" matters — a subclass that silently fell back to stock behaviour
would pass every functional assertion here. So the test reads the engine's own
log (the fixture captures the server's stdout) and asserts on the two lines the
scheduler emits:

* the one-shot **capability probe** at construction, which also proves the
  ``--scheduler-cls`` plugin actually loaded and found the upstream attributes
  it depends on;
* the periodic **telemetry line** (every 100 steps), which must show both
  regimes: a real token cap in force while online traffic is resident, and
  offline-mode fill (no cap, zero online tokens, batch tokens > 0) once online
  goes quiet.

**Batch is submitted directly to the engine**, not through the gateway: that is
technique B's production shape (the engine needs no external admission
control), and it is the same code path as the harness's ``technique_b``
condition, so this test smoke-tests that too. A rolling submission window is
used rather than firing all items at once, because a continuously non-empty
waiting queue is what the pre-gate actually gates — dumping everything into
``running`` in the first step would leave nothing for it to decide about.

Phases (~2.5 min after the engine is up):

    baseline:     online-only Poisson load
    co-serving:   online load + a rolling batch stream  -> capped-mode lines
    offline tail: batch only, online quiet              -> offline-mode lines
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import pytest

from tidal.eval.harness import (
    make_batch_items,
    parse_tidal_stats,
    submit_batch_direct,
    tidal_probe_line,
)
from tidal.eval.loadgen import OnlineLoadGen, poisson_schedule, summarize_latencies

pytestmark = pytest.mark.integration

MODEL = os.environ.get("TIDAL_EVAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

BASELINE_S = float(os.environ.get("TIDAL_IT_B_BASELINE_S", "30"))
WINDOW_S = float(os.environ.get("TIDAL_IT_B_WINDOW_S", "40"))
ONLINE_RPS = float(os.environ.get("TIDAL_IT_B_RPS", "1.0"))
#: Sized so the batch stream outlives the online window on this CPU box
#: (~3 items/s at 8-way concurrency with 32 output tokens each).
BATCH_ITEMS = int(os.environ.get("TIDAL_IT_B_BATCH_ITEMS", "140"))
BATCH_MAX_TOKENS = int(os.environ.get("TIDAL_IT_B_BATCH_MAX_TOKENS", "32"))
BATCH_CONCURRENCY = int(os.environ.get("TIDAL_IT_B_BATCH_CONCURRENCY", "8"))
#: The offline tail must span well over 100 engine steps, or the every-100-step
#: telemetry line might never be emitted inside it. 32 items x 64 tokens at
#: 8-way concurrency is ~260 steps.
TAIL_ITEMS = int(os.environ.get("TIDAL_IT_B_TAIL_ITEMS", "32"))
TAIL_MAX_TOKENS = int(os.environ.get("TIDAL_IT_B_TAIL_MAX_TOKENS", "64"))
DRAIN_TIMEOUT_S = float(os.environ.get("TIDAL_IT_B_DRAIN_S", "300"))
#: Catastrophic-regression guard, deliberately loose — and the loose number is
#: itself a finding. The pre-gate caps *admission*; a batch request that is
#: already resident keeps decoding until the guardband evicts it (documented
#: deviation 2 in ``tidal.engine.scheduler``). On this box KV usage peaks around
#: 0.3%, so the guardband never fires, ~6 of the 8 outstanding batch requests
#: stay resident, and online steps carry them: measured 4.2x online p99 at a
#: 60 ms TBT SLO (4.7x at the shipped 200 ms default). Certifying an online SLO
#: is the eval harness's job on a GPU, where KV pressure — and therefore
#: eviction — is real. Here we only assert nothing fell off a cliff.
LATENCY_BUDGET = float(os.environ.get("TIDAL_IT_B_LATENCY_BUDGET", "6.0"))

#: The engine knobs that make this a *test* rather than a demonstration of a
#: badly-chosen default. ``tbt_slo_ms=200`` ships for GPUs; an online-only step
#: on this CPU box is ~25 ms, so a 200 ms budget lets T̂ hand batch work an
#: 8x inflation of step time and still believe it is inside the SLO — measured
#: at 4.7x online p99 before this was set. 60 ms is the same knob pointed at
#: this machine, and the cold-start fraction is pulled down with it so the
#: window before T̂ converges is not a free-for-all either.
SCHEDULER_ENV = {
    "TIDAL_TBT_SLO_MS": os.environ.get("TIDAL_IT_B_TBT_SLO_MS", "60"),
    "TIDAL_COLD_START_BATCH_FRAC": os.environ.get("TIDAL_IT_B_COLD_START_FRAC", "0.02"),
}


async def test_tidal_scheduler_coserves_and_reports_both_regimes(engine):
    server = engine("tidal", **SCHEDULER_ENV)
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=900.0) as http:
        # -- phase 1: online-only baseline --------------------------------
        baseline_gen = OnlineLoadGen(base_url=server.base_url, model=MODEL, max_tokens=16)
        await baseline_gen.run(poisson_schedule(ONLINE_RPS, BASELINE_S, seed=303), client=http)
        baseline = summarize_latencies(baseline_gen.latencies())
        assert baseline["count"] > 0.5 * ONLINE_RPS * BASELINE_S, (
            f"baseline load produced too few samples: {baseline}"
        )
        assert not [r for r in baseline_gen.records if not r.ok], "online errors in the baseline"

        # -- phase 2: co-serving ------------------------------------------
        items = make_batch_items(BATCH_ITEMS, model=MODEL, max_tokens=BATCH_MAX_TOKENS)
        batch_task = asyncio.create_task(
            submit_batch_direct(
                server.base_url,
                items,
                t0=t0,
                concurrency=BATCH_CONCURRENCY,
                client=http,
            ),
            name="batch",
        )
        loaded_gen = OnlineLoadGen(base_url=server.base_url, model=MODEL, max_tokens=16)
        await loaded_gen.run(poisson_schedule(ONLINE_RPS, WINDOW_S, seed=404), client=http)
        window_end = time.perf_counter() - t0
        completions = await asyncio.wait_for(batch_task, timeout=DRAIN_TIMEOUT_S)
        loaded = summarize_latencies(loaded_gen.latencies())

        # -- phase 3: offline tail, no online traffic at all ---------------
        tail_items = make_batch_items(TAIL_ITEMS, model=MODEL, max_tokens=TAIL_MAX_TOKENS)
        tail = await asyncio.wait_for(
            submit_batch_direct(
                server.base_url,
                tail_items,
                t0=t0,
                concurrency=BATCH_CONCURRENCY,
                client=http,
            ),
            timeout=DRAIN_TIMEOUT_S,
        )

    # -- the batch work itself ------------------------------------------
    failed = [c for c in completions + tail if not c.ok]
    assert not failed, f"{len(failed)} batch items failed, first: {failed[0].error}"
    assert len(completions) == BATCH_ITEMS
    assert {c.custom_id for c in completions} == {f"item-{i}" for i in range(BATCH_ITEMS)}
    assert all(c.completion_tokens > 0 for c in completions)
    covered = sum(1 for c in completions if c.finished_at <= window_end)

    # -- the scheduler was demonstrably the one doing it -----------------
    log_text = server.log_text()
    probe = tidal_probe_line(log_text)
    stats = parse_tidal_stats(log_text)

    print(
        f"\n[b4] probe: {probe}"
        f"\n[b4] baseline p50={baseline['p50']:.3f}s p99={baseline['p99']:.3f}s "
        f"n={baseline['count']}"
        f"\n[b4] loaded   p50={loaded['p50']:.3f}s p99={loaded['p99']:.3f}s "
        f"n={loaded['count']} ratio={loaded['p99'] / baseline['p99']:.2f}x"
        f"\n[b4] batch    {len(completions)} items, {covered} finished inside the online window; "
        f"tail {len(tail)} items"
        f"\n[b4] telemetry lines={len(stats)} "
        f"capped={sum(1 for s in stats if s.get('x') is not None)} "
        f"offline={sum(1 for s in stats if s.get('x') is None)} "
        f"held_max={max((s.get('held', 0) for s in stats), default=0)} "
        f"evicted={sum(s.get('evicted', 0) for s in stats)} "
        f"model_ready={any(s.get('model_ready') for s in stats)}"
    )

    assert probe is not None, (
        "no TidalScheduler capability-probe line in the engine log — the "
        f"--scheduler-cls plugin did not load. Log tail:\n{server.log_tail()}"
    )
    assert "kv_cache_manager.usage=ok" in probe, probe
    assert "_preempt_request=ok" in probe, probe

    assert len(stats) >= 2, (
        f"expected periodic TidalScheduler telemetry, got {len(stats)} lines. "
        f"Log tail:\n{server.log_tail()}"
    )

    capped = [s for s in stats if s.get("x") is not None]
    assert capped, (
        "no telemetry line shows a batch token cap in force: the scheduler "
        "never entered online-protection mode while online traffic was resident"
    )
    assert all(s["x"] >= 0 for s in capped)

    offline = [
        s
        for s in stats
        if s.get("x") is None and s.get("online_tokens") == 0 and s.get("batch_tokens", 0) > 0
    ]
    assert offline, (
        "no telemetry line shows offline-mode fill (no cap, zero online tokens, "
        "batch tokens > 0) — the work-conserving path never engaged"
    )
    assert max(s["batch_tokens"] for s in offline) > 0

    # -- online traffic survived ----------------------------------------
    assert not [r for r in loaded_gen.records if not r.ok], "online errors under batch load"
    assert loaded["p99"] < LATENCY_BUDGET * baseline["p99"], (
        f"online p99 {loaded['p99']:.3f}s exceeded {LATENCY_BUDGET}x the in-session "
        f"baseline {baseline['p99']:.3f}s"
    )
