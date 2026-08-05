"""Task A8 — the gateway against a live vLLM, end to end.

What this proves, on real hardware and with the real ``openai`` SDK:

1. **The wire contract holds.** A batch uploaded as JSONL comes back as a
   ``completed`` batch with an output file whose every line parses, whose
   ``custom_id`` set is exactly the input's, and whose ``request_counts``
   agree with the file.
2. **Co-serving is not free but it is bounded.** Online p99 latency *while the
   batch is being injected* stays under 2.5x the online-only p99 measured in
   the same session, on the same engine, minutes apart. Measuring the baseline
   in-session is the whole point: a hard-coded latency budget would be a test
   of this laptop's thermal state, not of the dispatcher.
3. **The comparison is honest.** The loaded window is only a fair measurement
   if batch work was actually in flight throughout it, so the test asserts
   coverage from the dispatcher's own ``TickReport``s and fails loudly if the
   batch pool drained early.

Shape of the run (~3 min after the engine is up):

    baseline: online-only Poisson load           -> p99_baseline
    loaded:   same Poisson load + 200-item batch -> p99_loaded, tick reports
    drain:    wait for the batch to finalize     -> output file assertions

**The gateway runs in-process** (ASGI transport + an in-loop dispatcher task)
rather than as a ``tidal serve`` subprocess. The HTTP surface is identical —
the SDK is talking to the same FastAPI app through the same JSON — and in
exchange the test can assert on the controller's ``TickReport``s, which is
where the interesting behaviour actually lives. Online traffic still goes
straight to vLLM over real HTTP, exactly as in production.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from datetime import UTC, datetime

import httpx
import pytest
from openai import AsyncOpenAI

from tidal.api.app import create_app
from tidal.api.assembler import finalize_if_done
from tidal.config import TidalConfig
from tidal.dispatcher.loop import Dispatcher
from tidal.dispatcher.vllm_client import VllmClient
from tidal.eval.loadgen import OnlineLoadGen, poisson_schedule, summarize_latencies
from tidal.store.interfaces import BatchStatus, make_repository

pytestmark = pytest.mark.integration

MODEL = os.environ.get("TIDAL_EVAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GATEWAY_URL = "http://gateway"

#: Sized for a CPU box: long enough for a stable p99, short enough to run
#: tonight. Every knob is env-overridable so the same test scales to a GPU.
BASELINE_S = float(os.environ.get("TIDAL_IT_BASELINE_S", "45"))
WINDOW_S = float(os.environ.get("TIDAL_IT_WINDOW_S", "60"))
ONLINE_RPS = float(os.environ.get("TIDAL_IT_RPS", "1.0"))
#: 200 items x ~16 output tokens is ~60 s of batch work at 4-way concurrency on
#: this box — chosen so the batch outlives the loaded window rather than
#: draining in the first five seconds and leaving the p99 comparison vacuous.
BATCH_ITEMS = int(os.environ.get("TIDAL_IT_BATCH_ITEMS", "200"))
BATCH_MAX_TOKENS = int(os.environ.get("TIDAL_IT_BATCH_MAX_TOKENS", "16"))
DRAIN_TIMEOUT_S = float(os.environ.get("TIDAL_IT_DRAIN_S", "240"))
#: p99 under load, as a multiple of the in-session online-only p99.
LATENCY_BUDGET = float(os.environ.get("TIDAL_IT_LATENCY_BUDGET", "2.5"))
#: Fraction of the loaded window that must have had batch work in flight.
MIN_COVERAGE = 0.8

TERMINAL = {
    BatchStatus.COMPLETED,
    BatchStatus.FAILED,
    BatchStatus.EXPIRED,
    BatchStatus.CANCELLED,
}


def batch_input(count: int) -> tuple[bytes, list[str]]:
    """A JSONL batch file of ``count`` short chat requests + its custom_ids."""
    prompts = [
        "Name one fact about the ocean.",
        "Give a one-line definition of latency.",
        "What is a GPU used for? One sentence.",
        "Say hello in French.",
    ]
    lines, ids = [], []
    for i in range(count):
        custom_id = f"req-{i:04d}"
        ids.append(custom_id)
        lines.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"You are a helpful assistant. "
                                f"{prompts[i % len(prompts)]} (#{i})",
                            }
                        ],
                        "max_tokens": BATCH_MAX_TOKENS,
                        "temperature": 0.0,
                    },
                }
            )
        )
    return ("\n".join(lines) + "\n").encode(), ids


async def test_gateway_coserves_a_batch_without_wrecking_online_latency(engine, tmp_path):
    server = engine("stock")

    cfg = TidalConfig(
        vllm_base_url=server.base_url,
        vllm_metrics_url=f"{server.base_url}/metrics",
        dsn=f"sqlite:///{tmp_path / 'tidal.db'}",
        blob_dir=str(tmp_path / "blobs"),
        api_key="it-key",
        served_model=MODEL,
        max_inflight=4,
        poll_interval_s=0.5,
    )
    repo = make_repository(cfg.dsn, cfg.blob_dir)
    api = create_app(cfg, repo)

    # -- phase 1: online-only baseline, same engine, same session ----------
    baseline_gen = OnlineLoadGen(base_url=server.base_url, model=MODEL, max_tokens=16)
    await baseline_gen.run(poisson_schedule(ONLINE_RPS, BASELINE_S, seed=101))
    baseline = summarize_latencies(baseline_gen.latencies())
    assert baseline["count"] > 0.5 * ONLINE_RPS * BASELINE_S, (
        f"baseline load produced too few samples: {baseline}"
    )
    assert not [r for r in baseline_gen.records if not r.ok], "online errors during the baseline"

    content, custom_ids = batch_input(BATCH_ITEMS)

    transport = httpx.ASGITransport(app=api)
    async with httpx.AsyncClient(transport=transport, base_url=GATEWAY_URL, timeout=60.0) as http:
        sdk = AsyncOpenAI(
            api_key=cfg.api_key, base_url=f"{GATEWAY_URL}/v1", http_client=http, max_retries=0
        )

        uploaded = await sdk.files.create(file=("input.jsonl", content), purpose="batch")
        batch = await sdk.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"test": "a8"},
        )
        assert batch.status == "validating"
        assert batch.request_counts.total == BATCH_ITEMS

        # -- phase 2: the dispatcher injects while online load continues ---
        client = VllmClient(cfg)
        dispatcher = Dispatcher(cfg, repo, client, finalize=finalize_if_done)
        reports: list[tuple[float, int, int, int]] = []
        stop = asyncio.Event()
        t0 = time.perf_counter()

        async def control() -> None:
            await dispatcher.startup()
            while not stop.is_set():
                report = await dispatcher.tick(datetime.now(UTC))
                reports.append(
                    (time.perf_counter() - t0, report.target, report.submitted, report.inflight)
                )
                with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=cfg.poll_interval_s)

        controller = asyncio.create_task(control(), name="dispatcher")
        try:
            loaded_gen = OnlineLoadGen(base_url=server.base_url, model=MODEL, max_tokens=16)
            await loaded_gen.run(poisson_schedule(ONLINE_RPS, WINDOW_S, seed=202))
            window_end = time.perf_counter() - t0

            # -- phase 3: let the rest of the pool drain and finalize ------
            deadline = time.perf_counter() + DRAIN_TIMEOUT_S
            record = None
            while time.perf_counter() < deadline:
                record = await asyncio.to_thread(repo.get_batch, batch.id)
                if record is not None and record.status in TERMINAL:
                    break
                await asyncio.sleep(1.0)
        finally:
            stop.set()
            with contextlib.suppress(Exception):
                await controller
            with contextlib.suppress(Exception):
                await dispatcher.shutdown()
            with contextlib.suppress(Exception):
                await client.aclose()

        loaded = summarize_latencies(loaded_gen.latencies())
        in_window = [r for r in reports if r[0] <= window_end]
        coverage = sum(1 for r in in_window if r[3] > 0) / len(in_window) if in_window else 0.0
        print(
            f"\n[a8] baseline p50={baseline['p50']:.3f}s p99={baseline['p99']:.3f}s "
            f"n={baseline['count']}"
            f"\n[a8] loaded   p50={loaded['p50']:.3f}s p99={loaded['p99']:.3f}s "
            f"n={loaded['count']} ratio={loaded['p99'] / baseline['p99']:.2f}x"
            f"\n[a8] ticks={len(reports)} in-window={len(in_window)} coverage={coverage:.0%} "
            f"peak_target={max((r[1] for r in reports), default=0)} "
            f"submitted={sum(r[2] for r in reports)}"
        )

        # -- the batch finished, through the SDK, with a valid output file --
        assert record is not None and record.status is BatchStatus.COMPLETED, (
            f"batch did not complete within {DRAIN_TIMEOUT_S:.0f}s "
            f"(status={getattr(record, 'status', None)})"
        )

        final = await sdk.batches.retrieve(batch.id)
        assert final.status == "completed"
        assert final.output_file_id
        assert final.request_counts.total == BATCH_ITEMS
        assert final.request_counts.completed == BATCH_ITEMS
        assert final.request_counts.failed == 0

        downloaded = await sdk.files.content(final.output_file_id)
        lines = [ln for ln in downloaded.text.splitlines() if ln.strip()]
        assert len(lines) == BATCH_ITEMS

        seen = []
        for raw in lines:  # every line, not a sample
            payload = json.loads(raw)
            assert payload["error"] is None, payload
            response = payload["response"]
            assert response["status_code"] == 200
            body = response["body"]
            assert body["choices"][0]["message"]["content"] is not None
            assert body["usage"]["completion_tokens"] > 0
            seen.append(payload["custom_id"])
        assert sorted(seen) == sorted(custom_ids)
        assert len(set(seen)) == BATCH_ITEMS

    # -- online was not sacrificed ----------------------------------------
    assert not [r for r in loaded_gen.records if not r.ok], "online errors under batch load"
    assert coverage >= MIN_COVERAGE, (
        f"batch work covered only {coverage:.0%} of the loaded window; the p99 "
        f"comparison would be vacuous. Raise TIDAL_IT_BATCH_ITEMS above {BATCH_ITEMS}."
    )
    assert loaded["p99"] < LATENCY_BUDGET * baseline["p99"], (
        f"online p99 {loaded['p99']:.3f}s exceeded {LATENCY_BUDGET}x the in-session "
        f"baseline {baseline['p99']:.3f}s"
    )
    # Note for the paper, measured here: the *median* moves more than the tail
    # (2.7x vs 1.6x on this box). Co-serving adds a roughly constant per-step
    # cost to every request, so it dominates a fast median while the Poisson
    # tail was already slow. Reporting only p99 flatters the technique; the
    # harness records the full distribution for exactly this reason.

    # The controller respected its own bounds while doing it.
    assert max(r[1] for r in reports) <= cfg.max_inflight, "AIMD exceeded max_inflight"
    assert sum(r[2] for r in reports) >= BATCH_ITEMS, "not every item was submitted"
