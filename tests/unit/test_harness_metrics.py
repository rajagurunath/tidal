"""The harness's aggregation maths, exercised without a server.

Everything a result JSON reports is computed by these pure functions; if the
percentile or the ceiling arithmetic is wrong, every number in the paper is
wrong, and no amount of live-vLLM testing would catch it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from tidal.eval import harness
from tidal.eval.harness import (
    BatchCompletion,
    MetricSample,
    RunConfig,
    VllmServer,
    app,
    batch_limits,
    batch_summary,
    make_batch_items,
    parse_tidal_stats,
    percent_of_ceiling,
    resolve_preset,
    summarize_metrics,
    summarize_tick_reports,
    tidal_probe_line,
    write_logging_config,
)


def completion(index: int, finished: float, *, tokens: int = 10, error: str | None = None):
    return BatchCompletion(
        index=index,
        custom_id=f"item-{index}",
        started_at=max(0.0, finished - 1.0),
        finished_at=finished,
        latency_s=1.0,
        prompt_tokens=4,
        completion_tokens=tokens,
        error=error,
    )


# -- batch throughput --------------------------------------------------------


def test_batch_summary_counts_only_completions_inside_the_window():
    done = [completion(i, finished=float(i)) for i in range(1, 11)]  # 1s .. 10s
    summary = batch_summary(done, window_s=5.0)
    assert summary["completed"] == 10
    assert summary["completed_in_window"] == 5
    assert summary["output_tokens_in_window"] == 50
    assert summary["items_per_s"] == pytest.approx(1.0)
    assert summary["output_tokens_per_s"] == pytest.approx(10.0)
    assert summary["makespan_s"] == pytest.approx(10.0)


def test_batch_summary_flags_a_pool_that_drained_early():
    done = [completion(i, finished=float(i)) for i in range(1, 4)]
    assert batch_summary(done, window_s=60.0)["drained"] is True
    assert batch_summary(done, window_s=2.0)["drained"] is False


def test_batch_summary_separates_failures_from_completions():
    done = [completion(1, 1.0), completion(2, 2.0, error="HTTP 500")]
    summary = batch_summary(done, window_s=10.0)
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["output_tokens_in_window"] == 10


def test_unmeasured_item_latency_is_reported_as_missing_not_as_zero():
    """Technique A recovers completion *times* from the store but not per-item
    latency; a summary that averaged in a fabricated 0.0 would understate every
    other condition it is compared against."""
    done = [completion(1, 1.0), completion(2, 2.0)]
    done[0].latency_s = float("nan")
    summary = batch_summary(done, window_s=10.0)
    assert summary["completed"] == 2
    assert summary["item_latency"]["count"] == 1
    assert summary["item_latency"]["p50"] == pytest.approx(1.0)


def test_batch_summary_of_an_empty_run_is_zeroed_not_divided_by_zero():
    summary = batch_summary([], window_s=0.0)
    assert summary["completed"] == 0
    assert summary["items_per_s"] == 0.0
    assert summary["output_tokens_per_s"] == 0.0
    assert summary["makespan_s"] == 0.0


def test_pool_size_survives_a_run_where_nothing_finished():
    summary = batch_summary([], window_s=10.0, pool_size=400)
    assert summary["pool_size"] == 400
    assert summary["drained"] is False


# -- ceiling comparison ------------------------------------------------------


def test_percent_of_ceiling_is_relative_to_offline_only():
    results = {
        "offline_only": {"batch": {"output_tokens_per_s": 100.0}},
        "technique_a": {"batch": {"output_tokens_per_s": 60.0}},
        "technique_b": {"batch": {"output_tokens_per_s": 80.0}},
        "online_only": {"batch": {"output_tokens_per_s": 0.0}},
    }
    shares = percent_of_ceiling(results)
    assert shares["offline_only"] == pytest.approx(100.0)
    assert shares["technique_a"] == pytest.approx(60.0)
    assert shares["technique_b"] == pytest.approx(80.0)
    assert shares["online_only"] == pytest.approx(0.0)


def test_percent_of_ceiling_without_a_ceiling_run_is_empty():
    assert percent_of_ceiling({"technique_a": {"batch": {"output_tokens_per_s": 5.0}}}) == {}


# -- engine gauges -----------------------------------------------------------


def test_summarize_metrics_reports_means_and_peaks():
    samples = [
        MetricSample(t=0.0, running=1, waiting=0, kv_usage=0.1),
        MetricSample(t=1.0, running=3, waiting=2, kv_usage=0.5),
        MetricSample(t=2.0, running=2, waiting=1, kv_usage=0.3),
    ]
    stats = summarize_metrics(samples)
    assert stats["samples"] == 3
    assert stats["running_mean"] == pytest.approx(2.0)
    assert stats["running_max"] == 3
    assert stats["waiting_max"] == 2
    assert stats["kv_mean"] == pytest.approx(0.3)
    assert stats["kv_p95"] == pytest.approx(0.48)


def test_summarize_metrics_of_no_samples_is_a_count_not_a_crash():
    assert summarize_metrics([]) == {"samples": 0}


# -- technique A controller --------------------------------------------------


def _report(target, submitted, inflight, priorities=None, circuit_open=False):
    return {
        "target": target,
        "submitted": submitted,
        "inflight": inflight,
        "escalated_priorities": priorities or {},
        "circuit_open": circuit_open,
    }


def test_summarize_tick_reports_tracks_the_control_law():
    reports = [
        _report(4, 4, 4, {"b1": 100}),
        _report(2, 0, 2, {"b1": 100}),
        _report(1, 1, 1, {"b1": 40}),
    ]
    stats = summarize_tick_reports(reports)
    assert stats["ticks"] == 3
    assert stats["target_max"] == 4
    assert stats["target_min"] == 1
    assert stats["submitted_total"] == 5
    assert stats["min_priority"] == 40
    assert stats["escalated_ticks"] == 1
    assert stats["circuit_open_ticks"] == 0


def test_summarize_tick_reports_of_nothing():
    assert summarize_tick_reports([]) == {"ticks": 0}


# -- engine log parsing ------------------------------------------------------

LOG = """\
1754470000.500 INFO vllm.engine.core Starting engine
1754470001.000 INFO tidal.engine.scheduler TidalScheduler active: tau=8192 tbt_slo=200.0ms \
guard=0.20 cold_start_frac=0.25 kv_guardband=[0.05, 0.30] capabilities=['ok']
1754470002.250 INFO tidal.engine.scheduler tidal {'step': 100, 'online_tokens': 12, \
'batch_tokens': 340, 'held': 7, 'starved_slack': 0, 'x': 2048, 'kv': 0.12, \
'model_ready': False, 'mape': nan, 'evicted': 0, 'h': 0.05}
(EngineCore pid=87407) 1754470004.750 INFO tidal.engine.scheduler tidal {'step': 200, \
'online_tokens': 0, 'batch_tokens': 512, 'held': 0, 'starved_slack': 0, 'x': None, 'kv': 0.2, \
'model_ready': False, 'mape': nan, 'evicted': 0, 'h': 0.05}
1754470005.000 INFO vllm.engine.core Avg generation throughput: 3.4 tokens/s
"""


def test_parse_tidal_stats_extracts_the_periodic_telemetry():
    """The second line carries vLLM's `(EngineCore pid=…)` multiprocess prefix:
    the scheduler runs in the engine child, so *every* real telemetry line does.
    Anchoring the pattern at the start of the line finds nothing at all."""
    stats = parse_tidal_stats(LOG, epoch0=1754470000.0)
    assert [s["step"] for s in stats] == [100, 200]
    assert stats[0]["held"] == 7
    assert stats[0]["batch_tokens"] == 340
    assert stats[0]["t"] == pytest.approx(2.25)
    assert stats[1]["t"] == pytest.approx(4.75)
    # `nan` is neither a Python literal nor valid JSON: it reads back as None.
    assert stats[0]["mape"] is None
    assert json.dumps(stats)  # the whole thing must survive serialization


def test_offline_mode_lines_are_identifiable_by_a_null_cap():
    stats = parse_tidal_stats(LOG)
    offline = [s for s in stats if s["x"] is None]
    assert len(offline) == 1
    assert offline[0]["batch_tokens"] > 0
    assert offline[0]["online_tokens"] == 0


def test_parse_tidal_stats_ignores_unrelated_and_malformed_lines():
    assert parse_tidal_stats("nothing to see here") == []
    assert parse_tidal_stats("1754470002.250 INFO tidal.engine.scheduler tidal {broken") == []


def test_tidal_probe_line_found_and_absent():
    assert "TidalScheduler active" in (tidal_probe_line(LOG) or "")
    assert tidal_probe_line("no scheduler here") is None


# -- misc plumbing -----------------------------------------------------------


def test_logging_config_routes_both_vllm_and_tidal_loggers(tmp_path):
    path = write_logging_config(tmp_path / "logging.json")
    config = json.loads(Path(path).read_text())
    assert config["loggers"]["tidal"]["level"] == "INFO"
    assert config["loggers"]["vllm"]["level"] == "INFO"
    assert config["disable_existing_loggers"] is False
    assert "%(created)f" in config["formatters"]["tidal"]["format"]


def test_batch_items_are_deterministic_and_share_a_prefix():
    a = make_batch_items(5, model="m", max_tokens=8)
    assert a == make_batch_items(5, model="m", max_tokens=8)
    assert len(a) == 5
    assert all(item["max_tokens"] == 8 and item["model"] == "m" for item in a)
    prefixes = {item["messages"][0]["content"][:20] for item in a}
    assert len(prefixes) == 1  # shared prefix → prefix-cache hits


# ---------------------------------------------------------------------------
# engine shape: the GPU flags
#
# The harness was written against a Mac CPU build and hardcoded that fact in
# four places. Each is now a flag, and the thing these tests defend is that the
# *defaults did not move*: a bare `run` must produce the same command line, the
# same environment and the same connection pool it always did, or every CPU
# number in the paper is being compared against a differently-configured
# engine.
# ---------------------------------------------------------------------------


def test_run_config_defaults_are_still_the_cpu_testbeds():
    cfg = RunConfig(condition="online_only")
    assert cfg.enforce_eager is True
    assert cfg.hf_offline is True
    assert cfg.tensor_parallel == 1
    assert cfg.max_model_len == 4096
    assert cfg.batch_concurrency == 100
    assert cfg.warmup_s == 0.0


def test_default_command_is_byte_identical_to_the_pre_flag_harness():
    cmd = VllmServer(port=8400, vllm_bin="/bin/vllm", model="m").command()
    assert cmd == [
        "/bin/vllm",
        "serve",
        "m",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--scheduling-policy",
        "priority",
        "--port",
        "8400",
    ]


def test_command_reflects_the_gpu_flags():
    cmd = VllmServer(
        port=8400,
        vllm_bin="/bin/vllm",
        model="m",
        enforce_eager=False,
        tensor_parallel=8,
        max_model_len=32768,
    ).command()
    assert "--enforce-eager" not in cmd
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "8"
    assert cmd[cmd.index("--max-model-len") + 1] == "32768"
    # Orthogonal to the flavor switch: technique B still gets its scheduler.
    tidal_cmd = VllmServer(port=1, flavor="tidal", enforce_eager=False, tensor_parallel=4).command()
    assert "tidal.engine.scheduler.TidalScheduler" in tidal_cmd


def test_tensor_parallel_of_one_is_left_off_the_command_line():
    """`--tensor-parallel-size 1` is what vLLM does anyway; emitting it would
    change every existing single-GPU and CPU invocation for no benefit."""
    assert "--tensor-parallel-size" not in VllmServer(port=1, tensor_parallel=1).command()


def test_torchdynamo_is_disabled_only_under_enforce_eager(monkeypatch):
    """`--no-enforce-eager` with TORCHDYNAMO_DISABLE still inherited from the
    surrounding shell would be a silent no-op — exactly the failure mode the
    flag exists to prevent, so the variable is removed rather than not set."""
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    assert VllmServer(port=1).environment()["TORCHDYNAMO_DISABLE"] == "1"
    assert "TORCHDYNAMO_DISABLE" not in VllmServer(port=1, enforce_eager=False).environment()


def test_hf_offline_flag_wins_over_an_inherited_value(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert VllmServer(port=1, hf_offline=False).environment()["HF_HUB_OFFLINE"] == "0"
    monkeypatch.delenv("HF_HUB_OFFLINE")
    assert VllmServer(port=1).environment()["HF_HUB_OFFLINE"] == "1"
    assert VllmServer(port=1, hf_offline=False).environment()["HF_HUB_OFFLINE"] == "0"


def test_batch_limits_size_the_pool_in_both_directions():
    assert batch_limits(100).max_connections == 100
    wide = batch_limits(512)
    assert wide.max_connections == 512
    # Keepalives track the ceiling: a 512-wide pool that only keeps 20 alive
    # spends the run reconnecting.
    assert wide.max_keepalive_connections == 512


async def test_direct_submission_builds_a_client_with_the_requested_limits(monkeypatch):
    captured: dict = {}
    real_client = httpx.AsyncClient

    def factory(**kwargs):
        captured.update(kwargs)
        return real_client(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"usage": {}})),
        )

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    done = await harness.submit_batch_direct(
        "http://engine", make_batch_items(2), t0=0.0, max_connections=512
    )
    assert len(done) == 2
    assert captured["limits"].max_connections == 512


async def test_measure_threads_batch_concurrency_into_direct_submission(monkeypatch):
    """The flag is worthless if the pool size stops at RunConfig."""
    captured: dict = {}

    async def fake_submit(_base_url, _items, *, t0, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(harness, "submit_batch_direct", fake_submit)
    cfg = RunConfig(condition="offline_only", batch_items=2, batch_concurrency=512, port=9)
    await harness._measure(cfg, server=None)
    assert captured["max_connections"] == 512


# -- the preset --------------------------------------------------------------


def test_gpu_preset_flips_exactly_the_defaults_it_claims_to():
    assert resolve_preset(gpu_preset=False) == {
        "enforce_eager": True,
        "hf_offline": True,
        "batch_concurrency": 100,
        "warmup_s": 0.0,
    }
    assert resolve_preset(gpu_preset=True) == {
        "enforce_eager": False,
        "hf_offline": False,
        "batch_concurrency": 512,
        "warmup_s": 45.0,
    }


def test_explicit_flags_outrank_the_preset_in_both_directions():
    forced = resolve_preset(gpu_preset=True, enforce_eager=True, batch_concurrency=64)
    assert forced["enforce_eager"] is True  # CUDA-graph-free numbers on a GPU
    assert forced["batch_concurrency"] == 64
    assert forced["hf_offline"] is False  # untouched flags keep the preset
    assert resolve_preset(gpu_preset=False, hf_offline=False)["hf_offline"] is False
    # Zero is a value, not "unset": --gpu-preset --warmup-s 0 must be a way to
    # ask for the un-warmed measurement on a GPU.
    assert resolve_preset(gpu_preset=True, warmup_s=0)["warmup_s"] == 0.0
    assert resolve_preset(gpu_preset=False, warmup_s=10)["warmup_s"] == 10.0


# -- CLI wiring --------------------------------------------------------------


@pytest.fixture
def captured_config(monkeypatch, tmp_path):
    """Invoke `run` with the engine stubbed out; hand back the RunConfig built.

    The CLI's whole job for these flags is to assemble a RunConfig, so that is
    what is asserted on — no vLLM, no event loop work beyond the stub.
    """
    seen: list[RunConfig] = []

    async def fake_run_condition(cfg, *, log_dir=None):
        seen.append(cfg)
        return {
            "online": {"summary": {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}, "errors": 0},
            "batch": {
                "completed": 0,
                "pool_size": 0,
                "completed_in_window": 0,
                "output_tokens_per_s": 0.0,
                "makespan_s": 0.0,
                "drained": False,
            },
        }

    monkeypatch.setattr(harness, "run_condition", fake_run_condition)

    def invoke(*args: str) -> RunConfig:
        result = CliRunner().invoke(
            app,
            ["run", "--condition", "online_only", "--out", str(tmp_path / "r.json"), *args],
        )
        assert result.exit_code == 0, result.output
        return seen[-1]

    return invoke


def test_cli_without_the_new_flags_still_describes_the_cpu_testbed(captured_config):
    cfg = captured_config()
    assert (cfg.enforce_eager, cfg.hf_offline) == (True, True)
    assert (cfg.tensor_parallel, cfg.max_model_len, cfg.batch_concurrency) == (1, 4096, 100)
    assert cfg.warmup_s == 0.0


def test_cli_parses_each_engine_flag_into_the_run_config(captured_config):
    cfg = captured_config(
        "--no-enforce-eager",
        "--no-hf-offline",
        "--tensor-parallel",
        "8",
        "--max-model-len",
        "32768",
        "--batch-concurrency",
        "1024",
    )
    assert cfg.enforce_eager is False
    assert cfg.hf_offline is False
    assert cfg.tensor_parallel == 8
    assert cfg.max_model_len == 32768
    assert cfg.batch_concurrency == 1024


def test_cli_gpu_preset_flips_the_defaults(captured_config):
    cfg = captured_config("--gpu-preset")
    assert cfg.enforce_eager is False
    assert cfg.hf_offline is False
    assert cfg.batch_concurrency == 512
    # The preset is about engine *mode*, not topology: TP and context length
    # are node-specific and stay where they were.
    assert (cfg.tensor_parallel, cfg.max_model_len) == (1, 4096)


def test_cli_flags_after_gpu_preset_win(captured_config):
    cfg = captured_config("--gpu-preset", "--enforce-eager", "--batch-concurrency", "64")
    assert cfg.enforce_eager is True
    assert cfg.batch_concurrency == 64
    assert cfg.hf_offline is False


def test_cli_warmup_defaults_to_off_and_to_45s_under_the_preset(captured_config):
    assert captured_config().warmup_s == 0.0
    assert captured_config("--gpu-preset").warmup_s == 45.0


def test_cli_warmup_flag_wins_over_the_preset_in_both_directions(captured_config):
    assert captured_config("--warmup-s", "20").warmup_s == 20.0
    assert captured_config("--gpu-preset", "--warmup-s", "90").warmup_s == 90.0
    # Explicitly asking for the un-warmed measurement on a GPU.
    assert captured_config("--gpu-preset", "--warmup-s", "0").warmup_s == 0.0


def test_the_engine_shape_survives_into_the_result_document():
    """`run_condition` writes `asdict(cfg)` into the result JSON, so a result
    file says which engine produced it. The flags are useless as provenance if
    they are not serializable alongside everything else."""
    cfg = RunConfig(
        condition="naive", tensor_parallel=8, enforce_eager=False, hf_offline=False, warmup_s=45.0
    )
    payload = json.loads(json.dumps(asdict(cfg)))
    assert payload["tensor_parallel"] == 8
    assert payload["enforce_eager"] is False
    assert payload["hf_offline"] is False
    assert payload["max_model_len"] == 4096
    # A warmed run and an un-warmed one are not the same measurement, so the
    # result file has to say which one it is.
    assert payload["warmup_s"] == 45.0


# ---------------------------------------------------------------------------
# measurement warm-up
#
# `/health` answering 200 says the weights are loaded, not that the engine is at
# steady state. The first requests still pay CUDA-graph capture, torch.compile
# and a cold prefix cache — a cost that, unwarmed, lands inside the measured
# window of whichever condition the script happens to run first.
# ---------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that only moves when a request is served."""

    def __init__(self, step: float = 1.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += self.step


def warmup_client(clock: FakeClock, seen: list[dict]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        clock.tick()
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_warmup_stops_at_the_time_budget():
    clock = FakeClock(step=1.0)
    seen: list[dict] = []
    async with warmup_client(clock, seen) as client:
        stats = await harness.run_warmup(
            RunConfig(condition="offline_only", warmup_s=5.0), client=client, clock=clock
        )
    assert stats["requests"] == 5
    assert stats["elapsed_s"] == pytest.approx(5.0)
    assert len(seen) == 5


async def test_warmup_stops_at_the_request_cap_when_the_engine_is_fast():
    """A 45-second budget against an engine that answers in microseconds must
    not become thousands of requests: the cap is what makes the warm-up a
    bounded, one-off cost rather than a second load generator."""
    clock = FakeClock(step=0.001)
    seen: list[dict] = []
    async with warmup_client(clock, seen) as client:
        stats = await harness.run_warmup(
            RunConfig(condition="technique_a", warmup_s=45.0), client=client, clock=clock
        )
    assert stats["requests"] == harness.WARMUP_MAX_REQUESTS == 30


async def test_warmup_reuses_the_online_request_body_builder():
    """Warming with a differently-shaped request would warm the wrong paths."""
    clock = FakeClock()
    seen: list[dict] = []
    cfg = RunConfig(condition="naive", warmup_s=3.0, model="m", online_max_tokens=64, port=9)
    async with warmup_client(clock, seen) as client:
        await harness.run_warmup(cfg, client=client, clock=clock)
    expected = harness.OnlineLoadGen(base_url=cfg.base_url, model="m", max_tokens=64, priority=0)
    assert seen == [expected.body_for(i) for i in range(3)]


async def test_warmup_swallows_engine_errors_and_still_counts_them():
    """A refused or 500ing warm-up request is not a data point — it is still a
    request the engine was walked through, and it must not abort the run."""
    clock = FakeClock()

    def handler(_request: httpx.Request) -> httpx.Response:
        clock.tick()
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        stats = await harness.run_warmup(
            RunConfig(condition="online_only", warmup_s=3.0), client=client, clock=clock
        )
    assert stats["requests"] == 3


async def test_warmup_is_off_by_default_and_makes_no_requests():
    stats = await harness.run_warmup(RunConfig(condition="online_only"))
    assert stats == {"requests": 0, "elapsed_s": 0.0}


async def test_warmup_logs_exactly_one_line(capsys):
    clock = FakeClock()
    seen: list[dict] = []
    async with warmup_client(clock, seen) as client:
        await harness.run_warmup(
            RunConfig(condition="technique_b", warmup_s=2.0), client=client, clock=clock
        )
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["[technique_b] warmup: 2 requests in 2.0s"]


async def test_warmup_runs_before_t0_even_for_conditions_with_no_online_load(monkeypatch):
    """The whole point is that the cost falls *outside* the measured window,
    and warm-up is engine warm-up, not load shape: offline_only boots an engine,
    so it warms up too."""
    order: list[str] = []
    warmed_at: list[float] = []
    captured: dict = {}

    async def fake_warmup(cfg, **_kwargs):
        order.append("warmup")
        warmed_at.append(harness.time.perf_counter())
        return {"requests": 3, "elapsed_s": 1.0}

    async def fake_submit(_base_url, _items, *, t0, **_kwargs):
        order.append("batch")
        captured["t0"] = t0
        return []

    monkeypatch.setattr(harness, "run_warmup", fake_warmup)
    monkeypatch.setattr(harness, "submit_batch_direct", fake_submit)
    result = await harness._measure(
        RunConfig(condition="offline_only", batch_items=2, warmup_s=45.0, port=9), server=None
    )
    assert order == ["warmup", "batch"]
    assert warmed_at[0] <= captured["t0"]
    # And it is reported, so a result file says whether it was warmed.
    assert result["warmup"] == {"requests": 3, "elapsed_s": 1.0}


def test_warmup_never_reaches_the_engine_command_line():
    """It is a harness-side behaviour. vLLM has no such flag, so leaking it into
    argv would not slow the engine down — it would fail to start it."""
    for server in (
        VllmServer(port=8400, vllm_bin="/bin/vllm", model="m"),
        VllmServer(
            port=8400, vllm_bin="/bin/vllm", model="m", enforce_eager=False, tensor_parallel=8
        ),
    ):
        assert not any("warmup" in arg for arg in server.command())
        assert not any("warmup" in key.lower() for key in server.environment())
