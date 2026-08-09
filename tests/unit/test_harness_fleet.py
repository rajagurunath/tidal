"""The ``technique_a_fleet`` condition: N engines, one gateway, one store.

Everything testable without a GPU is tested here: the per-replica engine
environment (core isolation), the phase-shifted diurnal schedules, the CLI
plumbing, the per-replica aggregation, and the shape of the result document.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from typer.testing import CliRunner

from tidal.eval import harness
from tidal.eval.harness import (
    CONDITIONS,
    BatchCompletion,
    RunConfig,
    app,
    batch_summary_per_replica,
    diurnal_phases,
    replica_env,
    summarize_per_replica_ticks,
)
from tidal.eval.loadgen import RequestRecord, diurnal_schedule, schedule

A = "http://127.0.0.1:8399"
B = "http://127.0.0.1:8400"


# -- the condition exists ----------------------------------------------------


def test_the_fleet_condition_is_a_first_class_condition():
    assert "technique_a_fleet" in CONDITIONS
    # …and it is added, not substituted: every published condition still runs.
    assert {"online_only", "offline_only", "naive", "technique_a", "technique_b"} <= set(CONDITIONS)


# -- topology ----------------------------------------------------------------


def test_replica_ports_and_urls_are_consecutive_from_the_base():
    cfg = RunConfig(condition="technique_a_fleet", replicas=3, fleet_base_port=8399)
    assert cfg.replica_ports == [8399, 8400, 8401]
    assert cfg.replica_urls == [A, B, "http://127.0.0.1:8401"]


def test_a_one_replica_fleet_is_legal_and_is_just_one_engine():
    cfg = RunConfig(condition="technique_a_fleet", replicas=1)
    assert cfg.replica_urls == [A]


def test_a_fleet_of_none_is_refused():
    with pytest.raises(ValueError, match="replicas"):
        asyncio.run(harness.run_condition(RunConfig(condition="technique_a_fleet", replicas=0)))


def test_an_unknown_placement_policy_is_refused():
    cfg = RunConfig(condition="technique_a_fleet", fleet_placement="round-robin")
    with pytest.raises(ValueError, match="placement"):
        asyncio.run(harness.run_condition(cfg))


# -- core isolation ----------------------------------------------------------


def test_each_replica_gets_a_disjoint_core_range():
    envs = [replica_env(i, 2, cpu_count=14) for i in range(2)]
    ranges = [env["VLLM_CPU_OMP_THREADS_BIND"] for env in envs]
    spans = [tuple(int(x) for x in r.split("-")) for r in ranges]
    assert spans[0][1] < spans[1][0], f"overlapping core ranges: {ranges}"


def test_the_thread_count_matches_the_core_range_it_was_given():
    env = replica_env(0, 2, cpu_count=14)
    low, high = (int(x) for x in env["VLLM_CPU_OMP_THREADS_BIND"].split("-"))
    assert env["OMP_NUM_THREADS"] == str(high - low + 1)


def test_cores_are_reserved_for_the_harness_itself():
    # The gateway, the load generators and the metrics samplers all run in the
    # harness process; handing every core to the engines would measure the
    # harness starving rather than the engines co-serving.
    first = replica_env(0, 2, cpu_count=14, reserved=2)
    assert int(first["VLLM_CPU_OMP_THREADS_BIND"].split("-")[0]) == 2


def test_the_kv_cache_space_is_split_across_the_fleet():
    solo = replica_env(0, 1, cpu_count=14, kvcache_space_gb=4)
    pair = replica_env(0, 2, cpu_count=14, kvcache_space_gb=4)
    assert solo["VLLM_CPU_KVCACHE_SPACE"] == "4"
    assert pair["VLLM_CPU_KVCACHE_SPACE"] == "2"


def test_a_replica_always_gets_at_least_one_core_and_one_gigabyte():
    env = replica_env(3, 4, cpu_count=2, kvcache_space_gb=1)
    low, high = (int(x) for x in env["VLLM_CPU_OMP_THREADS_BIND"].split("-"))
    assert high >= low
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["VLLM_CPU_KVCACHE_SPACE"] == "1"


def test_a_tiny_box_drops_the_reservation_before_it_starves_a_replica():
    envs = [replica_env(i, 2, cpu_count=3, reserved=2) for i in (0, 1)]
    ranges = [env["VLLM_CPU_OMP_THREADS_BIND"] for env in envs]
    assert ranges == ["0-0", "1-1"], "reserving 2 of 3 cores would leave nothing to split"


def test_replica_envs_are_disjoint_for_any_plausible_fleet():
    for cpu_count in (4, 8, 10, 12, 14, 16, 24):
        for replicas in (2, 3, 4):
            seen: set[int] = set()
            for i in range(replicas):
                low, high = (
                    int(x)
                    for x in replica_env(i, replicas, cpu_count=cpu_count)[
                        "VLLM_CPU_OMP_THREADS_BIND"
                    ].split("-")
                )
                cores = set(range(low, high + 1))
                assert not (cores & seen), f"{cpu_count=} {replicas=} overlap at replica {i}"
                seen |= cores


# -- staggered diurnal load --------------------------------------------------


def test_two_replicas_default_to_half_a_period_apart():
    assert diurnal_phases(2) == pytest.approx([0.0, math.pi])


def test_phases_are_spread_evenly_over_the_period_for_any_fleet_size():
    assert diurnal_phases(4) == pytest.approx([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    assert diurnal_phases(1) == [0.0]


def test_an_explicit_phase_list_wins():
    assert diurnal_phases(2, "0,3.14159") == pytest.approx([0.0, 3.14159])
    assert diurnal_phases(2, " 1.0 , 2.0 ") == pytest.approx([1.0, 2.0])


def test_a_phase_list_that_does_not_match_the_fleet_is_an_error():
    with pytest.raises(ValueError, match="phase"):
        diurnal_phases(3, "0,3.14")


def test_schedule_threads_the_phase_into_the_diurnal_process():
    duration, period = 600.0, 300.0
    unshifted = schedule("diurnal", 2.0, duration, 5, period_s=period)
    shifted = schedule("diurnal", 2.0, duration, 5, period_s=period, phase=math.pi)
    assert unshifted == diurnal_schedule(2.0 * 0.2, 2.0, duration, period, 5, 0.0)
    assert shifted == diurnal_schedule(2.0 * 0.2, 2.0, duration, period, 5, math.pi)
    assert unshifted != shifted, "a half-period shift must actually change the arrivals"


def test_the_phase_is_ignored_by_the_poisson_shape():
    assert schedule("poisson", 1.0, 60.0, 3) == schedule("poisson", 1.0, 60.0, 3, phase=math.pi)


# -- per-replica aggregation -------------------------------------------------


def completion(index: int, replica: str | None, *, finished: float = 1.0) -> BatchCompletion:
    return BatchCompletion(
        index=index,
        custom_id=f"item-{index}",
        started_at=0.0,
        finished_at=finished,
        latency_s=finished,
        prompt_tokens=10,
        completion_tokens=5,
        replica=replica,
    )


def test_batch_completions_are_binned_by_the_replica_that_ran_them():
    completions = [completion(0, A), completion(1, B), completion(2, A)]
    per_replica = batch_summary_per_replica(completions, window_s=10.0)
    assert set(per_replica) == {A, B}
    assert per_replica[A]["completed"] == 2
    assert per_replica[B]["completed"] == 1
    assert per_replica[A]["output_tokens_in_window"] == 10


def test_completions_with_no_replica_are_not_silently_attributed():
    per_replica = batch_summary_per_replica([completion(0, None)], window_s=10.0)
    assert per_replica == {}


def test_tick_reports_summarize_per_replica_control():
    reports = [
        {
            "target": 6,
            "submitted": 2,
            "inflight": 3,
            "per_replica": {
                A: {"target": 4, "inflight": 2, "score": 0.5},
                B: {"target": 2, "inflight": 1, "score": 0.9},
            },
        },
        {
            "target": 4,
            "submitted": 1,
            "inflight": 2,
            "per_replica": {
                A: {"target": 4, "inflight": 2, "score": 0.7},
                B: {"target": 0, "inflight": 0, "score": None},
            },
        },
    ]
    summary = summarize_per_replica_ticks(reports)
    assert summary[A]["target_mean"] == 4.0
    assert summary[A]["down_ticks"] == 0
    assert summary[B]["down_ticks"] == 1
    assert summary[B]["target_mean"] == 1.0
    assert summary[A]["score_mean"] == pytest.approx(0.6)


def test_summarizing_a_single_engine_run_is_empty_not_a_crash():
    assert summarize_per_replica_ticks([{"target": 1, "submitted": 0, "inflight": 0}]) == {}


# -- the result document -----------------------------------------------------


@pytest.fixture
def stub_fleet(monkeypatch):
    """Run ``_measure_fleet`` with the dispatcher and the samplers stubbed out."""

    async def fake_run(cfg, items, gens, offsets_list, t0, *, replicas, placement):
        assert placement in ("fleet", "pinned")
        for index, gen in enumerate(gens):
            gen.records = [
                RequestRecord(
                    index=index,
                    scheduled_at=0.0,
                    started_at=0.0,
                    finished_at=0.5,
                    latency_s=0.5,
                    status=200,
                    prompt_tokens=7,
                    completion_tokens=3,
                )
            ]
        completions = [
            completion(0, replicas[0], finished=1.0),
            completion(1, replicas[-1], finished=2.0),
        ]
        reports = [
            {
                "t": 0.0,
                "target": 2,
                "submitted": 2,
                "inflight": 2,
                "per_replica": {
                    url: {"target": 1, "inflight": 1, "score": 0.1} for url in replicas
                },
            }
        ]
        return completions, reports, {c.custom_id: c.replica for c in completions}

    async def fake_sampler(metrics_url, stop, out, *, t0, interval_s=1.0):
        return None

    monkeypatch.setattr(harness, "_run_technique_a", fake_run)
    monkeypatch.setattr(harness, "sample_metrics", fake_sampler)
    monkeypatch.setattr(harness, "schedule", lambda *_a, **_k: [])
    return monkeypatch


async def test_the_fleet_result_carries_both_the_total_and_the_breakdown(stub_fleet):
    cfg = RunConfig(condition="technique_a_fleet", replicas=2, minutes=0.001, batch_items=2)
    async with asyncio.timeout(10):
        result = await harness._measure_fleet(cfg, servers=None)

    assert result["replicas"] == [A, B]
    assert result["placement"] == "fleet"
    assert result["batch"]["completed"] == 2, "the totals are still the fleet's totals"
    assert set(result["batch_per_replica"]) == {A, B}
    assert set(result["online_per_replica"]) == {A, B}
    assert set(result["metrics_per_replica"]) == {A, B}
    assert set(result["tick_per_replica"]) == {A, B}
    assert result["replica_of"] == {"item-0": A, "item-1": B}


async def test_every_batch_completion_says_which_engine_ran_it(stub_fleet):
    cfg = RunConfig(condition="technique_a_fleet", replicas=2, minutes=0.001, batch_items=2)
    result = await harness._measure_fleet(cfg, servers=None)
    assert [c["replica"] for c in result["batch_completions"]] == [A, B]


async def test_every_online_request_says_which_engine_served_it(stub_fleet):
    cfg = RunConfig(condition="technique_a_fleet", replicas=2, minutes=0.001, batch_items=2)
    result = await harness._measure_fleet(cfg, servers=None)
    assert [r["replica"] for r in result["online"]["requests"]] == [A, B]
    assert result["online"]["summary"]["count"] == 2, "the aggregate is the whole fleet"
    assert result["online_per_replica"][A]["requests"] == 1
    assert result["online_per_replica"][B]["phase"] == pytest.approx(math.pi)


async def test_the_pinned_control_run_is_recorded_in_the_document(stub_fleet):
    cfg = RunConfig(
        condition="technique_a_fleet", replicas=2, minutes=0.001, fleet_placement="pinned"
    )
    result = await harness._measure_fleet(cfg, servers=None)
    assert result["placement"] == "pinned"


async def test_the_fleet_launches_one_isolated_engine_per_replica(monkeypatch, tmp_path):
    """The engines are the experiment: each must get its own port and its own
    slice of the box, and the whole fleet must come down with the run."""
    built: list[harness.VllmServer] = []
    real_init = harness.VllmServer.__init__

    def spy_init(self, **kwargs):
        real_init(self, **kwargs)
        built.append(self)

    monkeypatch.setattr(harness.VllmServer, "__init__", spy_init)
    monkeypatch.setattr(harness.VllmServer, "start", lambda self: self)
    monkeypatch.setattr(harness.VllmServer, "wait_ready", lambda self, timeout=0: None)
    stopped: list[int] = []
    monkeypatch.setattr(harness.VllmServer, "stop", lambda self, grace=0: stopped.append(self.port))

    async def fake_measure(cfg, servers):
        return {"epoch0": 0.0}

    monkeypatch.setattr(harness, "_measure_fleet", fake_measure)

    cfg = RunConfig(condition="technique_a_fleet", replicas=2, fleet_base_port=8399)
    result = await harness.run_condition(cfg, log_dir=str(tmp_path))

    assert [server.port for server in built] == [8399, 8400]
    binds = [server.env_extra["VLLM_CPU_OMP_THREADS_BIND"] for server in built]
    assert len(set(binds)) == 2, f"replicas share a core range: {binds}"
    assert all(server.flavor == "stock" for server in built)
    assert sorted(stopped) == [8399, 8400], "every engine is torn down"
    assert result["condition"] == "technique_a_fleet"
    assert len(result["engine_logs"]) == 2


# -- CLI ---------------------------------------------------------------------


@pytest.fixture
def captured_config(monkeypatch, tmp_path):
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
            [
                "run",
                "--condition",
                "technique_a_fleet",
                "--out",
                str(tmp_path / "r.json"),
                *args,
            ],
        )
        assert result.exit_code == 0, result.output
        return seen[-1]

    return invoke


def test_the_fleet_defaults_to_two_replicas_and_the_load_aware_policy(captured_config):
    cfg = captured_config()
    assert cfg.replicas == 2
    assert cfg.fleet_placement == "fleet"
    assert cfg.diurnal_phase_list == ""
    assert cfg.fleet_base_port == 8399


def test_every_fleet_flag_reaches_the_run_config(captured_config):
    cfg = captured_config(
        "--replicas",
        "3",
        "--fleet-placement",
        "pinned",
        "--diurnal-phase-list",
        "0,2.1,4.2",
        "--fleet-base-port",
        "8500",
    )
    assert cfg.replicas == 3
    assert cfg.fleet_placement == "pinned"
    assert cfg.diurnal_phase_list == "0,2.1,4.2"
    assert cfg.replica_ports == [8500, 8501, 8502]


def test_the_fleet_flags_are_serialized_with_every_other_run_setting():
    from dataclasses import asdict

    cfg = RunConfig(condition="technique_a_fleet", replicas=4, fleet_placement="pinned")
    document = asdict(cfg)
    assert document["replicas"] == 4
    assert document["fleet_placement"] == "pinned"
