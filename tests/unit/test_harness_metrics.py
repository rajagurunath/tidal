"""The harness's aggregation maths, exercised without a server.

Everything a result JSON reports is computed by these pure functions; if the
percentile or the ceiling arithmetic is wrong, every number in the paper is
wrong, and no amount of live-vLLM testing would catch it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tidal.eval.harness import (
    BatchCompletion,
    MetricSample,
    batch_summary,
    make_batch_items,
    parse_tidal_stats,
    percent_of_ceiling,
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
