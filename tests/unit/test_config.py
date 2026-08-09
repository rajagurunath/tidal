"""``TidalConfig`` wiring — in particular the fleet's replica list.

``vllm_replicas`` is the one place a single-engine deployment becomes a
multi-engine one, so the tests pin both directions: an empty value must be
*exactly* the pre-fleet single-engine configuration, and a populated one must
survive the environment round-trip that every other field already gets.
"""

from __future__ import annotations

import pytest

from tidal.config import TidalConfig


def test_no_replicas_configured_is_the_single_engine_deployment():
    cfg = TidalConfig()
    assert cfg.vllm_replicas == ""
    assert cfg.replica_urls == [cfg.vllm_base_url]
    assert cfg.replica_metrics_urls == [cfg.vllm_metrics_url]


def test_replica_urls_splits_the_comma_list_in_order():
    cfg = TidalConfig(vllm_replicas="http://a:8000,http://b:8001,http://c:8002")
    assert cfg.replica_urls == ["http://a:8000", "http://b:8001", "http://c:8002"]


def test_replica_urls_tolerates_whitespace_blanks_and_trailing_slashes():
    cfg = TidalConfig(vllm_replicas=" http://a:8000/ , ,http://b:8001 ,")
    assert cfg.replica_urls == ["http://a:8000", "http://b:8001"]


def test_a_list_of_only_separators_falls_back_to_the_single_engine():
    cfg = TidalConfig(vllm_replicas=" , , ")
    assert cfg.replica_urls == [cfg.vllm_base_url]


def test_duplicate_replicas_are_rejected_rather_than_silently_deduped():
    # Two entries for one engine would give it two AIMD controllers fighting
    # over the same GPU — the exact bug the fleet exists to avoid.
    cfg = TidalConfig(vllm_replicas="http://a:8000,http://a:8000/")
    with pytest.raises(ValueError, match="duplicate"):
        _ = cfg.replica_urls


def test_metrics_urls_are_derived_per_replica():
    cfg = TidalConfig(vllm_replicas="http://a:8000,http://b:8001")
    assert cfg.replica_metrics_urls == ["http://a:8000/metrics", "http://b:8001/metrics"]


def test_vllm_metrics_url_is_honoured_verbatim_for_the_single_engine():
    # A single-engine deployment may point the scrape somewhere that is not
    # `${base}/metrics` (a sidecar exporter); the fleet must not break that.
    cfg = TidalConfig(vllm_metrics_url="http://sidecar:9100/federate")
    assert cfg.replica_metrics_urls == ["http://sidecar:9100/federate"]


def test_replicas_round_trip_through_the_environment(monkeypatch):
    monkeypatch.setenv("TIDAL_VLLM_REPLICAS", "http://a:8000,http://b:8001")
    cfg = TidalConfig.from_env()
    assert cfg.replica_urls == ["http://a:8000", "http://b:8001"]
