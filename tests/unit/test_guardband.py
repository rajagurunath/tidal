"""KV guardband tracker: clamped EWMA of online KV-demand bursts (spec §5.3)."""

from __future__ import annotations

import pytest

from tidal.engine.guardband import KvGuardband


def _steady(g: KvGuardband, demand_frac: float, steps: int = 200, pool: int = 1000) -> None:
    for _ in range(steps):
        g.observe_online_demand(round(demand_frac * pool), pool)


def test_h_starts_fully_guarded_and_stays_clamped():
    g = KvGuardband(h_min=0.05, h_max=0.30)
    assert g.h() == pytest.approx(0.30)  # cold start: maximum protection

    quiet = KvGuardband(h_min=0.05, h_max=0.30)
    _steady(quiet, 0.0)
    assert quiet.h() == pytest.approx(0.05)

    greedy = KvGuardband(h_min=0.05, h_max=0.30)
    _steady(greedy, 1.0)
    assert greedy.h() == pytest.approx(0.30)


def test_h_tracks_demand_inside_the_band():
    g = KvGuardband(h_min=0.05, h_max=0.30)
    _steady(g, 0.15)
    assert g.h() == pytest.approx(0.15, abs=1e-6)


def test_ewma_responds_to_bursts():
    g = KvGuardband(h_min=0.01, h_max=0.90, alpha=0.3)
    _steady(g, 0.10)
    calm = g.h()
    assert calm == pytest.approx(0.10, abs=1e-6)
    g.observe_online_demand(800, 1000)  # burst: 80% of the pool
    burst = g.h()
    assert burst > calm
    assert burst == pytest.approx(0.10 + 0.3 * (0.80 - 0.10), abs=1e-6)
    _steady(g, 0.10, steps=50)  # and decays back
    assert g.h() == pytest.approx(calm, abs=1e-6)


def test_alpha_controls_responsiveness():
    slow = KvGuardband(h_min=0.01, h_max=0.90, alpha=0.05)
    fast = KvGuardband(h_min=0.01, h_max=0.90, alpha=0.5)
    for g in (slow, fast):
        _steady(g, 0.10)
        g.observe_online_demand(900, 1000)
    assert fast.h() > slow.h()


def test_hysteresis_gap_admit_stops_before_eviction_starts():
    """Admission stops at kv > 1-H; eviction only starts at kv > 1-H/2 > 1-H."""
    for h_target in (0.05, 0.10, 0.20, 0.30):
        g = KvGuardband(h_min=0.05, h_max=0.30)
        _steady(g, h_target)
        h = g.h()
        admit_stop, evict_start = 1.0 - h, 1.0 - h / 2.0
        assert admit_stop < evict_start  # eviction threshold is the HIGHER usage

        below = admit_stop - 1e-6  # plenty of room
        assert g.admit_ok(below) and not g.evict_needed(below)

        middle = (admit_stop + evict_start) / 2.0  # hold band: no admit, no evict
        assert not g.admit_ok(middle)
        assert not g.evict_needed(middle)

        above = evict_start + 1e-6  # further growth -> proactive eviction
        assert not g.admit_ok(above)
        assert g.evict_needed(above)


def test_threshold_boundaries_are_inclusive_then_strict():
    g = KvGuardband(h_min=0.10, h_max=0.10)
    h = g.h()
    assert g.admit_ok(1.0 - h)  # kv_usage <= 1 - h
    assert not g.admit_ok(1.0 - h + 1e-9)
    assert not g.evict_needed(1.0 - h / 2.0)  # kv_usage > 1 - h/2
    assert g.evict_needed(1.0 - h / 2.0 + 1e-9)


def test_empty_or_invalid_pool_is_a_noop():
    g = KvGuardband(h_min=0.05, h_max=0.30)
    _steady(g, 0.0)
    before = g.h()
    g.observe_online_demand(10, 0)
    g.observe_online_demand(10, -5)
    assert g.h() == before


def test_demand_over_pool_is_clamped_to_one():
    g = KvGuardband(h_min=0.05, h_max=0.90, alpha=1.0)
    g.observe_online_demand(5000, 1000)
    assert g.h() == pytest.approx(0.90)
    g.observe_online_demand(-3, 1000)  # negative allocation clamps to 0
    assert g.h() == pytest.approx(0.05)


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        KvGuardband(h_min=0.30, h_max=0.05)  # inverted band
    with pytest.raises(ValueError):
        KvGuardband(h_min=0.0, h_max=0.30)  # h must be > 0 for a real gap
    with pytest.raises(ValueError):
        KvGuardband(h_min=0.05, h_max=1.0)  # 1 - h must stay > 0
    with pytest.raises(ValueError):
        KvGuardband(h_min=0.05, h_max=0.30, alpha=0.0)
    with pytest.raises(ValueError):
        KvGuardband(h_min=0.05, h_max=0.30, alpha=1.5)


def test_hysteresis_invariant_holds_across_the_whole_band():
    """For every reachable H the ordering 0 < 1-H < 1-H/2 < 1 must hold."""
    g = KvGuardband(h_min=0.05, h_max=0.30)
    for demand in [i / 100.0 for i in range(0, 101)]:
        g.observe_online_demand(int(demand * 1000), 1000)
        h = g.h()
        assert 0.05 <= h <= 0.30
        assert 0.0 < 1.0 - h < 1.0 - h / 2.0 < 1.0
