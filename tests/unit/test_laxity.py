"""Laxity escalation: pure math, deterministic clocks only."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tidal.config import TidalConfig
from tidal.dispatcher.laxity import (
    RATE_EPSILON,
    ObservedRate,
    TokenBucket,
    laxity_seconds,
    priority_for,
    urgency,
)

WINDOW_S = 24 * 3600.0
T0 = datetime(2026, 8, 6, 12, 0, 0)
EXPIRES = T0 + timedelta(hours=24)


class FakeClock:
    """Monotonic-ish clock we advance by hand."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


# --------------------------------------------------------------------------- laxity_seconds


def test_laxity_is_slack_minus_drain_time():
    # 10h left, 3600 items at 1/s → 1h of work → 9h of slack.
    lax = laxity_seconds(EXPIRES, EXPIRES - timedelta(hours=10), 3600, 1.0)
    assert lax == pytest.approx(9 * 3600.0)


def test_laxity_accepts_epoch_floats_as_well_as_datetimes():
    lax_dt = laxity_seconds(EXPIRES, T0, 100, 10.0)
    lax_f = laxity_seconds(EXPIRES.timestamp(), T0.timestamp(), 100, 10.0)
    assert lax_dt == pytest.approx(lax_f) == pytest.approx(WINDOW_S - 10.0)


def test_laxity_with_no_remaining_items_is_pure_slack():
    assert laxity_seconds(EXPIRES, T0, 0, 0.0) == pytest.approx(WINDOW_S)


def test_laxity_rate_epsilon_floor_makes_stalled_batch_hopeless():
    # rate 0 (and negative / NaN-ish rates) must not divide by zero; they clamp
    # to RATE_EPSILON, which makes the drain time enormous → deeply negative laxity.
    stalled = laxity_seconds(EXPIRES, T0, 10, 0.0)
    assert stalled == pytest.approx(WINDOW_S - 10 / RATE_EPSILON)
    assert stalled < 0
    assert laxity_seconds(EXPIRES, T0, 10, -5.0) == stalled


def test_laxity_goes_negative_past_expiry():
    assert laxity_seconds(EXPIRES, EXPIRES + timedelta(hours=1), 0, 1.0) == pytest.approx(-3600.0)


# --------------------------------------------------------------------------- urgency


def test_urgency_is_zero_when_laxity_equals_full_window():
    assert urgency(WINDOW_S, WINDOW_S) == 0.0


def test_urgency_clamps_at_zero_for_laxity_beyond_the_window():
    assert urgency(WINDOW_S * 3, WINDOW_S) == 0.0


def test_urgency_is_exactly_one_at_zero_laxity_and_clamps_below():
    assert urgency(0.0, WINDOW_S) == 1.0
    assert urgency(-WINDOW_S * 10, WINDOW_S) == 1.0


def test_urgency_is_linear_in_between():
    assert urgency(WINDOW_S / 2, WINDOW_S) == pytest.approx(0.5)
    assert urgency(WINDOW_S * 0.25, WINDOW_S) == pytest.approx(0.75)


def test_urgency_with_nonpositive_window_degenerates_safely():
    assert urgency(10.0, 0.0) == 0.0
    assert urgency(0.0, 0.0) == 1.0
    assert urgency(-1.0, -5.0) == 1.0


def test_urgency_monotone_in_time_and_backlog():
    cfg_rate = 1.0
    remaining = 1000

    # monotone non-decreasing as the clock advances toward expiry
    prev = -1.0
    for hours in range(0, 25):
        now = EXPIRES - timedelta(hours=24 - hours)
        u = urgency(laxity_seconds(EXPIRES, now, remaining, cfg_rate), WINDOW_S)
        assert u >= prev
        prev = u
    assert prev == 1.0

    # monotone non-decreasing as the backlog grows, at a fixed instant
    now = T0 + timedelta(hours=6)
    prev = -1.0
    for backlog in (0, 10, 100, 1_000, 10_000, 100_000, 1_000_000):
        u = urgency(laxity_seconds(EXPIRES, now, backlog, cfg_rate), WINDOW_S)
        assert u >= prev
        prev = u
    assert prev == 1.0


# --------------------------------------------------------------------------- priority_for


def test_on_track_batch_stays_at_100():
    cfg = TidalConfig()
    # a minute into a 24h window, 10 items left, healthy rate → laxity ≈ window
    now = T0 + timedelta(minutes=1)
    lax = laxity_seconds(EXPIRES, now, 10, 1.0)
    u = urgency(lax, WINDOW_S)
    assert u < 0.01
    assert priority_for(u, cfg) == cfg.batch_priority_max == 100


def test_at_risk_batch_escalates_early():
    cfg = TidalConfig()
    # 2h in, 100k items left, crawling at 0.05 items/s → 23 days of work in 22h
    now = T0 + timedelta(hours=2)
    u = urgency(laxity_seconds(EXPIRES, now, 100_000, 0.05), WINDOW_S)
    assert u == 1.0
    assert priority_for(u, cfg) == cfg.batch_priority_min == 1


def test_priority_capped_at_1_unless_strict():
    lax_cfg = TidalConfig(sla_strict=False)
    strict_cfg = TidalConfig(sla_strict=True)

    assert priority_for(1.0, lax_cfg) == 1
    assert priority_for(1.0, strict_cfg) == 0

    # strict only reaches 0 at urgency == 1.0 exactly; just below stays >= min
    assert priority_for(0.999999, strict_cfg) == strict_cfg.batch_priority_min
    assert priority_for(0.999999, lax_cfg) == lax_cfg.batch_priority_min


def test_priority_is_linear_and_rounds_half_up_at_band_edges():
    cfg = TidalConfig()  # 100 → 1 over 99 points
    assert priority_for(0.0, cfg) == 100
    assert priority_for(0.5, cfg) == 51  # 100 - 49.5 = 50.5 → 51 (half-up, not banker's)
    assert priority_for(0.25, cfg) == 75  # 100 - 24.75 = 75.25 → 75
    assert priority_for(0.99, cfg) == 2  # 100 - 98.01 = 1.99 → 2
    assert priority_for(1.0, cfg) == 1


def test_priority_is_monotone_non_increasing_in_urgency():
    cfg = TidalConfig()
    prev = 101
    for i in range(0, 1001):
        p = priority_for(i / 1000.0, cfg)
        assert p <= prev
        assert cfg.batch_priority_min <= p <= cfg.batch_priority_max
        prev = p


def test_priority_clamps_out_of_range_urgency():
    cfg = TidalConfig()
    assert priority_for(-3.0, cfg) == cfg.batch_priority_max
    assert priority_for(7.0, cfg) == cfg.batch_priority_min


def test_priority_respects_custom_bands():
    cfg = TidalConfig(batch_priority_max=10, batch_priority_min=2)
    assert priority_for(0.0, cfg) == 10
    assert priority_for(1.0, cfg) == 2
    assert priority_for(0.5, cfg) == 6


# --------------------------------------------------------------------------- ObservedRate


def test_observed_rate_returns_epsilon_when_nothing_completed():
    clock = FakeClock()
    r = ObservedRate(window_s=60.0, now=clock)
    assert r.rate() == RATE_EPSILON


def test_observed_rate_counts_completions_over_the_elapsed_span():
    clock = FakeClock()
    r = ObservedRate(window_s=60.0, now=clock, min_span_s=1.0)
    for _ in range(20):
        clock.advance(1.0)
        r.record()
    # 20 completions over a 20s span
    assert r.rate() == pytest.approx(1.0, rel=0.1)


def test_observed_rate_span_is_floored_so_a_burst_does_not_explode_the_rate():
    clock = FakeClock()
    r = ObservedRate(window_s=60.0, now=clock, min_span_s=1.0)
    for _ in range(5):
        r.record()  # five completions at t=0
    clock.advance(0.001)
    assert r.rate() == pytest.approx(5.0)  # 5 / min_span, not 5 / 0.001


def test_observed_rate_evicts_samples_outside_the_window():
    clock = FakeClock()
    r = ObservedRate(window_s=10.0, now=clock, min_span_s=1.0)
    for _ in range(10):
        clock.advance(1.0)
        r.record()
    assert r.rate() == pytest.approx(1.0, rel=0.2)
    clock.advance(100.0)  # everything ages out
    assert r.rate() == RATE_EPSILON


def test_observed_rate_decays_as_the_window_slides():
    clock = FakeClock()
    r = ObservedRate(window_s=10.0, now=clock, min_span_s=1.0)
    for _ in range(10):
        clock.advance(1.0)
        r.record()
    busy = r.rate()
    clock.advance(5.0)  # half the window now empty
    assert r.rate() < busy


def test_observed_rate_record_accepts_batches_and_explicit_timestamps():
    clock = FakeClock()
    r = ObservedRate(window_s=60.0, now=clock, min_span_s=1.0)
    r.record(n=4, now=0.0)
    r.record(n=6, now=10.0)
    assert r.rate(now=10.0) == pytest.approx(1.0)


def test_observed_rate_never_returns_zero():
    clock = FakeClock()
    r = ObservedRate(window_s=60.0, now=clock, epsilon=1e-3)
    r.record(n=0)
    assert r.rate() == 1e-3


# --------------------------------------------------------------------------- TokenBucket


def test_token_bucket_starts_full_and_drains():
    b = TokenBucket(rate_items_per_s=0.2, burst=2)
    assert b.try_take(0.0) is True
    assert b.try_take(0.0) is True
    assert b.try_take(0.0) is False


def test_token_bucket_refills_at_the_configured_rate():
    b = TokenBucket(rate_items_per_s=0.2, burst=2)  # one token per 5s
    assert b.try_take(0.0) and b.try_take(0.0)
    assert b.try_take(4.9) is False
    assert b.try_take(5.0) is True
    assert b.try_take(5.0) is False
    assert b.try_take(10.0) is True


def test_token_bucket_caps_at_burst():
    b = TokenBucket(rate_items_per_s=1.0, burst=3)
    assert [b.try_take(0.0) for _ in range(4)] == [True, True, True, False]
    # idle for a minute: refill is capped at burst, not 60 tokens
    assert [b.try_take(60.0) for _ in range(4)] == [True, True, True, False]


def test_token_bucket_can_start_empty():
    b = TokenBucket(rate_items_per_s=1.0, burst=2, initial_tokens=0.0)
    assert b.try_take(0.0) is False
    assert b.try_take(1.0) is True


def test_token_bucket_with_zero_rate_never_refills():
    b = TokenBucket(rate_items_per_s=0.0, burst=1)
    assert b.try_take(0.0) is True
    assert b.try_take(1e9) is False


def test_token_bucket_ignores_clock_going_backwards():
    b = TokenBucket(rate_items_per_s=1.0, burst=1, initial_tokens=0.0)
    assert b.try_take(0.0) is False  # first call establishes the epoch
    assert b.try_take(10.0) is True  # 10s of refill, capped at burst=1
    assert b.try_take(5.0) is False  # backwards clock must not mint tokens
    assert b.try_take(11.0) is True


def test_token_bucket_from_config_uses_floor_rate():
    cfg = TidalConfig(floor_items_per_s=0.5)
    b = TokenBucket.from_config(cfg)
    assert b.rate_items_per_s == 0.5
    assert b.try_take(0.0) is True


def test_token_bucket_rejects_nonpositive_burst():
    with pytest.raises(ValueError):
        TokenBucket(rate_items_per_s=1.0, burst=0)
