"""AIMD in-flight controller: pure state machine, no I/O and no clock."""

from __future__ import annotations

import pytest

from tidal.config import TidalConfig
from tidal.dispatcher.aimd import AimdController

CALM_KV = 0.10  # < kv_low   → additive increase
STEADY_KV = 0.75  # between kv_low and kv_high → hold
HOT_KV = 0.95  # > kv_high  → multiplicative decrease


def grow_to(ctl: AimdController, n: int) -> int:
    """Drive additive increase until the target reaches n (or stops moving)."""
    for _ in range(n * 4 + 8):
        before = ctl.target
        ctl.update(kv_usage=CALM_KV, waiting=0, inflight=0)
        if ctl.target >= n or ctl.target == before:
            break
    return ctl.target


def test_target_starts_at_zero():
    assert AimdController(TidalConfig()).target == 0


def test_aimd_halves_on_online_queueing_and_grows_on_low_kv():
    cfg = TidalConfig(max_inflight=64, kv_low=0.70, kv_high=0.85, online_waiting_tolerance=0)
    ctl = AimdController(cfg)

    # additive increase: +1 per tick while KV is below the low watermark
    assert ctl.update(CALM_KV, waiting=0, inflight=0) == 1
    assert ctl.update(CALM_KV, waiting=0, inflight=1) == 2
    assert grow_to(ctl, 16) == 16

    # an online request queues (waiting exceeds our in-flight lower bound) → halve
    assert ctl.update(CALM_KV, waiting=17, inflight=16) == 8
    assert ctl.update(CALM_KV, waiting=9, inflight=8) == 4

    # pressure gone, KV low again → back to additive increase
    assert ctl.update(CALM_KV, waiting=4, inflight=4) == 5


def test_aimd_halves_on_high_kv_even_without_online_queueing():
    cfg = TidalConfig(max_inflight=64)
    ctl = AimdController(cfg)
    grow_to(ctl, 10)
    assert ctl.update(HOT_KV, waiting=0, inflight=10) == 5


def test_aimd_holds_between_watermarks():
    cfg = TidalConfig(max_inflight=64, kv_low=0.70, kv_high=0.85)
    ctl = AimdController(cfg)
    grow_to(ctl, 8)
    for _ in range(5):
        assert ctl.update(STEADY_KV, waiting=0, inflight=8) == 8


def test_waiting_is_a_lower_bound_discounted_by_our_own_inflight():
    """vllm:num_requests_waiting is aggregate; our own queued batch items must
    not be mistaken for online queueing (spec §7)."""
    cfg = TidalConfig(max_inflight=64, online_waiting_tolerance=0)
    ctl = AimdController(cfg)
    grow_to(ctl, 8)
    # all 8 waiters could be our own batch items → no back-off, keep growing
    assert ctl.update(CALM_KV, waiting=8, inflight=8) == 9
    # one more waiter than we can explain → online traffic is queued → halve
    assert ctl.update(CALM_KV, waiting=10, inflight=9) == 4


def test_tolerance_allows_some_online_queueing_before_backing_off():
    cfg = TidalConfig(max_inflight=64, online_waiting_tolerance=2)
    ctl = AimdController(cfg)
    grow_to(ctl, 8)
    assert ctl.update(CALM_KV, waiting=10, inflight=8) == 9  # lb == 2, not > 2
    assert ctl.update(CALM_KV, waiting=13, inflight=9) == 4  # lb == 4 > 2 → halve


def test_aimd_never_exceeds_max_inflight():
    cfg = TidalConfig(max_inflight=4)
    ctl = AimdController(cfg)
    for _ in range(50):
        ctl.update(CALM_KV, waiting=0, inflight=ctl.target)
    assert ctl.target == 4


def test_aimd_never_below_floor():
    cfg = TidalConfig(max_inflight=64)
    ctl = AimdController(cfg)
    grow_to(ctl, 32)
    ctl.set_floor(3)  # a batch crossed floor_urgency
    for _ in range(20):  # sustained overload
        ctl.update(HOT_KV, waiting=999, inflight=1)
        assert ctl.target >= 3
    assert ctl.target == 3


def test_floor_of_zero_lets_target_collapse_to_zero():
    cfg = TidalConfig(max_inflight=64)
    ctl = AimdController(cfg)
    grow_to(ctl, 8)
    for _ in range(10):
        ctl.update(HOT_KV, waiting=999, inflight=0)
    assert ctl.target == 0


def test_set_floor_raises_the_target_immediately():
    cfg = TidalConfig(max_inflight=64)
    ctl = AimdController(cfg)
    assert ctl.target == 0
    ctl.set_floor(2)
    assert ctl.target == 2
    ctl.set_floor(0)  # lowering the floor does not drop the target
    assert ctl.target == 2
    assert ctl.floor == 0


def test_floor_is_clamped_to_max_inflight_and_non_negative():
    cfg = TidalConfig(max_inflight=4)
    ctl = AimdController(cfg)
    ctl.set_floor(99)
    assert ctl.floor == 4
    assert ctl.target == 4
    ctl.set_floor(-5)
    assert ctl.floor == 0


def test_halving_rounds_down_but_stops_at_one_then_zero():
    cfg = TidalConfig(max_inflight=64)
    ctl = AimdController(cfg)
    grow_to(ctl, 5)
    assert ctl.update(HOT_KV, waiting=0, inflight=5) == 2  # 5 // 2
    assert ctl.update(HOT_KV, waiting=0, inflight=2) == 1
    assert ctl.update(HOT_KV, waiting=0, inflight=1) == 0
    assert ctl.update(HOT_KV, waiting=0, inflight=0) == 0


def test_reset_opens_the_circuit_below_the_floor():
    """Engine down (A5): target must be forced to 0 regardless of any SLA floor."""
    cfg = TidalConfig(max_inflight=64)
    ctl = AimdController(cfg)
    grow_to(ctl, 8)
    ctl.set_floor(4)
    ctl.reset()
    assert ctl.target == 0
    # the floor still applies to subsequent updates
    assert ctl.update(HOT_KV, waiting=999, inflight=0) == 4


def test_update_is_pure_in_its_inputs():
    """No hidden clock/IO: identical inputs from identical state give identical output."""
    cfg = TidalConfig(max_inflight=64)
    a, b = AimdController(cfg), AimdController(cfg)
    for waiting, inflight, kv in [(0, 0, CALM_KV), (3, 1, HOT_KV), (0, 0, STEADY_KV)] * 4:
        assert a.update(kv, waiting, inflight) == b.update(kv, waiting, inflight)


def test_rejects_bad_watermarks():
    with pytest.raises(ValueError):
        AimdController(TidalConfig(kv_low=0.9, kv_high=0.5))
