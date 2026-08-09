"""Fleet placement — pure, deterministic, no engine and no clock.

``tidal.dispatcher.fleet`` is the only place that decides *which* replica a
batch item goes to. It is a pure function of injected state precisely so these
properties can be pinned without a GPU:

* headroom is a hard filter (never oversubscribe a replica's AIMD target);
* a dead replica is not a candidate, however much headroom it nominally has;
* prefix groups are sticky, so a batch's shared prefix keeps hitting the same
  engine's prefix cache — until that engine is loaded enough that the cache hit
  is no longer worth the queueing, at which point stickiness is abandoned;
* identical inputs give identical output, so a replay of a run's tick log
  reproduces its placement exactly.
"""

from __future__ import annotations

from tidal.dispatcher.fleet import (
    STICKY_EVICT_SCORE,
    LoadScore,
    choose_replica,
    headroom,
    load_score,
)
from tidal.dispatcher.vllm_client import EngineMetrics

A = "http://a:8000"
B = "http://b:8001"
C = "http://c:8002"


def m(running: int = 0, waiting: int = 0, kv: float = 0.0) -> EngineMetrics:
    return EngineMetrics(running=running, waiting=waiting, kv_usage=kv)


# -- LoadScore --------------------------------------------------------------


def test_an_idle_engine_scores_zero():
    assert load_score(m()) == 0.0


def test_the_score_is_the_documented_weighted_sum():
    score = LoadScore(w_kv=1.0, w_running=0.5, w_waiting=1.0, running_norm=32.0)
    assert score(m(kv=0.5)) == 0.5
    assert score(m(running=32)) == 0.5
    assert score(m(waiting=32)) == 1.0
    assert score(m(running=16, waiting=8, kv=0.25)) == 0.25 + 0.25 + 0.25


def test_the_score_rises_with_every_gauge_it_reads():
    base = load_score(m(running=4, waiting=0, kv=0.3))
    assert load_score(m(running=8, waiting=0, kv=0.3)) > base
    assert load_score(m(running=4, waiting=2, kv=0.3)) > base
    assert load_score(m(running=4, waiting=0, kv=0.6)) > base


def test_a_queue_weighs_at_least_as_much_as_the_same_number_running():
    # Queued requests are direct evidence of latency damage; running ones are
    # merely evidence of work. Placement must prefer the replica that is busy
    # over the replica that is backed up.
    assert load_score(m(waiting=8)) >= load_score(m(running=8))


def test_a_full_kv_cache_alone_is_enough_to_evict_a_sticky_group():
    # The eviction threshold has to be reachable by KV pressure on its own,
    # otherwise a replica that is about to preempt keeps its sticky work.
    assert load_score(m(kv=1.0, running=8)) >= STICKY_EVICT_SCORE
    assert load_score(m(kv=0.3, running=2)) < STICKY_EVICT_SCORE


# -- headroom ---------------------------------------------------------------


def test_headroom_is_the_sum_of_per_replica_slack():
    assert headroom({A: 4, B: 2}, {A: 1, B: 2}) == 3


def test_headroom_never_goes_negative_on_an_oversubscribed_replica():
    # A target that just halved leaves a replica over its allowance; that debt
    # is worked off by attrition, and must never be borrowed against elsewhere.
    assert headroom({A: 1, B: 4}, {A: 5, B: 0}) == 4


def test_headroom_counts_only_the_replicas_it_is_allowed_to_place_on():
    assert headroom({A: 4, B: 4}, {A: 0, B: 0}, allowed=[A]) == 4


# -- choose_replica ---------------------------------------------------------


def test_the_least_loaded_replica_wins():
    metrics = {A: m(running=8, kv=0.6), B: m(running=1, kv=0.05)}
    sticky: dict[int, str] = {}
    assert choose_replica(0, metrics, {A: 4, B: 4}, {A: 0, B: 0}, sticky) == B
    assert sticky == {0: B}


def test_a_replica_without_headroom_is_not_a_candidate_however_idle_it_looks():
    metrics = {A: m(kv=0.01), B: m(running=4, kv=0.5)}
    chosen = choose_replica(0, metrics, {A: 2, B: 8}, {A: 2, B: 1}, {})
    assert chosen == B, "A is at its AIMD target: idleness does not buy an extra slot"


def test_a_dead_replica_is_never_chosen():
    metrics = {A: None, B: m(running=8, kv=0.7)}
    assert choose_replica(0, metrics, {A: 8, B: 8}, {A: 0, B: 0}, {}) == B


def test_no_candidate_at_all_is_none_rather_than_an_arbitrary_replica():
    metrics = {A: None, B: m()}
    assert choose_replica(0, metrics, {A: 4, B: 1}, {A: 0, B: 1}, {}) is None


def test_a_prefix_group_sticks_to_its_replica_across_calls():
    sticky: dict[int, str] = {}
    metrics = {A: m(running=1, kv=0.10), B: m(running=2, kv=0.12)}
    first = choose_replica(7, metrics, {A: 4, B: 4}, {A: 0, B: 0}, sticky)
    assert first == A

    # B is now marginally the emptier engine, but not by enough to be worth
    # throwing away A's warm prefix cache for this group.
    metrics = {A: m(running=2, kv=0.20), B: m(running=1, kv=0.10)}
    assert choose_replica(7, metrics, {A: 4, B: 4}, {A: 1, B: 0}, sticky) == A


def test_two_prefix_groups_spread_over_the_fleet():
    sticky: dict[int, str] = {}
    metrics = {A: m(), B: m()}
    first = choose_replica(0, metrics, {A: 4, B: 4}, {A: 0, B: 0}, sticky)
    second = choose_replica(1, metrics, {A: 4, B: 4}, {A: 1, B: 0}, sticky)
    assert {first, second} == {A, B}, "a second group goes where there is room"


def test_stickiness_breaks_once_the_sticky_replica_is_loaded_enough():
    sticky = {7: A}
    metrics = {A: m(running=24, waiting=8, kv=0.95), B: m(running=1, kv=0.05)}
    assert load_score(metrics[A]) >= STICKY_EVICT_SCORE
    assert choose_replica(7, metrics, {A: 4, B: 4}, {A: 0, B: 0}, sticky) == B
    assert sticky[7] == B, "eviction re-homes the group, it does not just skip it once"


def test_stickiness_breaks_when_the_sticky_replica_runs_out_of_headroom():
    sticky = {7: A}
    metrics = {A: m(kv=0.05), B: m(kv=0.30)}
    assert choose_replica(7, metrics, {A: 2, B: 4}, {A: 2, B: 0}, sticky) == B
    assert sticky[7] == B


def test_stickiness_breaks_when_the_sticky_replica_dies():
    sticky = {7: A}
    metrics = {A: None, B: m(kv=0.30)}
    assert choose_replica(7, metrics, {A: 4, B: 4}, {A: 0, B: 0}, sticky) == B
    assert sticky[7] == B


def test_a_sticky_entry_for_a_replica_no_longer_in_the_fleet_is_replaced():
    sticky = {7: C}  # C was decommissioned between runs
    metrics = {A: m(), B: m(running=4)}
    assert choose_replica(7, metrics, {A: 4, B: 4}, {A: 0, B: 0}, sticky) == A
    assert sticky[7] == A


def test_placements_within_one_tick_spread_as_in_flight_counts_rise():
    # Metrics are one scrape old for the whole tick, so without an in-flight
    # term every item claimed this tick would pile onto the same replica.
    sticky: dict[int, str] = {}
    metrics = {A: m(), B: m()}
    inflight = {A: 0, B: 0}
    placed = []
    for group in range(4):
        chosen = choose_replica(group, metrics, {A: 4, B: 4}, inflight, sticky)
        placed.append(chosen)
        inflight[chosen] += 1
    assert placed.count(A) == 2 and placed.count(B) == 2


def test_the_in_flight_term_is_a_tiebreaker_not_the_policy():
    # A is three-quarters full but genuinely idle; B is empty but hot. The
    # engine's own load still decides.
    metrics = {A: m(kv=0.02), B: m(running=16, waiting=4, kv=0.7)}
    assert choose_replica(0, metrics, {A: 4, B: 4}, {A: 3, B: 0}, {}) == A


def test_a_tie_is_broken_by_url_so_placement_is_reproducible():
    metrics = {B: m(kv=0.2), A: m(kv=0.2), C: m(kv=0.2)}
    targets = {A: 4, B: 4, C: 4}
    inflight = {A: 0, B: 0, C: 0}
    assert choose_replica(0, metrics, targets, inflight, {}) == A
    # Insertion order of the metrics dict must not change the answer.
    reordered = {C: m(kv=0.2), B: m(kv=0.2), A: m(kv=0.2)}
    assert choose_replica(0, reordered, targets, inflight, {}) == A


def test_the_same_inputs_always_give_the_same_answer():
    metrics = {A: m(running=3, kv=0.4), B: m(running=3, waiting=1, kv=0.4), C: m(kv=0.9)}
    targets = {A: 4, B: 4, C: 4}
    inflight = {A: 1, B: 1, C: 1}
    answers = {choose_replica(3, metrics, targets, inflight, {}) for _ in range(50)}
    assert len(answers) == 1


def test_allowed_pins_placement_to_a_subset_of_the_fleet():
    # The `pinned` control condition: every batch item on replica 0, whatever
    # the load says, so the fleet's placement policy has a baseline to beat.
    metrics = {A: m(running=16, kv=0.8), B: m()}
    sticky: dict[int, str] = {}
    assert choose_replica(0, metrics, {A: 4, B: 4}, {A: 0, B: 0}, sticky, allowed=[A]) == A
    assert sticky == {0: A}


def test_allowed_still_respects_headroom_and_liveness():
    metrics = {A: m(), B: m()}
    assert choose_replica(0, metrics, {A: 2, B: 4}, {A: 2, B: 0}, {}, allowed=[A]) is None
    assert choose_replica(0, {A: None, B: m()}, {A: 4, B: 4}, {A: 0, B: 0}, {}, allowed=[A]) is None
