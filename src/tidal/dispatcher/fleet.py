"""Placement policy for technique A across a fleet of engines.

Everything here is a pure function of injected state — engine metrics, per-
replica AIMD targets, per-replica in-flight counts, and the sticky map — so a
run's placement can be replayed exactly from its tick log, and every property
below is unit-testable without an engine.

The policy, in one tick:

1. **Headroom is a hard filter.** A replica is a candidate only if it answered
   the last scrape *and* has fewer items in flight than its own AIMD target.
   Each replica's target is that engine's statement about how much batch work
   its online tenant can currently tolerate; borrowing against it from another
   replica would defeat the entire control loop.
2. **Prefix groups are sticky.** Batch pools share a long preamble, which is
   what the store's ``prefix_group`` ordering exists to exploit. Sending a
   group to a different engine every tick throws away a warm prefix cache, so a
   group stays where it was — *unless* that replica's load has crossed
   :data:`STICKY_EVICT_SCORE`, at which point the queueing costs more than the
   cache hit is worth.
3. **Otherwise, least loaded wins**, with in-flight placements from this same
   tick counted as pressure (see :data:`W_FILL`) and the URL as the final
   tie-break, so the choice is total and deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from tidal.dispatcher.vllm_client import EngineMetrics

__all__ = [
    "DEFAULT_SCORE",
    "RUNNING_NORM",
    "STICKY_EVICT_SCORE",
    "W_FILL",
    "W_KV",
    "W_RUNNING",
    "W_WAITING",
    "LoadScore",
    "choose_replica",
    "headroom",
    "load_score",
]

#: Weight on KV-cache utilisation. The reference gauge: a full KV cache is what
#: makes an engine start preempting, so it is worth 1.0 on its own.
W_KV = 1.0
#: Weight on running requests. Work in progress is evidence of *use*, not yet
#: of damage, so it counts for half of the same normalised amount of KV.
W_RUNNING = 0.5
#: Weight on queued requests. A request that is waiting is direct evidence of
#: latency damage — the thing co-serving must not cause — so it is weighted as
#: heavily as a full cache.
W_WAITING = 1.0
#: Request count treated as "fully loaded" when normalising the two counters.
#: 32 concurrent requests saturates the CPU testbed and is a sane fraction of a
#: single GPU's batch; the exact value only sets the scale of the count terms
#: relative to KV usage, which is already a fraction.
RUNNING_NORM = 32.0
#: Weight on a replica's *current* fill (in-flight ÷ target). Metrics are one
#: scrape old for the whole tick, so without this term every item claimed in a
#: tick would be placed on whichever replica was emptiest at scrape time. It is
#: deliberately small enough that it only orders replicas the engine gauges say
#: are comparable — see the "tiebreaker, not the policy" test.
W_FILL = 0.5
#: Load score above which a prefix group stops being sticky. Reachable by KV
#: pressure alone (``kv_usage`` ≈ 1.0), which is the point: a replica about to
#: preempt must be able to shed its sticky work even when nothing is queued
#: yet. A comfortable replica (KV well under the AIMD high watermark, a couple
#: of requests running) scores far below it and keeps its groups.
STICKY_EVICT_SCORE = 1.0


@dataclass(frozen=True)
class LoadScore:
    """Engine metrics → one comparable number. Lower is emptier; idle is 0.0.

    ``score = w_kv·kv_usage + w_running·(running/norm) + w_waiting·(waiting/norm)``

    Deliberately simple and unfitted: the fleet only needs an *ordering* over
    replicas that a human can read off a tick log, and any calibrated model
    would be a second, silently-drifting copy of the interference model
    technique B already owns.
    """

    w_kv: float = W_KV
    w_running: float = W_RUNNING
    w_waiting: float = W_WAITING
    running_norm: float = RUNNING_NORM

    def __call__(self, metrics: EngineMetrics) -> float:
        norm = self.running_norm if self.running_norm > 0 else 1.0
        return (
            self.w_kv * metrics.kv_usage
            + self.w_running * (metrics.running / norm)
            + self.w_waiting * (metrics.waiting / norm)
        )


#: The weights every caller gets unless it says otherwise.
DEFAULT_SCORE = LoadScore()


def load_score(metrics: EngineMetrics) -> float:
    """Load score of one replica under the default weights."""
    return DEFAULT_SCORE(metrics)


def headroom(
    targets: Mapping[str, int],
    inflight: Mapping[str, int],
    allowed: Iterable[str] | None = None,
) -> int:
    """How many more batch items the fleet may have in flight, in total.

    Per replica, and clamped at zero: a replica whose target has just halved is
    over its allowance, and that debt is worked off by attrition rather than
    borrowed against elsewhere in the fleet.
    """
    urls = list(targets) if allowed is None else [url for url in allowed if url in targets]
    return sum(max(0, int(targets[url]) - int(inflight.get(url, 0))) for url in urls)


def choose_replica(
    prefix_group: int,
    metrics: Mapping[str, EngineMetrics | None],
    targets: Mapping[str, int],
    inflight: Mapping[str, int],
    sticky: dict[int, str],
    *,
    allowed: Sequence[str] | None = None,
    score: LoadScore = DEFAULT_SCORE,
) -> str | None:
    """Pick the replica for one batch item, or ``None`` if the fleet is full.

    Args:
        prefix_group: the item's prefix bucket — the unit of stickiness.
        metrics: last scrape per replica; ``None`` means that replica is down.
        targets: per-replica AIMD target (allowed in-flight batch items).
        inflight: per-replica in-flight count, *including* items placed earlier
            in this same tick.
        sticky: ``{prefix_group: replica}``, updated in place.
        allowed: restrict placement to these replicas (the ``pinned`` control
            policy pins everything to replica 0). ``None`` means the fleet.
        score: load weights; the default is :data:`DEFAULT_SCORE`.

    Returns the chosen replica URL, or ``None`` when no replica is both alive
    and under its target — the caller must then hold the item back rather than
    oversubscribe someone.
    """
    permitted = set(metrics) if allowed is None else {url for url in allowed if url in metrics}
    scores = {
        url: score(sample)
        for url, sample in metrics.items()
        if sample is not None and url in permitted
    }
    candidates = [url for url in scores if int(inflight.get(url, 0)) < int(targets.get(url, 0))]
    if not candidates:
        return None

    held = sticky.get(prefix_group)
    if held in candidates and scores[held] < STICKY_EVICT_SCORE:
        return held

    def key(url: str) -> tuple[float, str]:
        target = max(1, int(targets.get(url, 0)))
        return (scores[url] + W_FILL * (int(inflight.get(url, 0)) / target), url)

    chosen = min(candidates, key=key)
    sticky[prefix_group] = chosen
    return chosen
