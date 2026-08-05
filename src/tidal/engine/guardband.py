"""KV guardband tracker (spec §5.3) — Borg-style reclamation headroom.

``H`` is a clamped EWMA of *online* KV-demand bursts, expressed as a fraction of
the block pool: keep roughly enough free blocks that a p99 online burst can be
admitted without evicting anybody. Two thresholds hang off it, and their
ordering is the whole point::

    admit_ok(kv)     <=>  kv <= 1 - H          batch admission stops first
    evict_needed(kv) <=>  kv >  1 - H/2        eviction only under further growth

Because ``0 < H < 1`` we always have ``1 - H < 1 - H/2``: as usage climbs, batch
admission is cut off at the *lower* watermark, and only if online keeps growing
into the reserve does proactive eviction of batch victims kick in. Between the
two is a hold band where nothing is admitted and nothing is evicted — that gap
(width ``H/2``) is the hysteresis that stops admit/evict flapping.

Deterministic and clock-free: the EWMA advances per observation, not per second.
"""

from __future__ import annotations

__all__ = ["KvGuardband"]


class KvGuardband:
    """Clamped EWMA of online KV demand, with admit/evict watermarks.

    Args:
        h_min: floor on the guardband, as a fraction of the block pool.
        h_max: ceiling on the guardband.
        alpha: EWMA weight on the newest observation, in (0, 1].

    The EWMA is seeded at ``h_max`` so a cold scheduler is maximally protective
    of online traffic; it decays to the observed demand within a few steps.
    """

    def __init__(self, h_min: float, h_max: float, alpha: float = 0.3) -> None:
        h_min, h_max, alpha = float(h_min), float(h_max), float(alpha)
        if not 0.0 < h_min <= h_max < 1.0:
            raise ValueError(f"require 0 < h_min <= h_max < 1, got h_min={h_min}, h_max={h_max}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.h_min = h_min
        self.h_max = h_max
        self.alpha = alpha
        self._ewma = h_max
        # Hysteresis invariant, checked once at construction: for every H the
        # tracker can report, admission stops strictly below the eviction
        # trigger, and both watermarks stay inside (0, 1).
        for h in (h_min, h_max):
            assert 0.0 < 1.0 - h < 1.0 - h / 2.0 < 1.0, f"bad guardband band at h={h}"

    def observe_online_demand(self, blocks_allocated: int, pool_blocks: int) -> None:
        """Fold one step of online KV demand into the EWMA.

        ``blocks_allocated`` is what online requests took this step; ``pool_blocks``
        the total KV block pool. A non-positive pool is a no-op (nothing to
        normalise against).
        """
        if pool_blocks <= 0:
            return
        demand = float(blocks_allocated) / float(pool_blocks)
        demand = min(1.0, max(0.0, demand))
        self._ewma = self.alpha * demand + (1.0 - self.alpha) * self._ewma

    def h(self) -> float:
        """Current guardband: the EWMA clamped to ``[h_min, h_max]``."""
        return min(self.h_max, max(self.h_min, self._ewma))

    def admit_ok(self, kv_usage: float) -> bool:
        """True while batch admission is allowed: ``kv_usage <= 1 - H``."""
        return float(kv_usage) <= 1.0 - self.h()

    def evict_needed(self, kv_usage: float) -> bool:
        """True once batch should be proactively evicted: ``kv_usage > 1 - H/2``."""
        return float(kv_usage) > 1.0 - self.h() / 2.0
