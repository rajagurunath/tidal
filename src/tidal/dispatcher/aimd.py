"""AIMD controller for the number of in-flight batch items (spec §7).

Additive-increase / multiplicative-decrease over the *target* in-flight batch
count, driven purely by engine metrics:

* halve when online traffic is queueing or KV usage is above the high
  watermark — a load spike collapses batch concurrency at the gateway within
  one poll while the engine preempts already-running batch tokens;
* +1 when KV usage is below the low watermark (``kv_low << kv_high`` gives the
  hysteresis that keeps the loop from thrashing);
* hold in between; always inside ``[floor, max_inflight]``.

The floor comes from the laxity escalation path (``set_floor``): once a batch
crosses ``cfg.floor_urgency`` it is guaranteed a minimum concurrency so a busy
online tenant cannot starve it past its 24h SLA.

No clock, no I/O: ``update`` is a pure function of (state, metrics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tidal.config import TidalConfig

__all__ = ["AimdController"]


class AimdController:
    """Tracks the allowed in-flight batch-item count. Starts at 0 (spec §7)."""

    __slots__ = ("_floor", "_target", "cfg")

    def __init__(self, cfg: TidalConfig, *, target: int = 0, floor: int = 0) -> None:
        if not 0.0 <= cfg.kv_low <= cfg.kv_high:
            raise ValueError("require 0 <= kv_low <= kv_high")
        if cfg.max_inflight < 0:
            raise ValueError("max_inflight must be >= 0")
        self.cfg = cfg
        self._floor = self._clamp_floor(floor)
        self._target = self._clamp(target)

    # -- state ------------------------------------------------------------
    @property
    def target(self) -> int:
        """Allowed in-flight batch items right now."""
        return self._target

    @property
    def floor(self) -> int:
        """SLA-driven minimum concurrency (0 when no batch is at risk)."""
        return self._floor

    def set_floor(self, n: int) -> None:
        """Set the deadline floor; raising it takes effect immediately.

        Lowering the floor never drops the current target — the AIMD loop walks
        it back down on its own so a released floor cannot cause a step change.
        """
        self._floor = self._clamp_floor(n)
        if self._target < self._floor:
            self._target = self._floor

    def reset(self, target: int = 0) -> None:
        """Force the target, bypassing the floor (circuit-open on ``EngineDown``)."""
        self._target = max(0, min(self.cfg.max_inflight, int(target)))

    # -- control law ------------------------------------------------------
    def update(self, kv_usage: float, waiting: int, inflight: int) -> int:
        """One control step from a metrics sample; returns the new target.

        ``waiting`` is vLLM's aggregate ``num_requests_waiting`` — it is not
        split by priority, so we discount our own in-flight batch items to get a
        conservative lower bound on *online* queueing (spec §7).
        """
        online_waiting_lb = max(0, int(waiting) - int(inflight))
        cfg = self.cfg
        if online_waiting_lb > cfg.online_waiting_tolerance or kv_usage > cfg.kv_high:
            self._target = max(self._floor, self._target // 2)  # multiplicative decrease
        elif kv_usage < cfg.kv_low:
            self._target = min(cfg.max_inflight, self._target + 1)  # additive increase
        self._target = self._clamp(self._target)
        return self._target

    # -- helpers ----------------------------------------------------------
    def _clamp(self, n: int) -> int:
        return max(self._floor, min(self.cfg.max_inflight, int(n)))

    def _clamp_floor(self, n: int) -> int:
        return max(0, min(self.cfg.max_inflight, int(n)))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"AimdController(target={self._target}, floor={self._floor}, "
            f"max_inflight={self.cfg.max_inflight})"
        )
