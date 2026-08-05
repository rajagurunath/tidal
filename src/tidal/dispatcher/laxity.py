"""Laxity-driven continuous escalation (Least-Laxity-First), spec §7.

Pure math only — every function here is deterministic and every stateful object
takes its clock by injection, so the dispatcher's escalation behaviour is
testable without sleeping, without a wall clock and without an engine::

    laxity(b)   = (b.expires_at - now) - remaining(b) / observed_rate(b)
    urgency(b)  = clamp(1 - laxity(b) / SLA_WINDOW, 0, 1)
    priority(b) = round(P_MAX - urgency * (P_MAX - P_MIN))

On-track batches sit at ``batch_priority_max`` (100) and cause zero
interference; batches whose projected drain time eats into their remaining
window climb continuously toward ``batch_priority_min`` (1, still below
online's 0 — unless ``sla_strict``, where urgency exactly 1.0 reaches 0).
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tidal.config import TidalConfig

__all__ = [
    "RATE_EPSILON",
    "ObservedRate",
    "TokenBucket",
    "laxity_seconds",
    "priority_for",
    "urgency",
]

#: Smallest completion rate we will ever divide by (items/sec). A batch that has
#: completed nothing yet gets an astronomically long projected drain time, hence
#: negative laxity, hence urgency 1.0 — fail-safe in the escalate direction.
RATE_EPSILON = 1e-6

Instant = datetime | float | int


def _seconds_between(later: Instant, earlier: Instant) -> float:
    """``later - earlier`` in seconds; accepts datetimes or epoch floats."""
    if isinstance(later, datetime) and isinstance(earlier, datetime):
        return (later - earlier).total_seconds()
    return _epoch(later) - _epoch(earlier)


def _epoch(value: Instant) -> float:
    return value.timestamp() if isinstance(value, datetime) else float(value)


def laxity_seconds(
    expires_at: Instant,
    now: Instant,
    remaining_items: int,
    rate_items_per_s: float,
) -> float:
    """Slack left after the projected drain time, in seconds.

    ``(expires_at - now) - remaining_items / rate``. Negative laxity means the
    batch is projected to miss its completion window at the current rate. The
    rate is floored at :data:`RATE_EPSILON` so a stalled batch never divides by
    zero (it simply becomes maximally urgent).
    """
    slack = _seconds_between(expires_at, now)
    if remaining_items <= 0:
        return slack
    rate = rate_items_per_s if rate_items_per_s > RATE_EPSILON else RATE_EPSILON
    return slack - remaining_items / rate


def urgency(laxity_s: float, window_s: float) -> float:
    """``clamp(1 - laxity / window, 0, 1)``.

    0.0 when the batch has a full SLA window of slack, 1.0 the moment its
    projected finish time reaches its deadline. A degenerate (non-positive)
    window is treated as "already due": urgent iff laxity has run out.
    """
    if window_s <= 0:
        return 1.0 if laxity_s <= 0 else 0.0
    if math.isnan(laxity_s):
        return 1.0
    return min(1.0, max(0.0, 1.0 - laxity_s / window_s))


def priority_for(urgency_value: float, cfg: TidalConfig) -> int:
    """Map urgency to a vLLM priority (lower number = scheduled sooner).

    Linear from ``batch_priority_max`` at urgency 0 to ``batch_priority_min`` at
    urgency 1. With ``cfg.sla_strict`` a batch at urgency *exactly* 1.0 is
    allowed to reach 0 — parity with online traffic — and nothing below 1.0 is.
    Rounding is half-up (not Python's banker's rounding) so band edges are
    predictable.
    """
    u = min(1.0, max(0.0, urgency_value))
    p_max, p_min = cfg.batch_priority_max, cfg.batch_priority_min
    if cfg.sla_strict and u >= 1.0:
        return 0
    span = p_max - p_min
    return min(p_max, max(p_min, math.floor(p_max - u * span + 0.5)))


class ObservedRate:
    """Sliding-window completion rate (items/sec) with an epsilon floor.

    The clock is injected (``now``), so tests advance time by hand and no code
    path in the dispatcher's math calls :func:`time.time`. Samples older than
    ``window_s`` are evicted; the divisor is the *observed* span (capped at the
    window, floored at ``min_span_s``) so a young batch reports a sane rate
    instead of one inflated by a sub-second burst or deflated by a window that
    has not filled yet.
    """

    __slots__ = ("_events", "_now", "_origin", "_total", "epsilon", "min_span_s", "window_s")

    def __init__(
        self,
        window_s: float = 60.0,
        *,
        now: Callable[[], float] = time.monotonic,
        epsilon: float = RATE_EPSILON,
        min_span_s: float = 1.0,
    ) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = float(window_s)
        self.epsilon = float(epsilon)
        self.min_span_s = float(min_span_s)
        self._now = now
        self._events: deque[tuple[float, int]] = deque()
        self._origin: float | None = None
        self._total = 0

    def record(self, n: int = 1, *, now: float | None = None) -> None:
        """Record ``n`` completions at ``now`` (defaults to the injected clock)."""
        if n <= 0:
            return
        ts = self._now() if now is None else float(now)
        if self._origin is None:
            self._origin = ts
        self._events.append((ts, int(n)))
        self._total += int(n)

    def rate(self, *, now: float | None = None) -> float:
        """Completions per second over the window, never below ``epsilon``."""
        ts = self._now() if now is None else float(now)
        self._evict(ts)
        if self._total <= 0 or self._origin is None:
            return self.epsilon
        span = max(min(self.window_s, ts - self._origin), self.min_span_s)
        return max(self._total / span, self.epsilon)

    def _evict(self, ts: float) -> None:
        cutoff = ts - self.window_s
        while self._events and self._events[0][0] <= cutoff:
            _, n = self._events.popleft()
            self._total -= n
        if not self._events:
            self._origin = None
        elif self._origin is None or self._origin < self._events[0][0]:
            self._origin = self._events[0][0]

    def __len__(self) -> int:
        return self._total


class TokenBucket:
    """Minimum-rate guarantee for batches past ``cfg.floor_urgency`` (spec §7).

    Deterministic: the caller passes the timestamp to :meth:`try_take`, so the
    dispatcher owns the clock and tests own time entirely.
    """

    __slots__ = ("_last", "_tokens", "burst", "rate_items_per_s")

    def __init__(
        self,
        rate_items_per_s: float,
        burst: int = 1,
        *,
        initial_tokens: float | None = None,
    ) -> None:
        if burst <= 0:
            raise ValueError("burst must be >= 1")
        self.rate_items_per_s = max(0.0, float(rate_items_per_s))
        self.burst = int(burst)
        self._tokens = float(self.burst if initial_tokens is None else initial_tokens)
        self._last: float | None = None

    @classmethod
    def from_config(cls, cfg: TidalConfig, burst: int = 1) -> TokenBucket:
        return cls(cfg.floor_items_per_s, burst)

    @property
    def tokens(self) -> float:
        return self._tokens

    def try_take(self, now: float, n: float = 1.0) -> bool:
        """Refill for the elapsed time, then take ``n`` tokens if available."""
        self._refill(float(now))
        if self._tokens + 1e-9 >= n:
            self._tokens = max(0.0, self._tokens - n)
            return True
        return False

    def _refill(self, now: float) -> None:
        if self._last is None:
            self._last = now
            return
        elapsed = now - self._last
        if elapsed <= 0:                      # clock went backwards: mint nothing
            return
        self._last = now
        self._tokens = min(float(self.burst), self._tokens + elapsed * self.rate_items_per_s)
