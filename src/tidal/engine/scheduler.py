"""TidalScheduler — work-conserving in-engine co-scheduler (spec §5, plan B3).

A `scheduler_cls` plugin for vLLM V1. It does **not** reimplement upstream's
600-line ``schedule()``; it gates *around* it:

    schedule()  ==  pre-gate  ->  super().schedule()  ->  post-telemetry

**Pre-gate.** Requests with ``priority >= 10`` are *batch* (best-effort);
``0..9`` are *online* (guaranteed). Each step we compute a token cap ``X`` for
batch work from the self-calibrating latency model (:mod:`tidal.engine.latency_model`)
and hide every waiting batch request that does not fit under it, so the stock
scheduler simply never sees them. Held requests are removed from the waiting
queue(s) and restored in a ``finally`` block — exception-safe, and re-added
through ``add_request`` so the ``(priority, arrival_time)`` heap order is exactly
what it would have been.

**Post.** Attribute the measured inter-``schedule()`` wall time to the *previous*
step's ``(P, C)`` and feed the latency model; record :class:`TickStats`.

Deviations from the spec, deliberate and documented for the paper:

1. *Chunk granularity, not token granularity.* ``X`` cannot be enforced inside
   ``super().schedule()``'s inner loop without copying it. We instead bound how
   many batch requests are **visible** each step, estimating each at one decode
   token or one prefill chunk. Coarser than per-token, faithful in aggregate,
   and robust to upstream drift.
2. *Running batch requests are not hidden.* Only the waiting queues are gated;
   a batch request already resident keeps decoding (1 token/step) unless the
   guardband evicts it. Its cost is counted against ``X`` so the cap still
   accounts for it.
3. *Step time* is the monotonic delta between consecutive ``schedule()`` calls,
   attributed to the previous step. Exact on a synchronous CPU/GPU loop;
   under async scheduling the attribution smears by one step (v1 pins
   ``--no-async-scheduling``).
4. *In-engine laxity aging is out of v1* — the engine only sees a priority int;
   deadlines live in the gateway, which applies laxity at submission time.

Robustness contract: **nothing here may break serving**. Every pre/post hook is
wrapped; on failure we log (throttled to once per minute) and fall through to
stock behaviour. Every upstream attribute is read through ``getattr`` with a
safe default, and a startup capability probe logs exactly what was found.

The module is importable without vLLM installed (``import tidal`` in an
engine-free environment must never explode); construction raises then.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tidal.config import TidalConfig
from tidal.engine.guardband import KvGuardband
from tidal.engine.latency_model import NotReadyError, StepLatencyModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.request_queue import RequestQueue
    from vllm.v1.request import Request

__all__ = ["BATCH_PRIORITY_THRESHOLD", "TickStats", "TidalScheduler"]

logger = logging.getLogger("tidal.engine.scheduler")

#: Gateway SLA ladder emits bands 100/50/10/1 for batch; online is 0..9.
BATCH_PRIORITY_THRESHOLD = 10
#: Throttle for the "something in the hook blew up" log.
ERROR_LOG_INTERVAL_S = 60.0
#: Cadence of the compact INFO telemetry line.
LOG_EVERY_N_STEPS = 100

# vLLM lives in the *host* environment (the engine venv), not in tidal's own
# dependency set. Import it lazily-at-module-level so that `import tidal` in a
# vLLM-free venv works; the failure is re-raised at construction time with a
# message that says what to do about it.
try:
    from vllm.v1.core.sched.request_queue import SchedulingPolicy
    from vllm.v1.core.sched.scheduler import Scheduler as _UpstreamScheduler
    from vllm.v1.request import RequestStatus

    _VLLM_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only in the tidal venv
    _UpstreamScheduler = object  # type: ignore[assignment,misc]
    SchedulingPolicy = None  # type: ignore[assignment]
    RequestStatus = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = exc


@dataclass
class TickStats:
    """Telemetry for one scheduling step (spec §6). Last one lives on
    ``TidalScheduler.tidal_stats``; the eval harness scrapes it."""

    step: int = 0
    #: Tokens scheduled this step for requests with ``priority < 10``.
    online_tokens: int = 0
    #: Tokens scheduled this step for requests with ``priority >= 10``.
    batch_tokens: int = 0
    #: Waiting batch requests hidden from the stock scheduler this step.
    held_count: int = 0
    #: Batch tokens we *permitted* but nobody filled — ``tidal:starved_slack``.
    slack_unfilled: int = 0
    #: The batch token cap in force. ``None`` == offline mode (no cap).
    capped_x: int | None = None
    #: KV block-pool occupancy in [0, 1] at pre-gate time.
    kv_usage: float = 0.0
    #: Whether T̂ had converged (else the cold-start static cap was used).
    model_ready: bool = False
    #: Rolling MAPE of T̂; NaN before the first fit.
    mape: float = float("nan")
    #: Batch requests proactively evicted by the guardband this step.
    evicted: int = 0
    #: Guardband H currently in force.
    guardband_h: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "online_tokens": self.online_tokens,
            "batch_tokens": self.batch_tokens,
            "held": self.held_count,
            "starved_slack": self.slack_unfilled,
            "x": self.capped_x,
            "kv": round(self.kv_usage, 4),
            "model_ready": self.model_ready,
            "mape": self.mape,
            "evicted": self.evicted,
            "h": round(self.guardband_h, 4),
        }


@dataclass
class _Capabilities:
    """What the upstream scheduler actually exposes, probed once at startup.

    Every one of these is optional: a missing capability degrades one mechanism,
    it never fails a step.
    """

    kv_usage: bool = False
    preempt: bool = False
    estimate_cached_tokens: bool = False
    prefill_stats: bool = False
    pool_blocks: int = 0
    notes: list[str] = field(default_factory=list)


class TidalScheduler(_UpstreamScheduler):  # type: ignore[misc,valid-type]
    """vLLM V1 scheduler that fills each step's slack with batch work.

    Constructed by vLLM itself via ``--scheduler-cls``; the signature is
    upstream's and is passed straight through, so the subclass survives
    upstream adding constructor arguments.

    Requires ``--scheduling-policy priority``: the whole design is a partition
    of the ``(priority, arrival_time)`` waiting heap into online and batch, and
    under FCFS the priority field is inert.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _VLLM_IMPORT_ERROR is not None:
            raise ImportError(
                "TidalScheduler requires vLLM in the running environment "
                "(it subclasses vllm.v1.core.sched.scheduler.Scheduler). Install "
                "vLLM in the engine venv and launch with "
                "`--scheduler-cls tidal.engine.scheduler.TidalScheduler`."
            ) from _VLLM_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        cfg = TidalConfig.from_env()
        self.tidal_config = cfg

        policy = getattr(self, "policy", None)
        if policy is not SchedulingPolicy.PRIORITY:
            raise ValueError(
                "TidalScheduler requires the priority scheduling policy "
                f"(got {getattr(policy, 'value', policy)!r}). Launch vLLM with "
                "`--scheduling-policy priority`: Tidal classifies work by the "
                f"request priority field (>= {BATCH_PRIORITY_THRESHOLD} is batch) "
                "and that field is ignored under FCFS."
            )

        self.latency_model = StepLatencyModel(
            window=int(cfg.fit_window_steps), r2_gate=float(cfg.fit_r2_gate)
        )
        self.guardband = KvGuardband(
            h_min=float(cfg.kv_guardband_min), h_max=float(cfg.kv_guardband_max)
        )
        self.tidal_stats = TickStats()

        #: Waiting batch requests hidden from the current ``super().schedule()``,
        #: as ``(queue_they_came_from, request)``. Always empty between steps.
        self._tidal_held: list[tuple[RequestQueue, Request]] = []
        self._tidal_step = 0
        self._last_schedule_ts: float | None = None
        #: ``(P, C)`` of the step whose latency the next tick will measure.
        self._pending_sample: tuple[int, int] | None = None
        self._last_error_log_ts = 0.0

        self._caps = self._probe_capabilities()
        logger.info(
            "TidalScheduler active: tau=%d tbt_slo=%.1fms guard=%.2f "
            "cold_start_frac=%.2f kv_guardband=[%.2f, %.2f] capabilities=%s",
            self._tau(),
            cfg.tbt_slo_ms,
            cfg.interference_guard,
            cfg.cold_start_batch_frac,
            cfg.kv_guardband_min,
            cfg.kv_guardband_max,
            self._caps.notes,
        )

    # -- classification ----------------------------------------------------

    @staticmethod
    def classify(request: Request) -> bool:
        """True if ``request`` is *batch* (best-effort) work.

        The gateway's SLA ladder emits priority bands 100/50/10/1; online
        traffic is 0. Band 1 (maximum escalation) still counts as batch for
        capping purposes but sorts ahead of every other batch item in the stock
        heap — intended.
        """
        return int(getattr(request, "priority", 0) or 0) >= BATCH_PRIORITY_THRESHOLD

    # -- the one override --------------------------------------------------

    def schedule(self, *args: Any, **kwargs: Any) -> SchedulerOutput:
        """Pre-gate → stock ``schedule()`` → post-telemetry.

        Signature is upstream's, forwarded verbatim (currently
        ``schedule(throttle_prefills: bool = False)``).
        """
        tick_ts = time.monotonic()
        self._tidal_step += 1

        try:
            self._observe_previous_step(tick_ts)
        except Exception:
            self._log_hook_failure("tidal: latency-model observation failed")

        cap: int | None = None
        try:
            cap = self._pre_gate()
        except Exception:
            self._log_hook_failure("tidal: pre-gate failed, scheduling stock this step")
            self._restore_held()

        try:
            output = super().schedule(*args, **kwargs)
        finally:
            # Non-negotiable: a hidden request must never be lost, not even if
            # the stock scheduler raises.
            self._restore_held()

        try:
            self._post(output, cap)
        except Exception:
            self._log_hook_failure("tidal: post-telemetry failed")
        return output

    # -- pre-gate ----------------------------------------------------------

    def _pre_gate(self) -> int | None:
        """Decide the batch token cap ``X`` and hide what does not fit.

        Returns the cap in force (``None`` == offline mode, no cap), which the
        post hook needs to compute unfilled slack.
        """
        cfg = self.tidal_config
        kv_usage = self._kv_usage()
        tau = self._tau()

        running = list(getattr(self, "running", ()))
        online_running = [r for r in running if not self.classify(r)]
        batch_running = [r for r in running if self.classify(r)]

        # 1. Proactive eviction (spec §5.3): under online growth into the
        #    reserve, drop the batch victim with the least unrecoverable work
        #    *before* the allocator has to. Stock inline preemption remains the
        #    backstop; its victim rule is `max(priority, arrival_time)`, which
        #    ignores lost work entirely.
        evicted = 0
        if batch_running and self.guardband.evict_needed(kv_usage):
            evicted = int(self._evict_cheapest_batch(batch_running))
            if evicted:
                running = list(getattr(self, "running", ()))
                batch_running = [r for r in running if self.classify(r)]

        # 2. Partition the waiting side. `waiting` is the live heap; a request
        #    just evicted above has already been prepended into it, so it is
        #    considered for holding here too (and, at this KV usage, held).
        queues = self._waiting_queues()
        waiting_online: list[Request] = []
        waiting_batch: list[tuple[RequestQueue, Request]] = []
        for queue in queues:
            for request in list(queue):
                if self.classify(request):
                    waiting_batch.append((queue, request))
                else:
                    waiting_online.append(request)

        # 3. Token cap X.
        num_online_resident = len(online_running) + len(waiting_online)
        p_on = self._estimate_online_tokens(online_running, waiting_online)
        slack = max(0, tau - p_on)
        model_ready = False
        cap: int | None
        if num_online_resident == 0:
            # OFFLINE MODE (spec §5.1): nothing to protect, fill tau entirely.
            cap = None
        else:
            c_total = self._resident_context(running)
            budget_ms = float(cfg.tbt_slo_ms) * (1.0 - float(cfg.interference_guard))
            try:
                cap = int(self.latency_model.max_batch_tokens(p_on, c_total, budget_ms))
                model_ready = True
            except NotReadyError:
                # Cold start / poor fit: conservative static cap, never trust an
                # unconverged model.
                cap = int(float(cfg.cold_start_batch_frac) * tau)
            cap = max(0, min(cap, slack))

        # 4. Hide the batch requests that do not fit under X, or all of them if
        #    the guardband says KV headroom is exhausted.
        admit_ok = self._admit_ok(kv_usage)
        estimate = sum(self._running_step_tokens(r) for r in batch_running)
        hold: dict[int, list[Request]] = {}
        hold_queues: dict[int, RequestQueue] = {}
        held_count = 0
        for queue, request in waiting_batch:
            cost = self._waiting_step_tokens(request)
            if admit_ok and (cap is None or estimate + cost <= cap):
                estimate += cost
                continue
            hold.setdefault(id(queue), []).append(request)
            hold_queues[id(queue)] = queue
            held_count += 1

        for key, requests in hold.items():
            queue = hold_queues[key]
            queue.remove_requests(requests)
            self._tidal_held.extend((queue, r) for r in requests)

        stats = self.tidal_stats
        stats.step = self._tidal_step
        stats.held_count = held_count
        stats.capped_x = cap
        stats.kv_usage = kv_usage
        stats.model_ready = model_ready
        stats.evicted = evicted
        stats.guardband_h = self.guardband.h()
        return cap

    def _restore_held(self) -> None:
        """Put every hidden request back. Idempotent; must never raise."""
        if not self._tidal_held:
            return
        held, self._tidal_held = self._tidal_held, []
        for queue, request in held:
            try:
                # `add_request` on a PriorityRequestQueue is a heappush, so
                # (priority, arrival_time) ordering is restored exactly.
                queue.add_request(request)
            except Exception:  # pragma: no cover - defensive
                self._log_hook_failure("tidal: failed to restore a held request")

    def _evict_cheapest_batch(self, batch_running: list[Request]) -> int:
        """Preempt the running batch request with the least unrecoverable work.

        Cost is ``num_computed_tokens - prefix-cached tokens``: recompute-only
        preemption plus the prefix cache make that the true loss. Returns the
        number of requests evicted (0 or 1 — one per step is enough to track
        online growth, and it keeps the pre-gate O(running)).
        """
        if not self._caps.preempt:
            return 0
        candidates = [r for r in batch_running if self._is_running_status(r)]
        if not candidates:
            return 0
        victim = min(candidates, key=self._unrecoverable_work)
        try:
            self.running.remove(victim)
        except ValueError:  # pragma: no cover - defensive
            return 0
        # Upstream contract: pop from `running` outside, then call this. It
        # frees blocks, resets num_computed_tokens and prepends to `waiting`.
        self._preempt_request(
            victim,
            time.monotonic(),
            drop_stale_output=bool(getattr(self, "requires_kv_delivery", False)),
        )
        logger.debug(
            "tidal: evicted batch request %s (lost %d tokens, kv=%.3f)",
            getattr(victim, "request_id", "?"),
            self._unrecoverable_work(victim),
            self._kv_usage(),
        )
        return 1

    def _unrecoverable_work(self, request: Request) -> int:
        """Tokens that a preemption would genuinely throw away."""
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        return max(0, computed - self._cached_tokens(request))

    def _cached_tokens(self, request: Request) -> int:
        """Prefix-cache-recoverable tokens for ``request``, best effort.

        Preferred source is ``kv_cache_manager.estimate_cached_tokens`` (counts
        this request's blocks that carry a prefix-cache hash, i.e. what a resume
        can hit). Falls back to the admission-time ``prefill_stats`` count, then
        to 0 (= treat everything as unrecoverable, the conservative reading).
        """
        if self._caps.estimate_cached_tokens:
            try:
                return int(self.kv_cache_manager.estimate_cached_tokens(request) or 0)
            except Exception:  # pragma: no cover - defensive
                pass
        stats = getattr(request, "prefill_stats", None)
        return int(getattr(stats, "num_cached_tokens", 0) or 0)

    # -- post --------------------------------------------------------------

    def _post(self, output: SchedulerOutput, cap: int | None) -> None:
        """Record this step's ledger and arm the next latency sample."""
        per_request: dict[str, int] = dict(getattr(output, "num_scheduled_tokens", {}) or {})
        total = int(getattr(output, "total_num_scheduled_tokens", 0) or 0)

        online_tokens = 0
        batch_tokens = 0
        context = 0
        online_blocks = 0
        block_size = max(1, int(getattr(self, "block_size", 1) or 1))
        for req_id, num_tokens in per_request.items():
            request = self.requests.get(req_id)
            is_batch = request is not None and self.classify(request)
            if is_batch:
                batch_tokens += num_tokens
            else:
                online_tokens += num_tokens
                online_blocks += math.ceil(num_tokens / block_size)
            if request is not None:
                # `_update_after_schedule` has already folded this step's tokens
                # into num_computed_tokens; C is the context they ran *against*.
                computed = int(getattr(request, "num_computed_tokens", 0) or 0)
                context += max(0, computed - num_tokens)

        tau = self._tau()
        permitted = tau - online_tokens if cap is None else min(cap, tau - online_tokens)
        slack_unfilled = max(0, permitted - batch_tokens)

        stats = self.tidal_stats
        stats.online_tokens = online_tokens
        stats.batch_tokens = batch_tokens
        stats.slack_unfilled = slack_unfilled
        stats.mape = self.latency_model.mape()

        self.guardband.observe_online_demand(online_blocks, self._caps.pool_blocks)
        self._pending_sample = (total, context) if total > 0 else None

        if self._tidal_step % LOG_EVERY_N_STEPS == 0:
            logger.info("tidal %s", stats.as_dict())

    def _observe_previous_step(self, tick_ts: float) -> None:
        """Feed T̂ one ``(P, C, step_ms)`` sample for the step just finished."""
        previous_ts, self._last_schedule_ts = self._last_schedule_ts, tick_ts
        sample, self._pending_sample = self._pending_sample, None
        if previous_ts is None or sample is None:
            return
        step_ms = (tick_ts - previous_ts) * 1000.0
        if step_ms <= 0.0 or not math.isfinite(step_ms):  # pragma: no cover
            return
        p, c = sample
        self.latency_model.observe(p, c, step_ms)

    # -- upstream-surface helpers -----------------------------------------

    def _probe_capabilities(self) -> _Capabilities:
        """One-shot check of every upstream attribute we depend on."""
        caps = _Capabilities()
        manager = getattr(self, "kv_cache_manager", None)

        usage = getattr(manager, "usage", None)
        caps.kv_usage = isinstance(usage, (int, float))
        caps.notes.append(f"kv_cache_manager.usage={'ok' if caps.kv_usage else 'MISSING'}")

        caps.preempt = callable(getattr(self, "_preempt_request", None))
        caps.notes.append(f"_preempt_request={'ok' if caps.preempt else 'MISSING'}")

        caps.estimate_cached_tokens = callable(getattr(manager, "estimate_cached_tokens", None))
        caps.notes.append(
            "estimate_cached_tokens="
            f"{'ok' if caps.estimate_cached_tokens else 'fallback:prefill_stats'}"
        )

        pool = getattr(getattr(self, "cache_config", None), "num_gpu_blocks", 0)
        caps.pool_blocks = int(pool or 0)
        caps.notes.append(f"num_gpu_blocks={caps.pool_blocks}")
        return caps

    def _tau(self) -> int:
        """The per-step compute budget (``max_num_batched_tokens``)."""
        return int(getattr(self, "max_num_scheduled_tokens", 0) or 0)

    def _chunk_cap(self) -> int:
        """Largest prefill chunk a single request can take in one step."""
        tau = self._tau()
        threshold = int(
            getattr(getattr(self, "scheduler_config", None), "long_prefill_token_threshold", 0) or 0
        )
        return min(tau, threshold) if threshold > 0 else tau

    def _waiting_queues(self) -> list[RequestQueue]:
        """Every queue the stock scheduler may pull from this step.

        ``skipped_waiting`` holds requests deferred by an earlier step (blocked
        status, LoRA pressure, in-flight stale output). Upstream's
        ``_select_waiting_queue_for_scheduling`` picks between it and
        ``waiting`` by heap head, so gating only ``waiting`` would leave a
        bypass for batch work that had been skipped once.
        """
        queues = []
        for name in ("waiting", "skipped_waiting"):
            queue = getattr(self, name, None)
            if queue is not None and len(queue):
                queues.append(queue)
        return queues

    def _kv_usage(self) -> float:
        if not self._caps.kv_usage:
            return 0.0
        try:
            return float(self.kv_cache_manager.usage)
        except Exception:  # pragma: no cover - defensive
            return 0.0

    def _admit_ok(self, kv_usage: float) -> bool:
        return bool(self.guardband.admit_ok(kv_usage))

    def _is_running_status(self, request: Request) -> bool:
        if RequestStatus is None:  # pragma: no cover - vLLM-free environment
            return False
        return getattr(request, "status", None) is RequestStatus.RUNNING

    def _running_step_tokens(self, request: Request) -> int:
        """Tokens a *resident* request is expected to consume next step."""
        total = int(
            getattr(request, "num_tokens_with_spec", None) or getattr(request, "num_tokens", 0) or 0
        )
        remaining = total - int(getattr(request, "num_computed_tokens", 0) or 0)
        if remaining <= 1:
            return 1  # decode: one token per step
        return max(1, min(remaining, self._chunk_cap()))

    def _waiting_step_tokens(self, request: Request) -> int:
        """Tokens a *waiting* request is expected to consume on admission."""
        remaining = int(getattr(request, "num_tokens", 0) or 0) - int(
            getattr(request, "num_computed_tokens", 0) or 0
        )
        return max(1, min(remaining, self._chunk_cap()))

    def _estimate_online_tokens(
        self, online_running: list[Request], waiting_online: list[Request]
    ) -> int:
        """``P_on``: online tokens this step, before the stock scheduler runs.

        Approximate by construction — the stock scheduler may admit fewer of the
        waiting online requests than fit here (slot budget, KV, LoRA). Over-
        estimating ``P_on`` shrinks ``X``, so the error is on the safe side for
        invariant I1 (online latency is never sacrificed to batch presence).
        """
        estimate = sum(self._running_step_tokens(r) for r in online_running)
        for request in waiting_online:
            estimate += self._waiting_step_tokens(request)
        return min(estimate, self._tau())

    def _resident_context(self, running: list[Request]) -> int:
        """``C_total``: resident KV context in tokens across running requests."""
        return sum(int(getattr(r, "num_computed_tokens", 0) or 0) for r in running)

    def _log_hook_failure(self, message: str) -> None:
        """Log a hook failure at most once per minute, with traceback."""
        now = time.monotonic()
        if now - self._last_error_log_ts < ERROR_LOG_INTERVAL_S:
            return
        self._last_error_log_ts = now
        logger.exception("%s (step %d)", message, self._tidal_step)
