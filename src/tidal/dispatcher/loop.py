"""The dispatcher: load-aware injection of batch items into a stock vLLM.

One tick (spec §7), all of it deterministic given an injected ``now``::

    scrape /metrics  →  EWMA smooth                       (per replica)
                     →  laxity per active batch  →  priorities + SLA floor
                     →  AIMD target                       (per replica)
                     →  claim Σ(target - in-flight) items in EDF x prefix order
                     →  place each on a replica (tidal.dispatcher.fleet)
                     →  submit each as an asyncio task at its batch's priority
                     →  expiry sweep (batches past expires_at)

**Fleets.** Given a :class:`~tidal.dispatcher.vllm_client.ReplicaSet` instead of
a single client, every control quantity above becomes per-replica: one
``AimdController`` and one EWMA per engine, a claim bounded by the fleet's total
headroom, and placement delegated to the pure policy in
:mod:`tidal.dispatcher.fleet`. The circuit follows: a replica that fails its
scrape has *its* controller reset and *its* in-flight items handed back, while
the rest of the fleet keeps flowing. Only a fleet with no live replica left
behaves like the single-engine circuit. A plain ``VllmClient`` — or a
``ReplicaSet`` of one — takes exactly the same code path with one replica in it,
and produces exactly the decisions it did before fleets existed.

Online traffic never passes through here: the gateway only decides *how much*
batch work is allowed to be in flight and *at what priority*, and the engine's
priority scheduler does the rest. Two properties fall out of that split:

* a load spike halves batch concurrency at the gateway within one poll, while
  the engine preempts already-running batch tokens immediately;
* an at-risk batch escalates continuously toward ``batch_priority_min`` and is
  guaranteed a token-bucket minimum rate, which is what turns "best effort"
  into a real 24h SLA.

The dispatcher is deliberately decoupled from metering (A6) and result
assembly (A7): both arrive as plain callables (``on_success``, ``finalize``),
so this module imports neither. Hook exceptions are logged, never fatal.

**One dispatcher, by assumption.** There is no ownership lease: nothing stops a
second Tidal process from pointing at the same store and running its own tick
loop. The store keeps that from corrupting anything — the claim is a single
``UPDATE ... WHERE state='pending' ... RETURNING``, so two dispatchers can
never claim the same item; terminal item transitions are idempotent, so a
duplicate or late result cannot drift ``request_counts``; and attaching result
files is a first-writer-wins claim, so two finalizers cannot produce two output
files. What is *not* covered is duplicated work and double metering: two
dispatchers would each submit items and each meter their own successes, and the
AIMD controllers would fight for the same GPU with independent targets. Run
exactly one dispatcher per store until a lease exists.

**Cold-start rate.** ``laxity = slack - remaining/rate`` needs an observed
completion rate, and a batch that has just been created has none — taken
literally (rate = ε) *every* new batch would be maximally urgent, pinning the
whole system at priority 1 and permanently engaging the SLA floor. So the rate
falls back: per-batch → global (all batches) → and if nothing has ever
completed, the projected drain term is dropped and laxity is pure slack. New
batches therefore start on-track and escalate on evidence or as their deadline
approaches, never on ignorance.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tidal.config import TidalConfig
from tidal.dispatcher.aimd import AimdController
from tidal.dispatcher.fleet import choose_replica, headroom, load_score
from tidal.dispatcher.laxity import ObservedRate, TokenBucket, laxity_seconds, priority_for, urgency
from tidal.dispatcher.vllm_client import EngineDown, EngineMetrics, FatalUpstream, UpstreamError
from tidal.store.interfaces import (
    BatchRecord,
    BatchStatus,
    IllegalTransition,
    ItemRecord,
    ItemState,
    Repository,
)

__all__ = ["Dispatcher", "TickReport"]

log = logging.getLogger("tidal.dispatcher")

#: Batches the dispatcher still has work to do for.
ACTIVE_STATUSES = frozenset({BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS})
#: Page size for the per-tick active-batch scan.
BATCH_PAGE_SIZE = 100
#: Hard stop on that scan so a huge history can never stall a tick.
MAX_BATCH_PAGES = 100
#: Window for the per-batch completion-rate estimate (seconds).
RATE_WINDOW_S = 300.0

_TERMINAL_ITEM_STATES = frozenset(
    {ItemState.SUCCEEDED, ItemState.FAILED, ItemState.EXPIRED, ItemState.CANCELLED}
)

#: (batch, pending, inflight) for one active batch.
BatchSnapshot = tuple[BatchRecord, int, int]

#: Placement policies. ``fleet`` is the load-aware policy; ``pinned`` sends
#: every batch item to the first replica and exists as the control condition
#: the fleet policy has to beat.
PLACEMENTS = ("fleet", "pinned")

#: What a replica-local circuit records against the items it hands back. The
#: store has no requeue that does not count an attempt, so a replica-local
#: circuit spends one of the item's ``max_item_attempts`` — exactly as an
#: in-flight request against that same dying replica already would.
REPLICA_DOWN_ERROR = "replica down: requeued by the dispatcher's replica-local circuit"


@dataclass
class TickReport:
    """What one tick did — the loop's whole observable surface.

    ``escalated_priorities`` holds the priority computed for *every* active
    batch this tick (``batch_priority_max`` = not escalated); ``inflight`` is
    the in-flight item count immediately after this tick's submissions.

    ``target`` and ``inflight`` are fleet *totals*; ``per_replica`` breaks them
    down as ``{url: {"target", "inflight", "score"}}``, where ``score`` is the
    replica's load score (``None`` when it failed this tick's scrape). A
    single-engine deployment reports one entry and unchanged totals.
    """

    target: int
    submitted: int
    inflight: int
    escalated_priorities: dict[str, int] = field(default_factory=dict)
    expired_batches: list[str] = field(default_factory=list)
    circuit_open: bool = False
    per_replica: dict[str, dict] = field(default_factory=dict)


class Dispatcher:
    """Polls the engine, decides how much batch work may fly, and submits it."""

    def __init__(
        self,
        cfg: TidalConfig,
        repo: Repository,
        client,
        on_success: Callable[[ItemRecord], object] | None = None,
        finalize: Callable[[Repository, str], object] | None = None,
        *,
        placement: str = "fleet",
    ) -> None:
        if placement not in PLACEMENTS:
            raise ValueError(f"unknown placement {placement!r}; want one of {PLACEMENTS}")
        self.cfg = cfg
        self.repo = repo
        self.client = client
        self.on_success = on_success
        self.finalize = finalize
        self.placement = placement
        self._finalize_wants_repo: bool | None = None
        #: A ``ReplicaSet`` names its own engines; a plain client is one engine.
        self._is_fleet = hasattr(client, "scrape_all")
        self.replicas: list[str] = list(client.urls) if self._is_fleet else [cfg.vllm_base_url]
        #: One AIMD controller per replica — a target is a statement about one
        #: engine's online tenant and is never transferable to another.
        self.controllers: dict[str, AimdController] = {
            url: AimdController(cfg) for url in self.replicas
        }
        #: EWMA-smoothed engine metrics per replica; ``None`` until a good scrape.
        self.replica_metrics: dict[str, EngineMetrics | None] = dict.fromkeys(self.replicas)
        self._ewma: dict[str, tuple[float, float, float] | None] = dict.fromkeys(self.replicas)
        self._rates: dict[str, ObservedRate] = {}
        self._global_rate = ObservedRate(RATE_WINDOW_S)
        self._buckets: dict[str, TokenBucket] = {}
        self._inflight: dict[str, asyncio.Task] = {}
        self._item_batch: dict[str, str] = {}
        self._item_replica: dict[str, str] = {}
        self._sticky: dict[int, str] = {}
        #: ``{custom_id: replica}`` for every item ever submitted — the run
        #: bookkeeping the eval harness bins its results by. Never pruned: it is
        #: the only record of placement once an item has finished.
        self.replica_of: dict[str, str] = {}
        self._now: datetime = datetime.now(UTC)
        self._now_ts: float = self._now.timestamp()

    # -- single-engine accessors (unchanged surface) -----------------------

    @property
    def aimd(self) -> AimdController:
        """The first replica's controller — the whole controller when N=1."""
        return self.controllers[self.replicas[0]]

    @property
    def metrics(self) -> EngineMetrics | None:
        """The first replica's smoothed metrics — all of them when N=1."""
        return self.replica_metrics[self.replicas[0]]

    @property
    def inflight_replicas(self) -> dict[str, str]:
        """``{item_id: replica}`` for the items in flight right now."""
        return dict(self._item_replica)

    # -- lifecycle ---------------------------------------------------------

    async def startup(self) -> int:
        """Crash recovery: anything left INFLIGHT by a previous process is ours
        to re-run (idempotent — the engine's prefix cache makes it cheap)."""
        requeued = await asyncio.to_thread(self.repo.requeue_inflight)
        if requeued:
            log.info("startup requeued %d in-flight items", requeued)
        return requeued

    async def run(self) -> None:
        """Tick forever at ``cfg.poll_interval_s``. Never dies on a tick error."""
        await self.startup()
        while True:
            try:
                await self.tick(datetime.now(UTC))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("dispatcher tick failed")
            await asyncio.sleep(self.cfg.poll_interval_s)

    async def shutdown(self) -> None:
        """Cancel in-flight submissions and hand their items back to the store."""
        await self._cancel_tasks(list(self._inflight))
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self.repo.requeue_inflight)

    async def drain(self) -> None:
        """Wait for every in-flight submission (and its hooks) to settle."""
        while self._inflight:
            await asyncio.gather(*list(self._inflight.values()), return_exceptions=True)

    # -- the tick ----------------------------------------------------------

    async def tick(self, now: datetime) -> TickReport:
        """Run exactly one control iteration against the clock it is given."""
        self._now, self._now_ts = now, now.timestamp()

        sample = await self._scrape_all()
        live = {url: value for url, value in sample.items() if value is not None}
        if not live:
            log.warning("no live replica in %s — opening circuit", self.replicas)
            await self._open_circuit()
            return TickReport(
                target=self._total_target(),
                submitted=0,
                inflight=0,
                circuit_open=True,
                per_replica=self._per_replica_report(),
            )

        for url in sample:
            if url not in live:
                await self._open_replica_circuit(url)

        metrics: dict[str, EngineMetrics | None] = dict.fromkeys(sample)
        for url, value in live.items():
            metrics[url] = self._smooth(url, value)

        snapshot = await asyncio.to_thread(self._snapshot)
        priorities, floor = self._escalate(snapshot, now)

        # The floor is a *deadline* guarantee, so it must be in place before the
        # control law runs: AIMD's multiplicative decrease clamps to it.
        targets: dict[str, int] = {}
        for url in self.replicas:
            controller = self.controllers[url]
            smoothed = metrics.get(url)
            if smoothed is None:  # down: its circuit already reset the target
                targets[url] = controller.target
                continue
            controller.set_floor(floor)
            targets[url] = controller.update(
                smoothed.kv_usage, smoothed.waiting, self._inflight_on(url)
            )

        submitted, resolved = await self._fill(targets, metrics, snapshot, priorities, now)
        inflight = len(self._inflight)
        expired = await self._sweep_expired(snapshot, resolved, now)

        return TickReport(
            target=sum(targets.values()),
            submitted=submitted,
            inflight=inflight,
            escalated_priorities=priorities,
            expired_batches=expired,
            per_replica=self._per_replica_report(targets),
        )

    # -- stages ------------------------------------------------------------

    async def _scrape_all(self) -> dict[str, EngineMetrics | None]:
        """One scrape per replica; ``None`` for any that did not answer."""
        if self._is_fleet:
            return await self.client.scrape_all()
        url = self.replicas[0]
        try:
            return {url: await self.client.scrape()}
        except EngineDown as exc:
            log.warning("engine down (%s)", exc)
            return {url: None}

    def _smooth(self, replica: str, sample: EngineMetrics) -> EngineMetrics:
        """EWMA over one replica's raw scrape (alpha = ``cfg.ewma_alpha``).

        Smoothing is what stops a one-poll KV blip from collapsing batch
        concurrency; a genuine sustained rise still crosses ``kv_high`` within
        a few ticks. Queue depth is smoothed too but rounded half-up, so a real
        online request queueing still trips the decrease branch immediately.

        The state is per replica: engines in a fleet are independent, and one
        engine's spike must not smear across another's control loop.
        """
        state = self._ewma.get(replica)
        if state is None:
            state = (float(sample.running), float(sample.waiting), sample.kv_usage)
        else:
            a = self.cfg.ewma_alpha
            running, waiting, kv_usage = state
            state = (
                a * sample.running + (1 - a) * running,
                a * sample.waiting + (1 - a) * waiting,
                a * sample.kv_usage + (1 - a) * kv_usage,
            )
        self._ewma[replica] = state
        running, waiting, kv_usage = state
        smoothed = EngineMetrics(
            running=_round_half_up(running),
            waiting=_round_half_up(waiting),
            kv_usage=kv_usage,
        )
        self.replica_metrics[replica] = smoothed
        return smoothed

    # -- fleet bookkeeping -------------------------------------------------

    def _inflight_on(self, replica: str) -> int:
        return sum(1 for url in self._item_replica.values() if url == replica)

    def _inflight_by_replica(self) -> dict[str, int]:
        counts = dict.fromkeys(self.replicas, 0)
        for url in self._item_replica.values():
            if url in counts:
                counts[url] += 1
        return counts

    def _total_target(self) -> int:
        return sum(c.target for c in self.controllers.values())

    def _placeable(self) -> list[str]:
        """Replicas this dispatcher is allowed to put batch work on."""
        return self.replicas if self.placement == "fleet" else self.replicas[:1]

    def _per_replica_report(self, targets: dict[str, int] | None = None) -> dict[str, dict]:
        inflight = self._inflight_by_replica()
        out: dict[str, dict] = {}
        for url in self.replicas:
            smoothed = self.replica_metrics.get(url)
            out[url] = {
                "target": (targets or {}).get(url, self.controllers[url].target),
                "inflight": inflight.get(url, 0),
                "score": None if smoothed is None else load_score(smoothed),
            }
        return out

    def _snapshot(self) -> list[BatchSnapshot]:
        """All active batches with their live (pending, inflight) counts."""
        out: list[BatchSnapshot] = []
        after: str | None = None
        for _ in range(MAX_BATCH_PAGES):
            page = self.repo.list_batches(BATCH_PAGE_SIZE, after)
            if not page:
                break
            for batch in page:
                if batch.status in ACTIVE_STATUSES:
                    pending, inflight = self.repo.pending_and_inflight(batch.id)
                    out.append((batch, pending, inflight))
            if len(page) < BATCH_PAGE_SIZE:
                break
            after = page[-1].id
        return out

    def _escalate(self, snapshot: list[BatchSnapshot], now: datetime) -> tuple[dict[str, int], int]:
        """Laxity → urgency → priority per batch, plus the SLA concurrency floor."""
        priorities: dict[str, int] = {}
        floor = 0
        for batch, pending, inflight in snapshot:
            remaining = pending + inflight
            priority, u = self._priority_for(batch, remaining, now)
            priorities[batch.id] = priority
            if remaining > 0 and u > self.cfg.floor_urgency:
                bucket = self._buckets.setdefault(batch.id, TokenBucket.from_config(self.cfg))
                if bucket.try_take(self._now_ts):
                    floor = max(floor, 1)
        return priorities, floor

    def _priority_for(self, batch: BatchRecord, remaining: int, now: datetime) -> tuple[int, float]:
        """``(priority, urgency)`` for one batch at ``now``.

        Urgency is measured against ``cfg.escalation_horizon_s`` — the last H
        seconds of *slack* — not against the SLA window, so a batch that is on
        track stays at ``batch_priority_max`` however deep into its 24h window
        it is.
        """
        rate = self._rate_for(batch.id)
        if rate is None:  # cold start: no drain estimate yet
            laxity = laxity_seconds(batch.expires_at, now, 0, 1.0)
        else:
            laxity = laxity_seconds(batch.expires_at, now, remaining, rate)
        u = urgency(laxity, self.cfg.escalation_horizon_s)
        return priority_for(u, self.cfg), u

    async def _fill(
        self,
        targets: dict[str, int],
        metrics: dict[str, EngineMetrics | None],
        snapshot: list[BatchSnapshot],
        priorities: dict[str, int],
        now: datetime,
    ) -> tuple[int, list[BatchRecord]]:
        """Claim up to Σ(target - in-flight) items, place them, and submit them.

        The claim is a *global* ``UPDATE ... RETURNING`` over every pending
        item, while the snapshot is a bounded page of active batches: the two
        can disagree. Any claimed item whose batch the snapshot missed has that
        batch resolved directly, so it is submitted against its own endpoint at
        its own laxity priority — never the defaults — and an item belonging to
        a batch that is expired, cancelling or otherwise no longer active is
        not submitted at all.

        Placement is :func:`tidal.dispatcher.fleet.choose_replica`, called once
        per item against an in-flight count that includes this tick's earlier
        placements. It can decline (no replica alive and under target), and a
        declined item is handed straight back to the store rather than left
        stranded in INFLIGHT — the claim's bound makes that unreachable unless
        the store returns more than it was asked for, and "unreachable" is not
        a reason to lose an item.

        Returns ``(submitted, resolved)``, where ``resolved`` are the batch
        records fetched outside the snapshot: the expiry sweep needs them, or a
        batch past its deadline but off the page would never be expired.
        """
        placeable = self._placeable()
        inflight = self._inflight_by_replica()
        need = headroom(targets, inflight, allowed=placeable)
        if need <= 0:
            return 0, []
        claimed = await asyncio.to_thread(
            self.repo.claim_pending_items, need, "edf_prefix", now=now
        )
        if not claimed:
            return 0, []

        batches = {batch.id: batch for batch, _p, _i in snapshot}
        missing = sorted({item.batch_id for item in claimed} - batches.keys())
        resolved: list[BatchRecord] = []
        if missing:
            for found, pending, open_items in await asyncio.to_thread(self._resolve, missing):
                batches[found.id] = found
                resolved.append(found)
                priorities.setdefault(
                    found.id, self._priority_for(found, pending + open_items, now)[0]
                )

        started: list[str] = []
        unplaced: list[str] = []
        submitted = 0
        for item in claimed:
            batch = batches.get(item.batch_id)
            if batch is None or batch.status not in ACTIVE_STATUSES:
                # Cancelled, expired, failed or already finalized: the sweep and
                # the store own this item's fate now, not the dispatcher.
                log.debug("item %s left unsubmitted: batch is not active", item.id)
                continue
            if now >= batch.expires_at:
                continue  # the sweep below will expire it
            replica = choose_replica(
                item.prefix_group, metrics, targets, inflight, self._sticky, allowed=placeable
            )
            if replica is None:
                log.warning("no replica with headroom for item %s — returning it", item.id)
                unplaced.append(item.id)
                continue
            if batch.status is BatchStatus.VALIDATING:
                started.append(batch.id)
                batch.status = BatchStatus.IN_PROGRESS
            self._submit(item, priorities[batch.id], batch.endpoint, replica)
            inflight[replica] = inflight.get(replica, 0) + 1
            submitted += 1

        if started:
            await asyncio.to_thread(self._mark_in_progress, started, now)
        if unplaced:
            await self._requeue_items(unplaced)
        return submitted, resolved

    async def _sweep_expired(
        self, snapshot: list[BatchSnapshot], resolved: list[BatchRecord], now: datetime
    ) -> list[str]:
        """Expire batches past their completion window; partial output survives.

        Sweeps the paged snapshot *and* any batch resolved during the fill, so a
        batch that fell outside the page still expires on the tick that touched
        one of its items.
        """
        expired: list[str] = []
        candidates = [batch for batch, _pending, _inflight in snapshot]
        seen = {batch.id for batch in candidates}
        candidates.extend(
            batch for batch in resolved if batch.id not in seen and batch.status in ACTIVE_STATUSES
        )
        for batch in candidates:
            if now < batch.expires_at:
                continue
            await self._cancel_tasks(
                [iid for iid, bid in self._item_batch.items() if bid == batch.id]
            )
            try:
                await asyncio.to_thread(self.repo.expire_batch, batch.id, now=now)
            except (KeyError, IllegalTransition):
                log.warning("could not expire batch %s", batch.id, exc_info=True)
                continue
            expired.append(batch.id)
            await self._call_finalize(batch.id)
        return expired

    async def _open_circuit(self) -> None:
        """No engine left: stop everything, requeue, and forget the stale EWMA.

        The global ``requeue_inflight`` is legitimate here precisely *because*
        nothing is live: every in-flight item belongs to a dead replica, and the
        bulk requeue costs those items no attempt.
        """
        for controller in self.controllers.values():
            controller.reset()
        await self._cancel_tasks(list(self._inflight))
        self.replica_metrics = dict.fromkeys(self.replicas)
        self._ewma = dict.fromkeys(self.replicas)
        try:
            await asyncio.to_thread(self.repo.requeue_inflight)
        except Exception:
            log.exception("requeue after engine-down failed")

    async def _open_replica_circuit(self, replica: str) -> None:
        """One engine is gone, the rest of the fleet is not.

        Its controller goes to zero and its EWMA is forgotten (a stale sample
        would let it come back at the target it had when it died), its in-flight
        items are cancelled and handed back to the store, and every prefix group
        homed on it is released so the next tick re-places them.
        """
        log.warning("replica %s down — opening its circuit", replica)
        self.controllers[replica].reset()
        self.replica_metrics[replica] = None
        self._ewma[replica] = None
        for group, url in list(self._sticky.items()):
            if url == replica:
                del self._sticky[group]
        item_ids = [iid for iid, url in self._item_replica.items() if url == replica]
        if not item_ids:
            return
        await self._cancel_tasks(item_ids)
        await self._requeue_items(item_ids)

    async def _requeue_items(self, item_ids: list[str]) -> None:
        """Hand specific items back to the store as a retryable failure."""
        for item_id in item_ids:
            try:
                await asyncio.to_thread(
                    self.repo.record_item_failure,
                    item_id,
                    REPLICA_DOWN_ERROR,
                    retryable=True,
                    now=self._now,
                )
            except (KeyError, IllegalTransition):
                log.debug("item %s could not be requeued", item_id, exc_info=True)

    # -- submission --------------------------------------------------------

    def _submit(self, item: ItemRecord, priority: int, endpoint: str, replica: str) -> None:
        self._item_batch[item.id] = item.batch_id
        self._item_replica[item.id] = replica
        self.replica_of[item.custom_id] = replica
        task = asyncio.create_task(
            self._run_item(item, priority, endpoint, replica), name=f"tidal-item-{item.id}"
        )
        self._inflight[item.id] = task

    async def _chat(
        self, replica: str, body: dict, priority: int, endpoint: str
    ) -> tuple[dict, int, int]:
        """Submit to one replica, through whichever client shape we were given."""
        if self._is_fleet:
            return await self.client.chat(replica, body, priority, endpoint=endpoint)
        return await self.client.chat(body, priority, endpoint=endpoint)

    async def _run_item(self, item: ItemRecord, priority: int, endpoint: str, replica: str) -> None:
        try:
            try:
                result, ptoks, ctoks = await self._chat(
                    replica, item.request_json, priority, endpoint
                )
            except asyncio.CancelledError:
                raise
            except FatalUpstream as exc:
                await self._record_failure(item, str(exc), retryable=False)
            except UpstreamError as exc:
                await self._record_failure(item, str(exc), retryable=True)
            except EngineDown as exc:
                await self._record_failure(item, str(exc), retryable=True)
            except Exception as exc:  # unknown → assume transient
                await self._record_failure(item, f"{type(exc).__name__}: {exc}", retryable=True)
            else:
                await self._record_success(item, result, ptoks, ctoks)
        finally:
            self._inflight.pop(item.id, None)
            self._item_batch.pop(item.id, None)
            self._item_replica.pop(item.id, None)

    async def _record_success(self, item: ItemRecord, result: dict, ptoks: int, ctoks: int) -> None:
        try:
            record = await asyncio.to_thread(
                self.repo.record_item_success,
                item.id,
                result,
                ptoks,
                ctoks,
                result.get("id"),
                now=self._now,
            )
        except (KeyError, IllegalTransition):
            log.warning("late result for item %s ignored", item.id, exc_info=True)
            await self._call_finalize(item.batch_id)
            return
        if record.state is not ItemState.SUCCEEDED:
            # The sweep expired or cancelled this item while the request was
            # still in the engine. The store kept it terminal and counted
            # nothing, so nothing may be metered here either — but finalize
            # still has to run, or a batch cancelled while its last item was in
            # flight would sit in CANCELLING forever.
            log.warning("late result for item %s dropped (state=%s)", item.id, record.state.value)
            await self._call_finalize(item.batch_id)
            return
        self._rate(item.batch_id).record(now=self._now_ts)
        self._global_rate.record(now=self._now_ts)
        await self._call_hook(self.on_success, record)
        await self._call_finalize(item.batch_id)

    async def _record_failure(self, item: ItemRecord, error: str, *, retryable: bool) -> None:
        try:
            record = await asyncio.to_thread(
                self.repo.record_item_failure,
                item.id,
                error,
                retryable=retryable,
                now=self._now,
            )
        except (KeyError, IllegalTransition):
            log.warning("late failure for item %s ignored", item.id, exc_info=True)
            await self._call_finalize(item.batch_id)
            return
        if record.state in _TERMINAL_ITEM_STATES:
            await self._call_finalize(item.batch_id)

    async def _cancel_tasks(self, item_ids: list[str]) -> None:
        tasks = [t for iid in item_ids if (t := self._inflight.pop(iid, None)) is not None]
        for iid in item_ids:
            self._item_batch.pop(iid, None)
            self._item_replica.pop(iid, None)
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- hooks -------------------------------------------------------------

    async def _call_finalize(self, batch_id: str) -> None:
        if self.finalize is None:
            return
        if self._finalize_wants_repo is None:
            self._finalize_wants_repo = _wants_repo(self.finalize)
        args = (self.repo, batch_id) if self._finalize_wants_repo else (batch_id,)
        await self._call_hook(self.finalize, *args)

    async def _call_hook(self, hook: Callable | None, *args: object) -> object:
        """Run a hook off the loop; a broken hook never breaks the dispatcher."""
        if hook is None:
            return None
        try:
            if inspect.iscoroutinefunction(hook):
                return await hook(*args)
            result = await asyncio.to_thread(hook, *args)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception:
            log.exception("dispatcher hook %r failed", getattr(hook, "__name__", hook))
            return None

    # -- rate bookkeeping --------------------------------------------------

    def _rate(self, batch_id: str) -> ObservedRate:
        rate = self._rates.get(batch_id)
        if rate is None:
            rate = self._rates[batch_id] = ObservedRate(RATE_WINDOW_S)
        return rate

    def _rate_for(self, batch_id: str) -> float | None:
        """Observed items/sec for a batch: its own, else global, else unknown."""
        own = self._rates.get(batch_id)
        if own is not None and len(own):
            return own.rate(now=self._now_ts)
        if len(self._global_rate):
            return self._global_rate.rate(now=self._now_ts)
        return None

    # -- store helpers (run inside one thread hop) -------------------------

    def _resolve(self, batch_ids: list[str]) -> list[BatchSnapshot]:
        """Fetch batches the snapshot missed, with their live item counts."""
        out: list[BatchSnapshot] = []
        for batch_id in batch_ids:
            try:
                batch = self.repo.get_batch(batch_id)
                if batch is None:
                    continue
                out.append((batch, *self.repo.pending_and_inflight(batch_id)))
            except KeyError:  # pragma: no cover - deleted mid-tick
                log.debug("claimed item's batch %s vanished", batch_id, exc_info=True)
        return out

    def _mark_in_progress(self, batch_ids: list[str], now: datetime) -> None:
        for batch_id in dict.fromkeys(batch_ids):
            try:
                self.repo.set_batch_status(batch_id, BatchStatus.IN_PROGRESS, now=now)
            except (KeyError, IllegalTransition):
                log.debug("batch %s not moved to in_progress", batch_id, exc_info=True)


def _wants_repo(hook: Callable) -> bool:
    """Does this ``finalize`` hook take ``(repo, batch_id)`` or just ``(batch_id,)``?

    The contract is ``(repo, batch_id)`` — the dispatcher owns no assembler and
    must hand its repository over. Callers that already closed over the
    repository (the CLI does) naturally write a one-argument hook, and silently
    logging a ``TypeError`` on every completion would mean batches never
    finalize, so both shapes are accepted. Unknown signatures (C callables)
    default to the contract.
    """
    try:
        params = list(inspect.signature(hook).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        return True
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2


def _round_half_up(value: float) -> int:
    """Half-up rounding for the smoothed queue depth (0.5 waiting → 1)."""
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)
