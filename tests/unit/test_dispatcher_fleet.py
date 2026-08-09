"""Dispatcher across a fleet of engines (technique A, multi-replica).

Same discipline as ``test_dispatcher.py``: an injected clock, fake
collaborators, no sleeping and no engine. The store fake is reused from the
single-engine suite on purpose — the fleet must not need a different store.

The two properties everything else hangs off:

* **The circuit is replica-local.** One engine dying costs the fleet that
  engine's capacity and its in-flight items, and nothing else; only a fleet
  with no live replica left behaves like today's single-engine circuit.
* **One replica behaves exactly like no fleet at all.** A ``ReplicaSet`` of one
  produces the same targets, submissions and priorities as the plain client, so
  turning the feature on cannot move any number already measured.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from tidal.config import TidalConfig
from tidal.dispatcher.loop import Dispatcher
from tidal.dispatcher.vllm_client import EngineDown, EngineMetrics
from tidal.store.interfaces import ItemState

from .test_dispatcher import T0, FakeClient, FakeRepo, run_tick

A = "http://a:8000"
B = "http://b:8001"


class FakeFleetClient:
    """Scriptable stand-in for :class:`ReplicaSet`."""

    def __init__(self, urls: list[str], **load) -> None:
        self.urls = list(urls)
        self._metrics = {url: EngineMetrics(**load) if load else EngineMetrics() for url in urls}
        self.down: set[str] = set()
        self.gates: dict[str, asyncio.Event] = {}
        #: ``(replica, body, priority, endpoint)`` per submission, in order.
        self.calls: list[tuple[str, dict, int, str]] = []
        self.scrapes = 0

    # -- scripting ---------------------------------------------------------

    def load(self, url: str, *, running: int = 0, waiting: int = 0, kv_usage: float = 0.10) -> None:
        self._metrics[url] = EngineMetrics(running=running, waiting=waiting, kv_usage=kv_usage)

    def kill(self, url: str) -> None:
        self.down.add(url)

    def revive(self, url: str) -> None:
        self.down.discard(url)

    def hold(self, url: str) -> asyncio.Event:
        """Make submissions to ``url`` hang until the returned event is set."""
        gate = self.gates[url] = asyncio.Event()
        return gate

    @property
    def placements(self) -> list[str]:
        return [replica for replica, _b, _p, _e in self.calls]

    # -- ReplicaSet surface ------------------------------------------------

    async def scrape_all(self) -> dict[str, EngineMetrics | None]:
        self.scrapes += 1
        return {url: (None if url in self.down else self._metrics[url]) for url in self.urls}

    async def chat(
        self, replica: str, body: dict, priority: int, *, endpoint: str = "/v1/chat/completions"
    ) -> tuple[dict, int, int]:
        self.calls.append((replica, body, priority, endpoint))
        gate = self.gates.get(replica)
        if gate is not None:
            await gate.wait()
        n = len(self.calls)
        return {"id": f"cmpl-{n}", "choices": [{"index": 0, "message": {"content": "ok"}}]}, 11, 7


def make_fleet(
    repo: FakeRepo, client: FakeFleetClient, cfg: TidalConfig | None = None, **kwargs
) -> Dispatcher:
    cfg = cfg or TidalConfig(vllm_replicas=",".join(client.urls))
    return Dispatcher(cfg, repo, client, **kwargs)


# --------------------------------------------------------------------------
# the single-replica regression
# --------------------------------------------------------------------------


async def test_a_one_replica_fleet_decides_exactly_what_the_plain_client_does():
    """Turning the fleet on with one engine must not move a single number."""
    load = [(0, 0.10), (0, 0.10), (0, 0.10), (4, 0.95), (0, 0.10), (2, 0.50), (0, 0.10)]

    single_repo, single_client = FakeRepo(), FakeClient(kv_usage=0.10)
    single_repo.add_batch(40)
    single = Dispatcher(TidalConfig(), single_repo, single_client)

    fleet_repo, fleet_client = FakeRepo(), FakeFleetClient([TidalConfig().vllm_base_url])
    fleet_repo.add_batch(40)
    fleet = make_fleet(fleet_repo, fleet_client, TidalConfig())

    single_reports, fleet_reports = [], []
    for i, (waiting, kv) in enumerate(load):
        single_client.load(waiting=waiting, kv_usage=kv)
        fleet_client.load(fleet_client.urls[0], waiting=waiting, kv_usage=kv)
        now = T0 + timedelta(seconds=i)
        single_reports.append(await run_tick(single, now))
        fleet_reports.append(await run_tick(fleet, now))

    assert [(r.target, r.submitted, r.inflight) for r in fleet_reports] == [
        (r.target, r.submitted, r.inflight) for r in single_reports
    ]
    assert [sorted(r.escalated_priorities.values()) for r in fleet_reports] == [
        sorted(r.escalated_priorities.values()) for r in single_reports
    ]
    assert len(fleet_client.calls) == len(single_client.calls)


async def test_a_plain_client_still_works_and_reports_one_replica():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    repo.add_batch(4)
    disp = Dispatcher(TidalConfig(), repo, client)

    report = await run_tick(disp, T0)
    assert disp.replicas == [TidalConfig().vllm_base_url]
    assert report.per_replica[TidalConfig().vllm_base_url]["target"] == report.target


async def test_a_single_replica_engine_down_is_still_the_old_global_circuit():
    repo, client = FakeRepo(), FakeClient(kv_usage=0.10)
    batch = repo.add_batch(4)
    disp = Dispatcher(TidalConfig(), repo, client)
    await run_tick(disp, T0)
    assert ItemState.SUCCEEDED in repo.item_states(batch.id)

    client.scrape_error = EngineDown("gone")
    report = await run_tick(disp, T0 + timedelta(seconds=1))
    assert report.circuit_open is True
    assert report.target == 0 and report.inflight == 0
    assert disp.aimd.target == 0
    assert repo.requeue_calls == 1


# --------------------------------------------------------------------------
# per-replica control
# --------------------------------------------------------------------------


async def test_each_replica_runs_its_own_aimd_target():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(200)
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=16))

    for i in range(4):
        client.load(A, kv_usage=0.10)
        client.load(B, kv_usage=0.10)
        await run_tick(disp, T0 + timedelta(seconds=i))
    assert disp.controllers[A].target == disp.controllers[B].target == 4

    # A's online tenant starts queueing; B is untouched.
    for i in range(4, 8):
        client.load(A, waiting=8, kv_usage=0.10)
        client.load(B, kv_usage=0.10)
        report = await run_tick(disp, T0 + timedelta(seconds=i))

    assert disp.controllers[A].target < 4, "A must back off"
    assert disp.controllers[B].target > 4, "B must keep growing — it is not the busy engine"
    assert report.target == disp.controllers[A].target + disp.controllers[B].target


async def test_placement_follows_the_load_scores():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(200)
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8))

    client.load(A, running=24, kv_usage=0.90)
    client.load(B, running=1, kv_usage=0.05)
    for i in range(3):
        await run_tick(disp, T0 + timedelta(seconds=i))

    assert client.placements.count(B) > client.placements.count(A)


async def test_the_report_carries_target_inflight_and_score_per_replica():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(50)
    disp = make_fleet(repo, client)
    client.load(A, running=8, kv_usage=0.40)
    client.load(B, running=0, kv_usage=0.05)

    report = await disp.tick(T0)
    await disp.drain()

    assert set(report.per_replica) == {A, B}
    for url in (A, B):
        assert set(report.per_replica[url]) == {"target", "inflight", "score"}
    assert report.per_replica[A]["score"] > report.per_replica[B]["score"]
    assert sum(r["target"] for r in report.per_replica.values()) == report.target


async def test_the_claim_is_bounded_by_the_fleets_total_headroom():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(200)
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8))
    claims: list[int] = []
    original = repo.claim_pending_items

    def spy(limit: int, order: str = "edf_prefix", *, now=None):
        claims.append(limit)
        return original(limit, order, now=now)

    repo.claim_pending_items = spy  # type: ignore[method-assign]
    for i in range(4):
        await run_tick(disp, T0 + timedelta(seconds=i))

    # Both controllers additively increase from 0, so the fleet's headroom is
    # the sum of the two targets each tick: 2, 4, 6, 8.
    assert claims == [2, 4, 6, 8]


async def test_a_store_that_over_delivers_never_oversubscribes_a_replica():
    """The claim is a hard bound; a store handing back more is not a licence."""
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(20)
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8))
    original = repo.claim_pending_items

    def greedy(limit: int, order: str = "edf_prefix", *, now=None):
        return original(limit + 5, order, now=now)

    repo.claim_pending_items = greedy  # type: ignore[method-assign]

    report = await run_tick(disp, T0)
    assert report.submitted == 2, "one slot per replica, whatever the store returned"
    # The five items the fleet could not place are back where they came from,
    # not left stranded in INFLIGHT until a crash-recovery sweep finds them.
    states = repo.item_states(next(iter(repo.batches)))
    assert states.count(ItemState.SUCCEEDED) == 2
    assert states.count(ItemState.PENDING) == 18
    assert ItemState.INFLIGHT not in states


# --------------------------------------------------------------------------
# replica-local circuit
# --------------------------------------------------------------------------


async def test_one_dead_replica_leaves_the_other_flowing():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(200)
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8))
    for i in range(3):
        await run_tick(disp, T0 + timedelta(seconds=i))
    before = len(client.calls)

    client.kill(A)
    report = await run_tick(disp, T0 + timedelta(seconds=3))

    assert report.circuit_open is False, "a fleet with a live replica is not open-circuit"
    assert report.per_replica[A]["score"] is None
    assert report.per_replica[A]["target"] == 0
    assert report.per_replica[B]["target"] > 0
    assert len(client.calls) > before
    assert all(replica == B for replica, *_ in client.calls[before:])


async def test_only_the_dead_replicas_in_flight_items_are_requeued():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    batch = repo.add_batch(20)
    # max_inflight=3 so both replicas are full by the third tick: the surviving
    # replica has no headroom left to re-claim the requeued work with, which is
    # what makes "untouched" observable rather than a race.
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=3))
    client.hold(A)
    client.hold(B)

    for i in range(3):  # fill both replicas; nothing completes (both gated)
        await disp.tick(T0 + timedelta(seconds=i))
    placed = disp.inflight_replicas
    on_a = [iid for iid, url in placed.items() if url == A]
    on_b = [iid for iid, url in placed.items() if url == B]
    assert on_a and on_b

    client.kill(A)
    await disp.tick(T0 + timedelta(seconds=3))

    states = {item.id: item.state for item in repo.items[batch.id]}
    assert all(states[iid] is ItemState.PENDING for iid in on_a), "A's work goes back to the store"
    assert all(states[iid] is ItemState.INFLIGHT for iid in on_b), "B's work is untouched"
    assert repo.requeue_calls == 0, "a live fleet must never do a global requeue"


async def test_a_recovered_replica_rejoins_the_fleet():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(200)
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8))
    client.kill(A)
    for i in range(3):
        await run_tick(disp, T0 + timedelta(seconds=i))
    assert A not in client.placements

    client.revive(A)
    for i in range(3, 8):
        await run_tick(disp, T0 + timedelta(seconds=i))
    assert A in client.placements


async def test_every_replica_down_opens_the_whole_circuit():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(50)
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8))
    for i in range(3):
        await run_tick(disp, T0 + timedelta(seconds=i))

    client.kill(A)
    client.kill(B)
    report = await run_tick(disp, T0 + timedelta(seconds=3))

    assert report.circuit_open is True
    assert report.target == 0 and report.inflight == 0
    assert all(c.target == 0 for c in disp.controllers.values())
    assert repo.requeue_calls == 1, "no replica left: the old global requeue is the cheap one"


# --------------------------------------------------------------------------
# placement policies
# --------------------------------------------------------------------------


async def test_pinned_placement_sends_every_item_to_the_first_replica():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(200)
    disp = make_fleet(
        repo,
        client,
        TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8),
        placement="pinned",
    )
    client.load(A, running=24, kv_usage=0.60)  # much the busier engine…
    client.load(B, running=0, kv_usage=0.02)  # …and still the only target
    for i in range(4):
        await run_tick(disp, T0 + timedelta(seconds=i))

    assert client.placements and set(client.placements) == {A}


async def test_pinned_placement_only_claims_the_pinned_replicas_headroom():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    repo.add_batch(200)
    disp = make_fleet(
        repo,
        client,
        TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8),
        placement="pinned",
    )
    claims: list[int] = []
    original = repo.claim_pending_items

    def spy(limit: int, order: str = "edf_prefix", *, now=None):
        claims.append(limit)
        return original(limit, order, now=now)

    repo.claim_pending_items = spy  # type: ignore[method-assign]
    for i in range(4):
        await run_tick(disp, T0 + timedelta(seconds=i))

    assert claims == [1, 2, 3, 4], "only replica 0's target may be claimed against"


async def test_an_unknown_placement_policy_is_refused_at_construction():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    with pytest.raises(ValueError, match="placement"):
        make_fleet(repo, client, placement="round-robin")


# --------------------------------------------------------------------------
# bookkeeping the harness reads back
# --------------------------------------------------------------------------


async def test_the_dispatcher_records_which_replica_each_item_went_to():
    repo, client = FakeRepo(), FakeFleetClient([A, B])
    batch = repo.add_batch(20)
    disp = make_fleet(repo, client, TidalConfig(vllm_replicas=f"{A},{B}", max_inflight=8))
    for i in range(4):
        await run_tick(disp, T0 + timedelta(seconds=i))

    custom_ids = {item.custom_id for item in repo.items[batch.id]}
    assert disp.replica_of, "placement must be recoverable after the run"
    assert set(disp.replica_of) <= custom_ids
    assert set(disp.replica_of.values()) <= {A, B}
    assert len(disp.replica_of) == len(client.calls)
