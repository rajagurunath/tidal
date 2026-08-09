"""``ReplicaSet`` — one :class:`VllmClient` per engine, fanned out.

The whole point of the class is *failure isolation*: one replica being down
must cost the fleet that replica and nothing else. Every test here is a
statement about that boundary, driven through ``httpx.MockTransport`` so no
engine is involved.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tidal.config import TidalConfig
from tidal.dispatcher.vllm_client import (
    EngineMetrics,
    FatalUpstream,
    ReplicaSet,
    RetryableUpstream,
)

A = "http://a:8000"
B = "http://b:8001"


def metrics_text(running: int, waiting: int, kv: float) -> str:
    return (
        f"vllm:num_requests_running {running}.0\n"
        f"vllm:num_requests_waiting {waiting}.0\n"
        f"vllm:kv_cache_usage_perc {kv}\n"
    )


def make_set(handler, replicas: str = f"{A},{B}", **cfg_kwargs) -> ReplicaSet:
    cfg = TidalConfig(vllm_replicas=replicas, **cfg_kwargs)
    return ReplicaSet(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


# -- construction -----------------------------------------------------------


def test_a_client_per_replica_in_configuration_order():
    fleet = make_set(lambda request: httpx.Response(200, text=""))
    assert fleet.urls == [A, B]
    assert fleet.client(A).cfg.vllm_base_url == A
    assert fleet.client(A).cfg.vllm_metrics_url == f"{A}/metrics"
    assert fleet.client(B).cfg.vllm_base_url == B


def test_an_unconfigured_fleet_is_one_replica_at_the_single_engine_url():
    fleet = make_set(lambda request: httpx.Response(200, text=""), replicas="")
    assert fleet.urls == [TidalConfig().vllm_base_url]


def test_an_unknown_replica_is_a_key_error_not_a_silent_default():
    fleet = make_set(lambda request: httpx.Response(200, text=""))
    with pytest.raises(KeyError):
        fleet.client("http://nowhere:9999")


# -- scrape_all -------------------------------------------------------------


async def test_scrape_all_returns_one_entry_per_replica_keyed_by_url():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a":
            return httpx.Response(200, text=metrics_text(4, 2, 0.42))
        return httpx.Response(200, text=metrics_text(1, 0, 0.10))

    fleet = make_set(handler)
    assert await fleet.scrape_all() == {
        A: EngineMetrics(running=4, waiting=2, kv_usage=0.42),
        B: EngineMetrics(running=1, waiting=0, kv_usage=0.10),
    }
    await fleet.aclose()


async def test_one_dead_replica_reads_as_none_and_leaves_the_others_intact():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, text=metrics_text(1, 0, 0.10))

    fleet = make_set(handler)
    sample = await fleet.scrape_all()
    assert sample[A] is None
    assert sample[B] == EngineMetrics(running=1, waiting=0, kv_usage=0.10)
    await fleet.aclose()


async def test_a_replica_answering_a_bad_status_is_down_not_idle():
    # An engine returning 503 on /metrics reads as *no sample*, never as an
    # idle engine with zero KV usage — the permissive reading would hand the
    # whole batch pool to the sickest replica in the fleet.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text=metrics_text(0, 0, 0.0))

    fleet = make_set(handler)
    sample = await fleet.scrape_all()
    assert sample[A] is None
    assert sample[B] == EngineMetrics(running=0, waiting=0, kv_usage=0.0)
    await fleet.aclose()


async def test_every_replica_down_is_a_full_dict_of_nones_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    fleet = make_set(handler)
    assert await fleet.scrape_all() == {A: None, B: None}
    await fleet.aclose()


async def test_scrape_all_keeps_the_configured_order():
    fleet = make_set(lambda request: httpx.Response(200, text=metrics_text(0, 0, 0.0)))
    assert list((await fleet.scrape_all()).keys()) == [A, B]
    await fleet.aclose()


# -- chat -------------------------------------------------------------------


async def test_chat_goes_to_the_replica_it_was_addressed_to():
    seen: list[tuple[str, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)["priority"]))
        return httpx.Response(200, json={"id": "cmpl-1", "usage": {"prompt_tokens": 3}})

    fleet = make_set(handler)
    _result, ptoks, ctoks = await fleet.chat(B, {"model": "m", "messages": []}, 42)

    assert seen == [(f"{B}/v1/chat/completions", 42)]
    assert (ptoks, ctoks) == (3, 0)
    await fleet.aclose()


async def test_chat_honours_the_batch_endpoint_per_replica():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"id": "cmpl-1", "usage": {}})

    fleet = make_set(handler)
    await fleet.chat(A, {"model": "m", "prompt": "hi"}, 1, endpoint="/v1/completions")
    assert seen == [f"{A}/v1/completions"]
    await fleet.aclose()


async def test_chat_error_taxonomy_survives_the_fan_out():
    fleet = make_set(lambda request: httpx.Response(503, text="overloaded"))
    with pytest.raises(RetryableUpstream):
        await fleet.chat(A, {"model": "m"}, 1)
    await fleet.aclose()

    fleet = make_set(lambda request: httpx.Response(400, text="bad model"))
    with pytest.raises(FatalUpstream):
        await fleet.chat(A, {"model": "m"}, 1)
    await fleet.aclose()


async def test_chat_to_an_unknown_replica_raises_rather_than_guessing():
    fleet = make_set(lambda request: httpx.Response(200, json={"id": "x", "usage": {}}))
    with pytest.raises(KeyError):
        await fleet.chat("http://nowhere:9999", {"model": "m"}, 1)
    await fleet.aclose()


# -- lifecycle --------------------------------------------------------------


async def test_the_shared_http_client_is_closed_once_when_owned():
    cfg = TidalConfig(vllm_replicas=f"{A},{B}")
    fleet = ReplicaSet(cfg)
    http = fleet.client(A)._http
    assert fleet.client(B)._http is http, "replicas share one connection pool"
    await fleet.aclose()
    assert http.is_closed


async def test_an_injected_http_client_is_left_open_for_its_owner():
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    fleet = ReplicaSet(TidalConfig(vllm_replicas=A), client=http)
    await fleet.aclose()
    assert not http.is_closed
    await http.aclose()


async def test_replica_set_is_an_async_context_manager():
    async with ReplicaSet(TidalConfig(vllm_replicas=A)) as fleet:
        assert fleet.urls == [A]


# -- the factory ------------------------------------------------------------


def test_make_client_gives_a_plain_client_when_no_fleet_is_configured():
    from tidal.dispatcher.vllm_client import VllmClient, make_client

    assert isinstance(make_client(TidalConfig()), VllmClient)
    assert isinstance(make_client(TidalConfig(vllm_replicas="  ")), VllmClient)


def test_make_client_gives_a_replica_set_as_soon_as_replicas_are_named():
    from tidal.dispatcher.vllm_client import make_client

    client = make_client(TidalConfig(vllm_replicas=f"{A},{B}"))
    assert isinstance(client, ReplicaSet)
    assert client.urls == [A, B]
