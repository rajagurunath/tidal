"""Async vLLM client: batch-item submission plus the Prometheus metrics scrape.

Two calls, both over one shared ``httpx.AsyncClient``:

``chat(body, priority)``
    POSTs an OpenAI request with ``stream=False`` and vLLM's ``priority``
    extra-body field (lower number = scheduled sooner; online traffic is 0).
    Returns ``(result_json, prompt_tokens, completion_tokens)`` straight from
    the engine's ``usage`` block — the metering ledger prices exactly what the
    engine reported, never an estimate.

``scrape()``
    GETs ``/metrics`` and pulls the three gauges the dispatcher controls on.
    A failure to reach or parse the engine raises :class:`EngineDown`, which
    the dispatcher treats as "circuit open": target 0, in-flight requeued.

Error taxonomy (spec §7): 5xx / timeouts / transport failures are
:class:`RetryableUpstream` (the item goes back to PENDING until
``max_item_attempts``); 4xx is :class:`FatalUpstream` (the item fails
immediately and lands in the batch's error file).

:class:`ReplicaSet` is the same two calls across a *fleet* of engines: one
``VllmClient`` per replica over one shared connection pool, a fan-out scrape
that reports a dead replica as ``None`` instead of raising, and a ``chat`` that
takes the replica to submit to. Nothing in it decides *which* replica — that is
:mod:`tidal.dispatcher.fleet`, which is pure.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, replace

import httpx

from tidal.config import TidalConfig

__all__ = [
    "EngineDown",
    "EngineMetrics",
    "FatalUpstream",
    "ReplicaSet",
    "RetryableUpstream",
    "UpstreamError",
    "VllmClient",
    "parse_metrics",
]

log = logging.getLogger("tidal.dispatcher.vllm_client")

#: Generation can legitimately take minutes on a CPU dev box.
DEFAULT_TIMEOUT_S = 600.0
#: The scrape drives a 1 Hz control loop — it must fail fast, not hang the tick.
DEFAULT_METRICS_TIMEOUT_S = 5.0

#: 4xx statuses that are still worth retrying (transient, not caller error).
_RETRYABLE_4XX = frozenset({408, 409, 425, 429})


class EngineDown(Exception):
    """The engine could not be reached or did not answer the metrics scrape."""


class UpstreamError(Exception):
    """Base for engine responses that failed a single item."""

    def __init__(self, status: int | None, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}" if status is not None else detail)


class RetryableUpstream(UpstreamError):
    """Transient failure — the item goes back to PENDING and is re-submitted."""


class FatalUpstream(UpstreamError):
    """Permanent failure (bad request) — the item fails without a retry."""


@dataclass(frozen=True)
class EngineMetrics:
    """The three gauges the AIMD controller reads, per scrape."""

    running: int = 0
    waiting: int = 0
    kv_usage: float = 0.0


def _gauge(name: str) -> re.Pattern[str]:
    """Label-tolerant matcher for one Prometheus gauge.

    Anchoring on ``{`` or whitespace after the name keeps
    ``vllm:num_requests_running_total`` from matching
    ``vllm:num_requests_running``.
    """
    return re.compile(
        rf"^{re.escape(name)}(?:\{{[^}}]*\}})?[ \t]+([-+0-9.eE]+|NaN)[ \t]*$",
        re.MULTILINE,
    )


_RUNNING = _gauge("vllm:num_requests_running")
_WAITING = _gauge("vllm:num_requests_waiting")
_KV_USAGE = _gauge("vllm:kv_cache_usage_perc")


def _values(pattern: re.Pattern[str], text: str) -> list[float]:
    out = []
    for raw in pattern.findall(text):
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - "NaN" only
            continue
        if value == value:  # skip NaN
            out.append(value)
    return out


def parse_metrics(text: str) -> EngineMetrics:
    """Parse a vLLM ``/metrics`` payload.

    Counts are summed across label sets (a server may expose more than one
    model) and KV usage takes the maximum — the fullest cache is the one that
    will start preempting. Missing gauges read as zero, which is the
    permissive direction: an engine that reports nothing is idle, and the AIMD
    loop still bounds concurrency by ``max_inflight``.
    """
    running = _values(_RUNNING, text)
    waiting = _values(_WAITING, text)
    kv = _values(_KV_USAGE, text)
    return EngineMetrics(
        running=int(sum(running)),
        waiting=int(sum(waiting)),
        kv_usage=max(kv) if kv else 0.0,
    )


class VllmClient:
    """Thin async wrapper over a stock vLLM OpenAI server."""

    def __init__(
        self,
        cfg: TidalConfig,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        metrics_timeout: float = DEFAULT_METRICS_TIMEOUT_S,
    ) -> None:
        self.cfg = cfg
        self.timeout = timeout
        self.metrics_timeout = metrics_timeout
        self._owns_http = client is None
        self._http = client if client is not None else httpx.AsyncClient(timeout=timeout)

    # -- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> VllmClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # -- submission --------------------------------------------------------

    async def chat(
        self,
        body: dict,
        priority: int,
        *,
        endpoint: str = "/v1/chat/completions",
    ) -> tuple[dict, int, int]:
        """Submit one batch item; return ``(result, prompt_tokens, completion_tokens)``.

        ``stream`` is forced off (the batch wire format has no place for a
        stream) and ``priority`` is stamped on every request — vLLM fixes a
        request's priority at admission, which is why laxity escalation is
        applied at submission time rather than to already-queued work.
        """
        payload = {**body, "stream": False, "priority": int(priority)}
        try:
            response = await self._http.post(
                self._url(endpoint), json=payload, timeout=self.timeout
            )
        except httpx.TimeoutException as exc:
            raise RetryableUpstream(None, f"timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise RetryableUpstream(None, f"transport error: {exc}") from exc

        if response.status_code != 200:
            detail = response.text[:500]
            if response.status_code >= 500 or response.status_code in _RETRYABLE_4XX:
                raise RetryableUpstream(response.status_code, detail)
            raise FatalUpstream(response.status_code, detail)

        try:
            result = response.json()
        except ValueError as exc:
            raise RetryableUpstream(response.status_code, f"malformed response: {exc}") from exc

        usage = result.get("usage") or {}
        return (
            result,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )

    # -- metrics -----------------------------------------------------------

    async def scrape(self) -> EngineMetrics:
        """Poll ``/metrics``; :class:`EngineDown` if the engine is unreachable."""
        url = self.cfg.vllm_metrics_url
        try:
            response = await self._http.get(url, timeout=self.metrics_timeout)
        except httpx.HTTPError as exc:
            raise EngineDown(f"{url}: {exc}") from exc
        if response.status_code != 200:
            raise EngineDown(f"{url}: HTTP {response.status_code}")
        return parse_metrics(response.text)

    # -- helpers -----------------------------------------------------------

    def _url(self, endpoint: str) -> str:
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        return f"{self.cfg.vllm_base_url.rstrip('/')}/{endpoint.lstrip('/')}"


class ReplicaSet:
    """One :class:`VllmClient` per engine in ``cfg.replica_urls``.

    Two properties the fleet dispatcher relies on:

    * **The scrape never raises.** ``scrape_all`` fans out concurrently and maps
      a replica that is unreachable, slow or answering non-200 to ``None``. A
      single dead engine must cost the fleet that engine's capacity and nothing
      else, and an exception here would take the whole tick down with it.
    * **Submission is addressed.** ``chat`` takes the replica URL, so placement
      is a decision made (and unit-tested) elsewhere and merely *executed* here.
      An unknown URL is a ``KeyError``: routing to a replica that does not exist
      is a bug, and quietly falling back to some default one would hide it.

    All replicas share one ``httpx.AsyncClient`` — connection pools are per
    host, so one pool already keeps the replicas' connections separate, and one
    client means one place to close.
    """

    def __init__(
        self,
        cfg: TidalConfig,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        metrics_timeout: float = DEFAULT_METRICS_TIMEOUT_S,
    ) -> None:
        self.cfg = cfg
        self._owns_http = client is None
        self._http = client if client is not None else httpx.AsyncClient(timeout=timeout)
        self._clients: dict[str, VllmClient] = {}
        for url, metrics_url in zip(cfg.replica_urls, cfg.replica_metrics_urls, strict=True):
            self._clients[url] = VllmClient(
                replace(cfg, vllm_base_url=url, vllm_metrics_url=metrics_url),
                client=self._http,
                timeout=timeout,
                metrics_timeout=metrics_timeout,
            )

    # -- topology ----------------------------------------------------------

    @property
    def urls(self) -> list[str]:
        """Replica base URLs in configuration order (the fleet's stable keys)."""
        return list(self._clients)

    def client(self, replica_url: str) -> VllmClient:
        """The client for one replica; ``KeyError`` if it is not in the fleet."""
        return self._clients[replica_url]

    def __len__(self) -> int:
        return len(self._clients)

    # -- metrics -----------------------------------------------------------

    async def scrape_all(self) -> dict[str, EngineMetrics | None]:
        """Scrape every replica concurrently. ``None`` = that replica is down."""
        urls = self.urls
        results = await asyncio.gather(
            *(self._clients[url].scrape() for url in urls), return_exceptions=True
        )
        out: dict[str, EngineMetrics | None] = {}
        for url, result in zip(urls, results, strict=True):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):  # pragma: no cover - teardown
                    raise result
                log.warning("replica %s scrape failed: %s", url, result)
                out[url] = None
            else:
                out[url] = result
        return out

    # -- submission --------------------------------------------------------

    async def chat(
        self,
        replica_url: str,
        body: dict,
        priority: int,
        *,
        endpoint: str = "/v1/chat/completions",
    ) -> tuple[dict, int, int]:
        """Submit one item to one replica; same return and errors as ``VllmClient``."""
        return await self._clients[replica_url].chat(body, priority, endpoint=endpoint)

    # -- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> ReplicaSet:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()
