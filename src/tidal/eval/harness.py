"""A/B evaluation harness (plan B5): one condition per invocation, one JSON out.

    python -m tidal.eval.harness run --condition technique_a \
        --minutes 10 --online-rps 1.0 --batch-items 400 --out results/technique_a.json

Five conditions, all sharing the *same seeded arrival schedule* and the same
batch item pool, so the only variable is how batch work reaches the engine:

``online_only``
    Stock vLLM, online traffic only. The latency baseline every other
    condition is compared against, and the "wasted capacity" reference.
``offline_only``
    Stock vLLM, batch pool only, no online traffic. The batch throughput
    *ceiling*: nothing else is competing, so no co-serving scheme can beat it.
``naive``
    Stock vLLM, online traffic + the whole batch pool fired at once at
    ``priority=100``. No gateway, no admission policy — what you get if you
    "just send the batch". The strawman that shows why policy is needed.
``technique_a``
    Stock vLLM + the Tidal **gateway**: batch goes into the store and the
    dispatcher injects it under AIMD watermarks and laxity escalation
    (external control, no engine changes).
``technique_a_fleet``
    ``technique_a`` across *N* engines: N stock vLLMs on consecutive ports with
    disjoint CPU allocations, one phase-shifted online load generator per
    engine, and a *single* store + dispatcher placing batch work across the
    fleet. The condition exists to test the claim the product rests on — that
    one replica's diurnal trough pays for another's peak — which is invisible
    to any single-engine run. ``--fleet-placement pinned`` is its control: the
    same fleet with all batch work nailed to replica 0.
``technique_b``
    vLLM launched with ``--scheduler-cls tidal.engine.scheduler.TidalScheduler``
    + the batch pool fired at once, exactly like ``naive``. The engine itself
    does the co-scheduling; the point of the condition is that identical
    client behaviour to ``naive`` produces a different outcome.

Each run writes one JSON with the raw material for every plot: per-request
online latencies with timestamps, per-item batch completions with token
counts, 1 Hz ``/metrics`` samples, technique-A ``TickReport``s, and (for
technique B) the ``TidalScheduler`` telemetry lines parsed out of the engine
log. Aggregation is done in pure functions at the bottom of this module so the
maths is unit-testable without a GPU, a CPU, or a server.

The harness owns the vLLM process for the run (launch → wait for /health →
``--warmup-s`` seconds of unmeasured traffic → teardown of the whole process
group), because a condition is only meaningful against a *fresh* engine: prefix
cache, KV pool and the latency model's fit must not carry over between
conditions. Fresh is not the same as *ready*, though, which is what the warm-up
is for: a cold engine's CUDA-graph capture and ``torch.compile`` would
otherwise be charged to the first condition's latency.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import typer

from tidal.config import TidalConfig
from tidal.dispatcher.vllm_client import parse_metrics
from tidal.eval.loadgen import (
    DEFAULT_PROMPTS,
    OnlineLoadGen,
    RequestRecord,
    percentile,
    schedule,
    summarize_latencies,
)

__all__ = [
    "CONDITIONS",
    "CPU_DEFAULTS",
    "FLEET_BASE_PORT",
    "FLEET_KVCACHE_SPACE_GB",
    "FLEET_PLACEMENTS",
    "FLEET_RESERVED_CPUS",
    "GPU_PRESET",
    "BatchCompletion",
    "MetricSample",
    "RunConfig",
    "VllmServer",
    "app",
    "batch_limits",
    "batch_summary",
    "batch_summary_per_replica",
    "diurnal_phases",
    "make_batch_items",
    "parse_tidal_stats",
    "percent_of_ceiling",
    "replica_env",
    "resolve_preset",
    "run_condition",
    "run_warmup",
    "summarize_metrics",
    "summarize_per_replica_ticks",
    "summarize_tick_reports",
    "write_logging_config",
]

CONDITIONS = (
    "online_only",
    "offline_only",
    "naive",
    "technique_a",
    "technique_a_fleet",
    "technique_b",
)

#: The engine venv on the dev box. vLLM is *not* a dependency of tidal itself.
DEFAULT_VLLM_BIN = os.environ.get(
    "TIDAL_EVAL_VLLM_BIN",
    "/Users/gurunathlunkupalivenugopal/ionet/repos/vllm-experiments/.venv/bin/vllm",
)
DEFAULT_MODEL = os.environ.get("TIDAL_EVAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
#: `src/` of this checkout — the engine process needs it on PYTHONPATH to be
#: able to import `tidal.engine.scheduler` for `--scheduler-cls`.
TIDAL_SRC = str(Path(__file__).resolve().parents[2])

BATCH_PRIORITY = 100
HEALTH_TIMEOUT_S = float(os.environ.get("TIDAL_EVAL_HEALTH_TIMEOUT_S", "300"))
METRICS_INTERVAL_S = 1.0

#: How long a cancelled direct submission waits for its in-flight item tasks to
#: unwind before it returns what it has. Seconds, not minutes: by this point the
#: run is already over budget and the only remaining job is to not lose data.
CANCEL_GRACE_S = 15.0
#: …and how long the *caller* waits for that unwind, teardown included. Beyond
#: it the caller reads the sink directly, so even a wedged httpx close cannot
#: take the completed items down with it.
SALVAGE_TIMEOUT_S = 30.0
#: Closing an httpx client with hundreds of half-open connections to a saturated
#: engine is itself a place the harness has been observed to wedge.
CLIENT_CLOSE_TIMEOUT_S = 10.0

DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_TENSOR_PARALLEL = 1

#: First port of a fleet; replica *i* gets ``FLEET_BASE_PORT + i``.
FLEET_BASE_PORT = 8399
#: Batch placement policies the fleet condition understands. ``pinned`` is the
#: control arm: same fleet, same load, all batch work on replica 0.
FLEET_PLACEMENTS = ("fleet", "pinned")
#: Logical CPUs held back from the engines for the harness itself — the
#: gateway's tick loop, N load generators and N metrics samplers all live in
#: this process, and an engine that has taken every core measures the harness
#: starving rather than the engines co-serving.
FLEET_RESERVED_CPUS = 2
#: Total ``VLLM_CPU_KVCACHE_SPACE`` (GiB) shared out across the fleet. One
#: engine's worth, split — two replicas on one box have one box's memory.
FLEET_KVCACHE_SPACE_GB = 4

#: Hard cap on warm-up requests, whatever ``--warmup-s`` says. The warm-up
#: exists to pay one-off costs, and those are paid within a handful of
#: requests; a broken engine that answers instantly must not be allowed to
#: turn the time budget into thousands of pointless round trips.
WARMUP_MAX_REQUESTS = 30

#: What the harness has always done, and still does when no flag is passed: a
#: Mac CPU build, eager-only, offline hub, httpx's stock 100-connection pool,
#: and no warm-up. Every one of these is wrong on a GPU, which is what
#: ``GPU_PRESET`` is.
CPU_DEFAULTS = {
    "enforce_eager": True,
    "hf_offline": True,
    "batch_concurrency": 100,
    "warmup_s": 0.0,
}

#: ``--gpu-preset``. CUDA graphs and ``torch.compile`` on (the single biggest
#: decode-throughput lever), the hub reachable so a cold node can fetch
#: weights, a connection pool wide enough that batch concurrency is set by the
#: scheduler under test rather than by httpx, and a warm-up window: on a GPU,
#: ``/health`` answering 200 is *not* the same as the engine being at steady
#: state — the first requests still pay CUDA-graph capture, ``torch.compile``,
#: kernel autotuning and an empty prefix cache, and on a 15-minute window that
#: lands entirely inside the measurement.
GPU_PRESET = {
    "enforce_eager": False,
    "hf_offline": False,
    "batch_concurrency": 512,
    "warmup_s": 45.0,
}

app = typer.Typer(add_completion=False, help="Tidal A/B evaluation harness.")


def resolve_preset(
    *,
    gpu_preset: bool = False,
    enforce_eager: bool | None = None,
    hf_offline: bool | None = None,
    batch_concurrency: int | None = None,
    warmup_s: float | None = None,
) -> dict:
    """Fold ``--gpu-preset`` and the individual flags into concrete settings.

    The preset only moves *defaults*: anything passed explicitly wins, so
    ``--gpu-preset --enforce-eager`` is a legitimate way to ask for CUDA-graph-
    free numbers on a GPU. ``None`` means "not passed" — which is why the CLI
    declares the two booleans as tri-state rather than plain flags.
    """
    resolved = dict(GPU_PRESET if gpu_preset else CPU_DEFAULTS)
    if enforce_eager is not None:
        resolved["enforce_eager"] = enforce_eager
    if hf_offline is not None:
        resolved["hf_offline"] = hf_offline
    if batch_concurrency is not None:
        resolved["batch_concurrency"] = batch_concurrency
    if warmup_s is not None:
        resolved["warmup_s"] = float(warmup_s)
    return resolved


# ---------------------------------------------------------------------------
# fleet topology
# ---------------------------------------------------------------------------


def replica_env(
    index: int,
    replicas: int,
    *,
    cpu_count: int | None = None,
    reserved: int = FLEET_RESERVED_CPUS,
    kvcache_space_gb: int = FLEET_KVCACHE_SPACE_GB,
) -> dict[str, str]:
    """Environment that keeps replica ``index`` out of its neighbours' way.

    Two engines on one box will happily oversubscribe every core and then
    measure each other's context switches, so each replica is given a disjoint
    slice of the machine and a share of the KV budget:

    * ``VLLM_CPU_OMP_THREADS_BIND`` — vLLM's own CPU-list format (``"2-7"``).
    * ``OMP_NUM_THREADS`` — the width of that slice, set explicitly.
    * ``VLLM_CPU_KVCACHE_SPACE`` — the fleet's total, divided.

    **macOS caveat.** ``taskset`` does not exist on macOS, and neither does
    user-level CPU affinity: vLLM's own CPU list builder says so in as many
    words (``_get_cpu_list``: *"For MacOS, no user-level CPU affinity and SMT,
    return all CPUs"*), and the binding it applies is a set of OpenMP
    environment variables (``OMP_PLACES``/``OMP_PROC_BIND``, or
    ``KMP_AFFINITY``/``GOMP_CPU_AFFINITY`` under an explicit libomp) that
    Apple's libomp does not honour as pinning. What *does* carry over is the
    thread count: vLLM derives ``OMP_NUM_THREADS`` from the width of the list,
    so a disjoint range still stops two replicas from each spawning a full
    machine's worth of OpenMP threads. So on macOS this is thread-count
    isolation, not core isolation — the portable half of the intent — and on
    Linux (every GPU box this will ever run on) it is both.
    """
    if replicas < 1:
        raise ValueError("replicas must be >= 1")
    if not 0 <= index < replicas:
        raise ValueError(f"replica index {index} outside a fleet of {replicas}")
    total = cpu_count or os.cpu_count() or 1
    usable = total - reserved
    if usable < replicas:
        # Reserving cores is a nicety; starving a replica is not. On a box too
        # small to do both, the engines get the whole machine.
        reserved, usable = 0, total
    per = max(1, usable // replicas)
    low = reserved + index * per
    high = low + per - 1
    return {
        "VLLM_CPU_OMP_THREADS_BIND": f"{low}-{high}",
        "OMP_NUM_THREADS": str(per),
        "VLLM_CPU_KVCACHE_SPACE": str(max(1, kvcache_space_gb // replicas)),
    }


def diurnal_phases(replicas: int, spec: str = "") -> list[float]:
    """Phase offset (radians) for each replica's diurnal arrival process.

    The default spreads the fleet evenly over one period, so two replicas peak
    half a period apart — the arrangement the whole multi-replica thesis rests
    on, since a fleet whose engines peak together has no trough to sell. An
    explicit ``spec`` (``"0,3.14159"``) overrides it and must name every
    replica: a short list would silently leave the tail in phase.
    """
    if replicas < 1:
        raise ValueError("replicas must be >= 1")
    if not spec.strip():
        return [i * 2.0 * math.pi / replicas for i in range(replicas)]
    values = [float(part) for part in spec.split(",") if part.strip()]
    if len(values) != replicas:
        raise ValueError(f"--diurnal-phase-list has {len(values)} phases for {replicas} replicas")
    return values


# ---------------------------------------------------------------------------
# engine process management
# ---------------------------------------------------------------------------


def write_logging_config(path: str | Path) -> str:
    """Write a ``VLLM_LOGGING_CONFIG_PATH`` file that also shows ``tidal`` logs.

    vLLM's default logging config attaches a handler to the ``vllm`` logger
    only; the root logger keeps its default level, so ``logger.info`` from
    ``tidal.engine.scheduler`` inside the engine process would be swallowed.
    Since the ``TidalScheduler`` telemetry line *is* the observability surface
    for technique B, the harness installs a config that logs both trees.

    The format leads with ``%(created)f`` (epoch seconds) so telemetry lines
    can be aligned with the run's own clock without parsing a date format.
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "tidal": {"format": "%(created)f %(levelname)s %(name)s %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "tidal",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "vllm": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "tidal": {"handlers": ["console"], "level": "INFO", "propagate": False},
        },
    }
    path = Path(path)
    path.write_text(json.dumps(config, indent=2))
    return str(path)


@dataclass
class VllmServer:
    """A vLLM OpenAI server owned for the lifetime of one condition.

    Args:
        port: HTTP port.
        flavor: ``"stock"`` or ``"tidal"`` (adds ``--scheduler-cls``).
        log_path: where combined stdout/stderr goes; the technique-B
            assertions and the timeline plot read it back.
        model / max_model_len / vllm_bin: engine knobs.
        enforce_eager: pass ``--enforce-eager`` *and* disable torchdynamo. The
            two travel together: eager mode is only the right answer where
            ``torch.compile`` is also a non-starter, i.e. the CPU build.
        tensor_parallel: ``--tensor-parallel-size``. Emitted only when >1, so a
            single-GPU (or CPU) command line stays exactly what it always was.
        hf_offline: set ``HF_HUB_OFFLINE=1``. False forces ``0``, which is what
            a cold node needs to fetch weights at all.
        env_extra: additional environment (``TIDAL_*`` for the scheduler).

    Started with ``start_new_session=True`` so teardown can signal the whole
    process group — vLLM spawns an EngineCore child, and killing only the
    parent leaves a process holding the port.
    """

    port: int
    flavor: str = "stock"
    log_path: str | None = None
    model: str = DEFAULT_MODEL
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    vllm_bin: str = DEFAULT_VLLM_BIN
    enforce_eager: bool = CPU_DEFAULTS["enforce_eager"]
    tensor_parallel: int = DEFAULT_TENSOR_PARALLEL
    hf_offline: bool = CPU_DEFAULTS["hf_offline"]
    env_extra: dict[str, str] = field(default_factory=dict)
    proc: subprocess.Popen | None = None
    _log_file: object | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        cmd = [
            self.vllm_bin,
            "serve",
            self.model,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self.max_model_len),
        ]
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.tensor_parallel > 1:
            cmd += ["--tensor-parallel-size", str(self.tensor_parallel)]
        cmd += [
            "--scheduling-policy",
            "priority",
            "--port",
            str(self.port),
        ]
        if self.flavor == "tidal":
            cmd += ["--scheduler-cls", "tidal.engine.scheduler.TidalScheduler"]
        return cmd

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "VLLM_CPU_KVCACHE_SPACE": env.get("VLLM_CPU_KVCACHE_SPACE", "4"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        if self.enforce_eager:
            # Inductor is a non-starter on the CPU build, and there is nothing
            # for it to do when the engine is running eager anyway.
            env["TORCHDYNAMO_DISABLE"] = "1"
        else:
            # An inherited TORCHDYNAMO_DISABLE would silently defeat
            # --no-enforce-eager, so drop it rather than merely not setting it.
            env.pop("TORCHDYNAMO_DISABLE", None)
        # Explicit in both directions: an inherited value must not be able to
        # override the flag the run was launched with.
        env["HF_HUB_OFFLINE"] = "1" if self.hf_offline else "0"
        if self.flavor == "tidal":
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{TIDAL_SRC}:{existing}" if existing else TIDAL_SRC
        env.update(self.env_extra)
        return env

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> VllmServer:
        log_path = self.log_path or tempfile.mkstemp(prefix="vllm-", suffix=".log")[1]
        self.log_path = log_path
        self._log_file = open(log_path, "wb")  # noqa: SIM115 - closed in stop()
        self.proc = subprocess.Popen(
            self.command(),
            cwd="/tmp",  # never launch from a source dir: vllm/ would shadow the package
            env=self.environment(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self

    def wait_ready(self, timeout: float = HEALTH_TIMEOUT_S) -> None:
        """Block until ``/health`` answers 200, or raise with the log tail."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self.proc.returncode} before becoming "
                    f"ready:\n{self.log_tail()}"
                )
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"vLLM not ready after {timeout:.0f}s:\n{self.log_tail()}")

    def stop(self, grace: float = 15.0) -> None:
        """SIGTERM the process group, SIGKILL what survives. Never raises."""
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(os.getpgid(proc.pid), sig)
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if self._log_file is not None:
            with contextlib.suppress(Exception):
                self._log_file.close()
            self._log_file = None

    def log_text(self) -> str:
        if not self.log_path or not Path(self.log_path).exists():
            return ""
        return Path(self.log_path).read_text(errors="replace")

    def log_tail(self, lines: int = 40) -> str:
        return "\n".join(self.log_text().splitlines()[-lines:])

    def __enter__(self) -> VllmServer:
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# batch work
# ---------------------------------------------------------------------------


def make_batch_items(
    count: int,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    *,
    shared_prefix: str = "You are a helpful assistant. ",
) -> list[dict]:
    """The batch pool: ``count`` OpenAI chat bodies, deterministic in the index.

    Every item carries the same system-ish preamble so the engine's prefix
    cache has something to hit — that is what batch workloads look like, and
    it is what the store's ``prefix_group`` ordering exists to exploit.
    """
    items = []
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        items.append(
            {
                "model": model,
                "messages": [{"role": "user", "content": f"{shared_prefix}{prompt} (#{i})"}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
        )
    return items


@dataclass
class BatchCompletion:
    """One batch item's outcome, timed against the run's ``t0``.

    ``replica`` is the engine that ran it — ``None`` for every single-engine
    condition, where the question does not arise.
    """

    index: int
    custom_id: str
    started_at: float
    finished_at: float
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    replica: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return asdict(self)


def batch_limits(max_connections: int) -> httpx.Limits:
    """Connection pool for direct batch submission.

    httpx's default is 100 connections with 20 kept alive, which quietly caps
    batch concurrency at ~100 no matter how large the pool is — on a GPU that
    is the harness measuring httpx rather than the engine. Keepalives are sized
    to match the ceiling so a wide pool does not spend the run reconnecting.
    """
    return httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )


async def _close_quietly(
    http: httpx.AsyncClient, timeout_s: float = CLIENT_CLOSE_TIMEOUT_S
) -> None:
    """Close a client without letting the close become the new hang.

    ``aclose`` is run as its own task and merely *waited on*: a close that is
    still going after ``timeout_s`` is abandoned rather than awaited, which is
    the difference between a run that reports salvaged results and one that gets
    killed by the external case timeout with its results still in memory.
    """
    closing = asyncio.ensure_future(http.aclose())
    with contextlib.suppress(Exception):
        await asyncio.wait({closing}, timeout=timeout_s)
    if not closing.done():
        closing.cancel()


async def drain_batch_task(
    batch_task: asyncio.Task,
    *,
    budget_s: float,
    sink: Sequence[BatchCompletion] = (),
    salvage_timeout_s: float = SALVAGE_TIMEOUT_S,
) -> tuple[list[BatchCompletion], bool]:
    """Wait ``budget_s`` for a batch submission; salvage what it did on timeout.

    Returns ``(completions, drain_timed_out)``. Three layers, because each one
    has been seen to fail on a GPU with a five-figure pool:

    1. the bounded wait itself — ``asyncio.wait`` rather than ``wait_for``, so a
       coordinator that returns partial results on cancellation is read as a
       result and not as a swallowed timeout;
    2. the cancel, after which the coordinator hands back everything it
       finished (see ``submit_batch_direct``);
    3. ``sink`` — the same list the coordinator appends to — read directly if
       even the cancelled coordinator does not come back in time. Completed work
       cannot be lost to a teardown.
    """
    _, pending = await asyncio.wait({batch_task}, timeout=max(0.0, budget_s))
    if not pending:
        if batch_task.cancelled():
            return _by_index(sink), True
        # `.result()` re-raises: a submission that failed for its own reasons is
        # a bug in the harness or the engine, and has always aborted the run.
        return list(batch_task.result()), False

    batch_task.cancel()
    with contextlib.suppress(Exception):
        await asyncio.wait({batch_task}, timeout=salvage_timeout_s)
    if batch_task.done() and not batch_task.cancelled() and batch_task.exception() is None:
        return list(batch_task.result()), True
    return _by_index(sink), True


def _by_index(completions: Sequence[BatchCompletion]) -> list[BatchCompletion]:
    """Item order, not completion order: the result file has always been sorted."""
    return sorted(completions, key=lambda c: c.index)


async def submit_batch_direct(
    base_url: str,
    items: Sequence[dict],
    *,
    t0: float,
    priority: int = BATCH_PRIORITY,
    concurrency: int | None = None,
    max_connections: int = CPU_DEFAULTS["batch_concurrency"],
    timeout_s: float = 900.0,
    per_request_timeout_s: float | None = None,
    client: httpx.AsyncClient | None = None,
    sink: list[BatchCompletion] | None = None,
) -> list[BatchCompletion]:
    """Fire the whole pool straight at vLLM (the ``naive``/``technique_b`` path).

    ``concurrency=None`` means *all at once*, which is exactly the point of the
    naive condition: no client-side admission control whatsoever. The real
    ceiling is then ``max_connections``, the size of the HTTP pool — an
    unavoidable one, but it must be a stated number rather than an httpx
    default nobody chose.

    **Cancellation returns the work already done.** A pool of tens of thousands
    of items against a saturated engine will not finish inside any drain the
    harness is willing to wait for, and a `gather` that is cancelled discards
    every completed item with it — the run then reports nothing at all rather
    than the hours of measurement it actually holds. So each completion is
    appended to a list as it lands (``sink``, if the caller wants to watch it
    from outside), outstanding items are cancelled and briefly awaited, the
    client is closed under a guard, and the salvaged completions are *returned*
    rather than lost to the ``CancelledError``.

    ``per_request_timeout_s`` bounds one item; it defaults to ``timeout_s`` so
    an unset caller measures exactly what it did before. Callers that know the
    run's shape pass the window plus the drain, which is the longest an item can
    matter for.
    """
    owns = client is None
    http = client or httpx.AsyncClient(timeout=timeout_s, limits=batch_limits(max_connections))
    gate = asyncio.Semaphore(concurrency) if concurrency else None
    request_timeout = timeout_s if per_request_timeout_s is None else per_request_timeout_s
    done: list[BatchCompletion] = sink if sink is not None else []

    async def one(index: int, body: dict) -> BatchCompletion:
        if gate is not None:
            await gate.acquire()
        began = time.perf_counter()
        error = None
        ptoks = ctoks = 0
        try:
            response = await http.post(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                json={**body, "stream": False, "priority": priority},
                timeout=request_timeout,
            )
            if response.status_code == 200:
                usage = (response.json() or {}).get("usage") or {}
                ptoks = int(usage.get("prompt_tokens") or 0)
                ctoks = int(usage.get("completion_tokens") or 0)
            else:
                error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if gate is not None:
                gate.release()
        ended = time.perf_counter()
        completion = BatchCompletion(
            index=index,
            custom_id=f"item-{index}",
            started_at=began - t0,
            finished_at=ended - t0,
            latency_s=ended - began,
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            error=error,
        )
        # Recorded before the coordinator sees it, so a cancellation that lands
        # between "response parsed" and "gather returns" still keeps the item.
        done.append(completion)
        return completion

    tasks = [asyncio.create_task(one(i, b), name=f"batch-item-{i}") for i, b in enumerate(items)]
    try:
        return list(await asyncio.gather(*tasks))
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        # Not `gather(..., return_exceptions=True)`: this task is already
        # cancelling, and `wait` neither re-raises the children's errors nor
        # waits past the grace period for one that ignores its cancel.
        with contextlib.suppress(Exception):
            await asyncio.wait(tasks, timeout=CANCEL_GRACE_S)
        return sorted(done, key=lambda c: c.index)
    finally:
        if owns:
            await _close_quietly(http)


# ---------------------------------------------------------------------------
# engine telemetry
# ---------------------------------------------------------------------------


@dataclass
class MetricSample:
    """One ``/metrics`` scrape, timed against the run's ``t0``."""

    t: float
    running: int
    waiting: int
    kv_usage: float

    def as_dict(self) -> dict:
        return asdict(self)


async def sample_metrics(
    metrics_url: str,
    stop: asyncio.Event,
    out: list[MetricSample],
    *,
    t0: float,
    interval_s: float = METRICS_INTERVAL_S,
) -> None:
    """Poll ``/metrics`` at ``interval_s`` until ``stop`` is set. Never raises."""
    async with httpx.AsyncClient(timeout=5.0) as http:
        while not stop.is_set():
            try:
                response = await http.get(metrics_url)
                if response.status_code == 200:
                    engine = parse_metrics(response.text)
                    out.append(
                        MetricSample(
                            t=time.perf_counter() - t0,
                            running=engine.running,
                            waiting=engine.waiting,
                            kv_usage=engine.kv_usage,
                        )
                    )
            except Exception:
                pass
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval_s)


#: `1754468123.456 INFO tidal.engine.scheduler tidal {'step': 100, ...}`, possibly
#: behind vLLM's multiprocess prefix — the scheduler runs in the EngineCore
#: child, whose stdout arrives as `(EngineCore pid=1234) <line>`. Hence a
#: `search`, not a `match`.
_TIDAL_STATS_RE = re.compile(r"(\d+\.\d+)\s+\w+\s+tidal\.engine\.scheduler\s+tidal\s+(\{.*\})\s*$")
_TIDAL_PROBE_RE = re.compile(r"TidalScheduler active:.*")
#: `repr(float('nan'))` is `nan`, which is not a Python *literal* — and it is
#: also not valid JSON. Both problems go away by reading it as null.
_NON_FINITE_RE = re.compile(r"(?<![\w'\"])-?(?:nan|inf)(?![\w'\"])")


def parse_tidal_stats(log_text: str, *, epoch0: float | None = None) -> list[dict]:
    """Pull ``TidalScheduler`` telemetry dicts out of an engine log.

    The scheduler logs one compact line every 100 steps. Returns the parsed
    dicts with an added ``t`` (seconds since ``epoch0``, when given) so the
    technique-B timeline can be drawn against the same clock as the online
    latencies. Non-finite floats (``mape`` before T̂'s first fit) become
    ``None`` so the result document stays strict JSON.
    """
    out: list[dict] = []
    for line in log_text.splitlines():
        match = _TIDAL_STATS_RE.search(line.strip())
        if not match:
            continue
        try:
            payload = ast.literal_eval(_NON_FINITE_RE.sub("None", match.group(2)))
        except (ValueError, SyntaxError):
            continue
        if not isinstance(payload, dict):
            continue
        stamp = float(match.group(1))
        payload["epoch"] = stamp
        if epoch0 is not None:
            payload["t"] = stamp - epoch0
        out.append(payload)
    return out


def tidal_probe_line(log_text: str) -> str | None:
    """The one-shot capability-probe line, if the scheduler ever started."""
    match = _TIDAL_PROBE_RE.search(log_text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# pure aggregation (unit-tested in tests/unit/test_harness_metrics.py)
# ---------------------------------------------------------------------------


def batch_summary(
    completions: Sequence[BatchCompletion], window_s: float, *, pool_size: int | None = None
) -> dict:
    """Throughput of the batch tier over the measurement window.

    ``window_s`` is the *online* window: every condition is compared over the
    same wall-clock span, so a condition that drains its pool early is not
    rewarded for the idle tail. ``drained`` says exactly that happened and the
    pool should be enlarged for a fair ceiling comparison.
    """
    ok = [c for c in completions if c.ok]
    in_window = [c for c in ok if c.finished_at <= window_s]
    # Technique A reports `nan` per-item latency (see `_run_technique_a`); a
    # summary over "not measured" would be a fabricated number.
    latencies = [c.latency_s for c in ok if math.isfinite(c.latency_s)]
    out_tokens = sum(c.completion_tokens for c in in_window)
    prompt_tokens = sum(c.prompt_tokens for c in in_window)
    makespan = max((c.finished_at for c in ok), default=0.0)
    total = pool_size if pool_size is not None else len(completions)
    return {
        "pool_size": total,
        "completed": len(ok),
        "failed": len(completions) - len(ok),
        "completed_in_window": len(in_window),
        "output_tokens_in_window": out_tokens,
        "prompt_tokens_in_window": prompt_tokens,
        "items_per_s": (len(in_window) / window_s) if window_s > 0 else 0.0,
        "output_tokens_per_s": (out_tokens / window_s) if window_s > 0 else 0.0,
        "makespan_s": makespan,
        "drained": bool(ok) and makespan <= window_s and len(ok) >= total,
        "item_latency": summarize_latencies(latencies),
    }


def summarize_metrics(samples: Sequence[MetricSample]) -> dict:
    """Mean/peak of the three gauges the AIMD controller reads."""
    if not samples:
        return {"samples": 0}
    running = [s.running for s in samples]
    waiting = [s.waiting for s in samples]
    kv = [s.kv_usage for s in samples]
    return {
        "samples": len(samples),
        "running_mean": sum(running) / len(running),
        "running_max": max(running),
        "waiting_mean": sum(waiting) / len(waiting),
        "waiting_max": max(waiting),
        "kv_mean": sum(kv) / len(kv),
        "kv_max": max(kv),
        "kv_p95": percentile(kv, 0.95),
    }


def summarize_tick_reports(reports: Sequence[dict]) -> dict:
    """What technique A's controller actually did over the run."""
    if not reports:
        return {"ticks": 0}
    targets = [r["target"] for r in reports]
    inflight = [r["inflight"] for r in reports]
    priorities = [p for r in reports for p in r.get("escalated_priorities", {}).values()]
    return {
        "ticks": len(reports),
        "target_mean": sum(targets) / len(targets),
        "target_min": min(targets),
        "target_max": max(targets),
        "inflight_mean": sum(inflight) / len(inflight),
        "submitted_total": sum(r["submitted"] for r in reports),
        "circuit_open_ticks": sum(1 for r in reports if r.get("circuit_open")),
        "min_priority": min(priorities) if priorities else None,
        "escalated_ticks": sum(
            1 for r in reports if any(p < 100 for p in r.get("escalated_priorities", {}).values())
        ),
    }


def batch_summary_per_replica(
    completions: Sequence[BatchCompletion], window_s: float
) -> dict[str, dict]:
    """``batch_summary`` per engine, for the completions that name one.

    A completion with no ``replica`` is not attributed to anybody: guessing
    would put a single-engine condition's whole pool on whichever URL happened
    to be first, and the per-replica breakdown exists to answer "did the fleet
    actually spread the work", which a guess would always answer yes to.
    """
    grouped: dict[str, list[BatchCompletion]] = {}
    for completion in completions:
        if completion.replica is None:
            continue
        grouped.setdefault(completion.replica, []).append(completion)
    return {url: batch_summary(group, window_s) for url, group in grouped.items()}


def summarize_per_replica_ticks(reports: Sequence[dict]) -> dict[str, dict]:
    """What each replica's controller did over a fleet run.

    ``down_ticks`` counts the ticks a replica failed its scrape (``score`` is
    ``None``); its target is still averaged in as the zero the circuit set it
    to, because "this replica was contributing nothing" is the honest reading.
    """
    per_replica: dict[str, list[dict]] = {}
    for report in reports:
        for url, entry in (report.get("per_replica") or {}).items():
            per_replica.setdefault(url, []).append(entry)
    out: dict[str, dict] = {}
    for url, entries in per_replica.items():
        scores = [e["score"] for e in entries if e.get("score") is not None]
        out[url] = {
            "ticks": len(entries),
            "target_mean": sum(e["target"] for e in entries) / len(entries),
            "target_max": max(e["target"] for e in entries),
            "inflight_mean": sum(e["inflight"] for e in entries) / len(entries),
            "score_mean": (sum(scores) / len(scores)) if scores else None,
            "score_max": max(scores) if scores else None,
            "down_ticks": sum(1 for e in entries if e.get("score") is None),
        }
    return out


def percent_of_ceiling(results: dict[str, dict], *, key: str = "output_tokens_per_s") -> dict:
    """Each condition's batch throughput as a percentage of ``offline_only``.

    The ceiling is what the engine can do for batch work with nothing else
    running; a co-serving scheme is interesting exactly to the extent that it
    gets close to it *without* moving online latency.
    """
    ceiling = (results.get("offline_only") or {}).get("batch", {}).get(key)
    if not ceiling:
        return {}
    out = {}
    for name, payload in results.items():
        value = (payload.get("batch") or {}).get(key)
        if value is not None:
            out[name] = 100.0 * value / ceiling
    return out


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Everything one condition needs. Serialized verbatim into the result."""

    condition: str
    minutes: float = 10.0
    online_rps: float = 1.0
    batch_items: int = 400
    arrival: str = "poisson"
    seed: int = 7
    model: str = DEFAULT_MODEL
    port: int = 8400
    online_max_tokens: int = 16
    batch_max_tokens: int = 16
    max_inflight: int = 4
    vllm_bin: str = DEFAULT_VLLM_BIN
    drain_timeout_s: float = 300.0
    out: str = "results/run.json"
    #: Engine shape. The defaults are the CPU testbed's; `--gpu-preset` and the
    #: individual flags move them, and since the whole config is serialized into
    #: the result JSON, every number carries the engine it was measured on.
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    tensor_parallel: int = DEFAULT_TENSOR_PARALLEL
    enforce_eager: bool = CPU_DEFAULTS["enforce_eager"]
    hf_offline: bool = CPU_DEFAULTS["hf_offline"]
    batch_concurrency: int = CPU_DEFAULTS["batch_concurrency"]
    #: Seconds of unmeasured traffic between ``/health`` and ``t0``. Zero (the
    #: CPU default) reproduces every number measured before the flag existed.
    warmup_s: float = CPU_DEFAULTS["warmup_s"]
    #: -- fleet (``technique_a_fleet`` only; ignored by every other condition) --
    replicas: int = 2
    fleet_base_port: int = FLEET_BASE_PORT
    fleet_placement: str = "fleet"
    #: Comma-separated phase offsets, one per replica. Empty spreads them
    #: evenly over the period (two replicas ⇒ half a period apart).
    diurnal_phase_list: str = ""

    @property
    def duration_s(self) -> float:
        return self.minutes * 60.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def replica_ports(self) -> list[int]:
        return [self.fleet_base_port + i for i in range(self.replicas)]

    @property
    def replica_urls(self) -> list[str]:
        return [f"http://127.0.0.1:{port}" for port in self.replica_ports]


async def run_warmup(
    cfg: RunConfig,
    *,
    base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
    clock: Callable[[], float] = time.perf_counter,
    max_requests: int = WARMUP_MAX_REQUESTS,
) -> dict:
    """Unmeasured traffic between ``/health`` and ``t0``. Returns what it did.

    ``/health`` answering 200 means the weights are loaded, not that the engine
    is at steady state: the first requests of a run still pay CUDA-graph
    capture, ``torch.compile``, kernel autotuning and a cold prefix cache. On a
    15-minute window that cost lands entirely inside the measurement, and it
    lands on whichever condition happens to be first — which is a property of
    the loop order in ``run_gpu_eval.sh``, not of the scheduler under test.

    Requests are **sequential** on purpose: the point is to walk the engine
    through its one-off costs, not to load it. They reuse the online request
    body builder, so the warm-up exercises exactly the shape the measured
    window will send. Results, errors and timings are all discarded — this is
    the one place in the harness where a failed request is not a data point.
    Whichever of ``warmup_s`` and ``max_requests`` runs out first ends it.

    This is engine warm-up, not load shape, so it applies to every condition
    that boots an engine — including ``offline_only``, whose ceiling would
    otherwise be measured against a colder engine than the conditions it is the
    divisor for. ``base_url`` aims it at one replica of a fleet; every engine in
    a fleet needs its own, for the same reason a single engine does.
    """
    if cfg.warmup_s <= 0:
        return {"requests": 0, "elapsed_s": 0.0}

    target = base_url or cfg.base_url
    gen = OnlineLoadGen(
        base_url=target, model=cfg.model, max_tokens=cfg.online_max_tokens, priority=0
    )
    url = f"{target.rstrip('/')}/v1/chat/completions"
    owns = client is None
    http = client or httpx.AsyncClient(timeout=cfg.warmup_s)

    started = clock()
    sent = 0
    try:
        while sent < max_requests:
            remaining = cfg.warmup_s - (clock() - started)
            if remaining <= 0:
                break
            # The per-request timeout is the remaining budget, so a single
            # wedged request cannot push the warm-up past the window it was
            # given (plus the one second of floor).
            with contextlib.suppress(Exception):
                await http.post(url, json=gen.body_for(sent), timeout=max(1.0, remaining))
            sent += 1
    finally:
        if owns:
            await http.aclose()

    elapsed = clock() - started
    typer.echo(f"[{cfg.condition}] warmup: {sent} requests in {elapsed:.1f}s")
    return {"requests": sent, "elapsed_s": elapsed}


async def run_condition(cfg: RunConfig, *, log_dir: str | None = None) -> dict:
    """Launch the right engine, run the condition, return the result document."""
    if cfg.condition not in CONDITIONS:
        raise ValueError(f"unknown condition {cfg.condition!r}; want one of {CONDITIONS}")

    log_dir_path = Path(log_dir or tempfile.mkdtemp(prefix="tidal-eval-"))
    log_dir_path.mkdir(parents=True, exist_ok=True)
    logging_config = write_logging_config(log_dir_path / "logging.json")

    if cfg.condition == "technique_a_fleet":
        return await _run_fleet_condition(cfg, log_dir_path, logging_config)

    flavor = "tidal" if cfg.condition == "technique_b" else "stock"
    log_path = log_dir_path / f"vllm-{cfg.condition}.log"

    server = VllmServer(
        port=cfg.port,
        flavor=flavor,
        log_path=str(log_path),
        model=cfg.model,
        vllm_bin=cfg.vllm_bin,
        max_model_len=cfg.max_model_len,
        enforce_eager=cfg.enforce_eager,
        tensor_parallel=cfg.tensor_parallel,
        hf_offline=cfg.hf_offline,
        env_extra={"VLLM_LOGGING_CONFIG_PATH": logging_config},
    )
    typer.echo(f"[{cfg.condition}] launching {flavor} vLLM on port {cfg.port} ...")
    started_at = datetime.now(UTC).isoformat()
    with server:
        typer.echo(f"[{cfg.condition}] engine ready; running {cfg.minutes:g} min")
        result = await _measure(cfg, server)
    result["condition"] = cfg.condition
    result["config"] = asdict(cfg)
    result["started_at"] = started_at
    result["finished_at"] = datetime.now(UTC).isoformat()
    result["engine_log"] = str(log_path)
    if flavor == "tidal":
        text = server.log_text()
        result["tidal_probe"] = tidal_probe_line(text)
        result["tidal_stats"] = parse_tidal_stats(text, epoch0=result["epoch0"])
    return result


async def _run_fleet_condition(cfg: RunConfig, log_dir: Path, logging_config: str) -> dict:
    """Boot ``cfg.replicas`` stock engines side by side and measure the fleet.

    The engines are started and health-checked one at a time: two cold vLLMs
    loading weights simultaneously on one box is a memory spike for no benefit,
    and a fleet where the second engine never came up must fail loudly at
    launch rather than halfway through a measured window.
    """
    if cfg.replicas < 1:
        raise ValueError(f"a fleet needs at least one replica; got replicas={cfg.replicas}")
    if cfg.fleet_placement not in FLEET_PLACEMENTS:
        raise ValueError(
            f"unknown fleet placement {cfg.fleet_placement!r}; want one of {FLEET_PLACEMENTS}"
        )

    servers = [
        VllmServer(
            port=port,
            flavor="stock",
            log_path=str(log_dir / f"vllm-{cfg.condition}-{index}.log"),
            model=cfg.model,
            vllm_bin=cfg.vllm_bin,
            max_model_len=cfg.max_model_len,
            enforce_eager=cfg.enforce_eager,
            tensor_parallel=cfg.tensor_parallel,
            hf_offline=cfg.hf_offline,
            env_extra={
                "VLLM_LOGGING_CONFIG_PATH": logging_config,
                **replica_env(index, cfg.replicas),
            },
        )
        for index, port in enumerate(cfg.replica_ports)
    ]
    typer.echo(
        f"[{cfg.condition}] launching {cfg.replicas} stock vLLM(s) on "
        f"ports {cfg.replica_ports} ({cfg.fleet_placement} placement) ..."
    )
    started_at = datetime.now(UTC).isoformat()
    with contextlib.ExitStack() as stack:
        for server in servers:
            stack.enter_context(server)
        typer.echo(f"[{cfg.condition}] fleet ready; running {cfg.minutes:g} min")
        result = await _measure_fleet(cfg, servers)
    result["condition"] = cfg.condition
    result["config"] = asdict(cfg)
    result["started_at"] = started_at
    result["finished_at"] = datetime.now(UTC).isoformat()
    result["engine_logs"] = [server.log_path for server in servers]
    result["engine_log"] = servers[0].log_path
    return result


async def _measure(cfg: RunConfig, server: VllmServer) -> dict:
    """Run the load for one condition against an already-healthy engine."""
    offsets = (
        []
        if cfg.condition == "offline_only"
        else schedule(cfg.arrival, cfg.online_rps, cfg.duration_s, cfg.seed)
    )
    items = make_batch_items(cfg.batch_items, model=cfg.model, max_tokens=cfg.batch_max_tokens)
    gen = OnlineLoadGen(
        base_url=cfg.base_url, model=cfg.model, max_tokens=cfg.online_max_tokens, priority=0
    )

    # Strictly before t0: anything the warm-up costs must fall outside the
    # measured window, or it would be reported as latency the scheduler caused.
    warmup = await run_warmup(cfg)

    t0 = time.perf_counter()
    epoch0 = time.time()
    samples: list[MetricSample] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(
        sample_metrics(f"{cfg.base_url}/metrics", stop, samples, t0=t0), name="metrics"
    )

    completions: list[BatchCompletion] = []
    drain_timed_out = False
    tick_reports: list[dict] = []
    online: list[RequestRecord] = []
    try:
        if cfg.condition == "online_only":
            online = await gen.run(offsets, t0=t0)
        elif cfg.condition in ("offline_only", "naive", "technique_b"):
            # One code path for all three direct-submission arms. `offline_only`
            # used to have no timeout at all — its window *is* the batch, which
            # is not the same thing as unbounded, and a five-figure pool against
            # a saturated engine ran until the external case timeout killed the
            # process and every completed item with it.
            sink: list[BatchCompletion] = []
            budget = cfg.drain_timeout_s
            if cfg.condition == "offline_only":
                budget += cfg.duration_s
            batch_task = asyncio.create_task(
                submit_batch_direct(
                    cfg.base_url,
                    items,
                    t0=t0,
                    max_connections=cfg.batch_concurrency,
                    per_request_timeout_s=cfg.duration_s + cfg.drain_timeout_s,
                    sink=sink,
                ),
                name="batch",
            )
            if cfg.condition != "offline_only":
                online = await gen.run(offsets, t0=t0)
            completions, drain_timed_out = await drain_batch_task(
                batch_task, budget_s=budget, sink=sink
            )
            if drain_timed_out:
                typer.echo(
                    f"[{cfg.condition}] drain timed out after {budget:g}s; "
                    f"salvaged {len(completions)}/{len(items)} completions"
                )
        else:  # technique_a
            completions, tick_reports, _placement = await _run_technique_a(
                cfg, items, [gen], [offsets], t0
            )
            online = list(gen.records)
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await sampler

    window = cfg.duration_s if offsets else max((c.finished_at for c in completions), default=0.0)
    latencies = gen.latencies()
    batch = batch_summary(completions, window, pool_size=len(items))
    # Salvaged runs are still runs — the throughput inside the window is real —
    # but the makespan and `drained` are not, so analysis has to be able to tell
    # them apart from a pool that finished on its own.
    batch["drain_timed_out"] = drain_timed_out
    return {
        "epoch0": epoch0,
        "window_s": window,
        "warmup": warmup,
        "online": {
            "requests": [r.as_dict() for r in online],
            "summary": summarize_latencies(latencies),
            "errors": sum(1 for r in online if not r.ok),
            "achieved_rps": (len(online) / window) if window > 0 else 0.0,
        },
        "batch": batch,
        "batch_completions": [c.as_dict() for c in completions],
        "metrics": [s.as_dict() for s in samples],
        "metrics_summary": summarize_metrics(samples),
        "tick_reports": tick_reports,
        "tick_summary": summarize_tick_reports(tick_reports),
    }


async def _measure_fleet(cfg: RunConfig, servers: Sequence[VllmServer] | None) -> dict:
    """Run ``technique_a_fleet`` against an already-healthy fleet.

    Structure of the document it returns: every key a single-engine result has,
    with fleet-wide meaning (``batch`` is the fleet's throughput, ``online`` is
    every replica's requests merged and each tagged with its ``replica``), plus
    a ``*_per_replica`` sibling for each so nothing is lost to the aggregate.
    The exception is ``metrics``/``metrics_summary``, which stay *one* engine's
    gauges — summing KV usage across engines would be meaningless — with the
    rest in ``metrics_per_replica``.
    """
    urls = list(cfg.replica_urls)
    phases = diurnal_phases(cfg.replicas, cfg.diurnal_phase_list)
    # Seeds differ per replica so the fleet is not N copies of one arrival
    # process merely shifted; the phase is what staggers the *shape*.
    offsets_list = [
        schedule(cfg.arrival, cfg.online_rps, cfg.duration_s, cfg.seed + i, phase=phase)
        for i, phase in enumerate(phases)
    ]
    items = make_batch_items(cfg.batch_items, model=cfg.model, max_tokens=cfg.batch_max_tokens)
    gens = [
        OnlineLoadGen(base_url=url, model=cfg.model, max_tokens=cfg.online_max_tokens, priority=0)
        for url in urls
    ]

    # Strictly before t0, and per engine: a cold replica's first-request costs
    # would otherwise be charged to whichever replica the fleet placed work on.
    warmup = {url: await run_warmup(cfg, base_url=url) for url in urls}

    t0 = time.perf_counter()
    epoch0 = time.time()
    samples: dict[str, list[MetricSample]] = {url: [] for url in urls}
    stop = asyncio.Event()
    samplers = [
        asyncio.create_task(
            sample_metrics(f"{url}/metrics", stop, samples[url], t0=t0), name=f"metrics-{i}"
        )
        for i, url in enumerate(urls)
    ]
    try:
        completions, tick_reports, replica_of = await _run_technique_a(
            cfg, items, gens, offsets_list, t0, replicas=urls, placement=cfg.fleet_placement
        )
    finally:
        stop.set()
        for sampler in samplers:
            with contextlib.suppress(Exception):
                await sampler

    window = (
        cfg.duration_s
        if any(offsets_list)
        else max((c.finished_at for c in completions), default=0.0)
    )

    online_records: list[dict] = []
    online_per_replica: dict[str, dict] = {}
    latencies: list[float] = []
    errors = 0
    for url, gen, phase in zip(urls, gens, phases, strict=True):
        online_records.extend({**r.as_dict(), "replica": url} for r in gen.records)
        latencies.extend(gen.latencies())
        replica_errors = sum(1 for r in gen.records if not r.ok)
        errors += replica_errors
        online_per_replica[url] = {
            "phase": phase,
            "requests": len(gen.records),
            "summary": summarize_latencies(gen.latencies()),
            "errors": replica_errors,
            "achieved_rps": (len(gen.records) / window) if window > 0 else 0.0,
        }

    batch = batch_summary(completions, window, pool_size=len(items))
    batch["drain_timed_out"] = False
    primary = urls[0]
    return {
        "epoch0": epoch0,
        "window_s": window,
        "warmup": warmup,
        "replicas": urls,
        "placement": cfg.fleet_placement,
        "online": {
            "requests": online_records,
            "summary": summarize_latencies(latencies),
            "errors": errors,
            "achieved_rps": (len(online_records) / window) if window > 0 else 0.0,
        },
        "online_per_replica": online_per_replica,
        "batch": batch,
        "batch_per_replica": batch_summary_per_replica(completions, window),
        "batch_completions": [c.as_dict() for c in completions],
        "replica_of": replica_of,
        "metrics": [s.as_dict() for s in samples[primary]],
        "metrics_summary": summarize_metrics(samples[primary]),
        "metrics_per_replica": {url: [s.as_dict() for s in rows] for url, rows in samples.items()},
        "metrics_summary_per_replica": {
            url: summarize_metrics(rows) for url, rows in samples.items()
        },
        "tick_reports": tick_reports,
        "tick_summary": summarize_tick_reports(tick_reports),
        "tick_per_replica": summarize_per_replica_ticks(tick_reports),
    }


async def _run_technique_a(
    cfg: RunConfig,
    items: Sequence[dict],
    gens: Sequence[OnlineLoadGen],
    offsets_list: Sequence[Sequence[float]],
    t0: float,
    *,
    replicas: Sequence[str] | None = None,
    placement: str = "fleet",
) -> tuple[list[BatchCompletion], list[dict], dict[str, str]]:
    """Batch through the gateway store + dispatcher; online straight to vLLM.

    The dispatcher is driven in-process rather than through ``tidal serve``: the
    HTTP surface adds nothing to the measurement (it is exercised by the
    integration tests) and this way every ``TickReport`` — the controller's
    entire observable state — lands in the result file.

    One code path serves both the single engine and the fleet. ``replicas``
    ``None`` is the single-engine run: one ``VllmClient`` at ``cfg.base_url``,
    one load generator, and the same decisions the condition has always made.
    A list of URLs swaps in a ``ReplicaSet`` and one generator per engine, all
    driven concurrently against a single store.

    Returns ``(completions, tick_reports, replica_of)`` where ``replica_of``
    maps ``custom_id`` to the engine that ran it (empty for a single engine).
    """
    from tidal.api.assembler import finalize_if_done
    from tidal.api.jsonl import parse_batch_input
    from tidal.dispatcher.loop import Dispatcher
    from tidal.dispatcher.vllm_client import ReplicaSet, VllmClient
    from tidal.store.interfaces import BatchStatus, ItemState, make_repository

    workdir = Path(tempfile.mkdtemp(prefix="tidal-eval-store-"))
    tcfg = TidalConfig.from_env()
    primary = replicas[0] if replicas else cfg.base_url
    tcfg.vllm_base_url = primary
    tcfg.vllm_metrics_url = f"{primary}/metrics"
    tcfg.vllm_replicas = ",".join(replicas) if replicas else ""
    tcfg.dsn = f"sqlite:///{workdir / 'tidal.db'}"
    tcfg.blob_dir = str(workdir / "blobs")
    tcfg.max_inflight = cfg.max_inflight
    tcfg.served_model = cfg.model

    repo = make_repository(tcfg.dsn, tcfg.blob_dir)
    content = _jsonl(items)
    source = repo.create_file("batch", "eval.jsonl", content)
    parsed = [tuple(line) for line in parse_batch_input(content, "/v1/chat/completions")]
    batch = repo.create_batch(
        source.id, "/v1/chat/completions", tcfg.completion_window_hours, {"eval": "1"}, parsed
    )

    client = ReplicaSet(tcfg) if replicas else VllmClient(tcfg)
    dispatcher = Dispatcher(
        tcfg,
        repo,
        client,
        finalize=lambda r, bid: finalize_if_done(r, bid),
        placement=placement,
    )
    reports: list[dict] = []
    stop = asyncio.Event()

    async def control() -> None:
        await dispatcher.startup()
        while not stop.is_set():
            report = await dispatcher.tick(datetime.now(UTC))
            reports.append(
                {
                    "t": time.perf_counter() - t0,
                    "target": report.target,
                    "submitted": report.submitted,
                    "inflight": report.inflight,
                    "escalated_priorities": dict(report.escalated_priorities),
                    "expired_batches": list(report.expired_batches),
                    "circuit_open": report.circuit_open,
                    "per_replica": {k: dict(v) for k, v in report.per_replica.items()},
                }
            )
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=tcfg.poll_interval_s)

    controller = asyncio.create_task(control(), name="dispatcher")
    try:
        await asyncio.gather(
            *(gen.run(offsets, t0=t0) for gen, offsets in zip(gens, offsets_list, strict=True))
        )
        # Let the batch drain past the online window so the pool's makespan is
        # a real number, bounded so a wedged engine cannot hang the harness.
        deadline = time.perf_counter() + cfg.drain_timeout_s
        while time.perf_counter() < deadline:
            record = await asyncio.to_thread(repo.get_batch, batch.id)
            if record is not None and record.status in (
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.EXPIRED,
                BatchStatus.CANCELLED,
            ):
                break
            await asyncio.sleep(1.0)
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await controller
        with contextlib.suppress(Exception):
            await dispatcher.shutdown()
        with contextlib.suppress(Exception):
            await client.aclose()

    # Completion times come from the store, so they are stamped with the
    # dispatcher's *tick* clock, not the moment the HTTP response landed: they
    # are accurate to one `poll_interval_s`, which is all the throughput
    # metrics need. Per-item latency is genuinely not recoverable this way, so
    # it is reported as `nan` (null in the JSON) rather than invented — the
    # direct-submission conditions measure it properly and are the ones to
    # compare item latency across.
    epoch_t0 = time.time() - (time.perf_counter() - t0)
    replica_of = dict(dispatcher.replica_of)
    completions = []
    for item in await asyncio.to_thread(repo.list_items, batch.id, None):
        if item.state is not ItemState.SUCCEEDED or item.finished_at is None:
            continue
        finished = item.finished_at.timestamp() - epoch_t0
        started = item.submitted_at.timestamp() - epoch_t0 if item.submitted_at else finished
        completions.append(
            BatchCompletion(
                index=item.line_no,
                custom_id=item.custom_id,
                started_at=started,
                finished_at=finished,
                latency_s=(finished - started) if item.submitted_at else float("nan"),
                prompt_tokens=item.usage_prompt_tokens or 0,
                completion_tokens=item.usage_completion_tokens or 0,
                replica=replica_of.get(item.custom_id) if replicas else None,
            )
        )
    completions.sort(key=lambda c: c.index)
    return completions, reports, replica_of


def _jsonable(value):
    """Replace non-finite floats with ``None`` so the file is *strict* JSON.

    ``json.dumps`` happily writes bare ``NaN``/``Infinity``, which Python reads
    back but ``jq``, JavaScript and most other consumers reject. An empty
    percentile is genuinely "no value", so null is also the honest encoding.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _jsonl(items: Sequence[dict]) -> bytes:
    lines = [
        json.dumps(
            {
                "custom_id": f"item-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
        for i, body in enumerate(items)
    ]
    return ("\n".join(lines) + "\n").encode()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def run(
    condition: str = typer.Option(..., help=f"One of: {', '.join(CONDITIONS)}"),
    minutes: float = typer.Option(10.0, help="Length of the online window."),
    online_rps: float = typer.Option(
        1.0, help="Online arrival rate (peak rate when --arrival diurnal)."
    ),
    batch_items: int = typer.Option(400, help="Size of the batch pool."),
    out: str = typer.Option("results/run.json", help="Where to write the result JSON."),
    arrival: str = typer.Option("poisson", help="Arrival shape: poisson | diurnal."),
    seed: int = typer.Option(7, help="Arrival RNG seed — keep it fixed across conditions."),
    port: int = typer.Option(8400, help="Port for the vLLM this run owns."),
    model: str = typer.Option(DEFAULT_MODEL, help="Served model."),
    online_max_tokens: int = typer.Option(16, help="max_tokens for online requests."),
    batch_max_tokens: int = typer.Option(16, help="max_tokens for batch items."),
    max_inflight: int = typer.Option(4, help="technique_a: dispatcher concurrency ceiling."),
    vllm_bin: str = typer.Option(DEFAULT_VLLM_BIN, help="Path to the `vllm` executable."),
    drain_timeout_s: float = typer.Option(300.0, help="Cap on the post-window batch drain."),
    log_dir: str = typer.Option("", help="Directory for engine logs (default: a temp dir)."),
    gpu_preset: bool = typer.Option(
        False,
        "--gpu-preset",
        help=(
            "Flip the defaults to GPU-sane: --no-enforce-eager, --no-hf-offline, "
            "--batch-concurrency 512. Individual flags still win."
        ),
    ),
    max_model_len: int = typer.Option(DEFAULT_MAX_MODEL_LEN, help="Engine --max-model-len."),
    tensor_parallel: int = typer.Option(
        DEFAULT_TENSOR_PARALLEL, help="Engine --tensor-parallel-size (>1 shards across GPUs)."
    ),
    enforce_eager: bool | None = typer.Option(
        None,
        "--enforce-eager/--no-enforce-eager",
        help="Eager mode + TORCHDYNAMO_DISABLE. Default: on (off under --gpu-preset).",
    ),
    hf_offline: bool | None = typer.Option(
        None,
        "--hf-offline/--no-hf-offline",
        help="HF_HUB_OFFLINE. Default: on (off under --gpu-preset, so a cold node can fetch).",
    ),
    batch_concurrency: int | None = typer.Option(
        None,
        help="HTTP pool size for direct batch submission. Default: 100 (512 under --gpu-preset).",
    ),
    warmup_s: float | None = typer.Option(
        None,
        "--warmup-s",
        help=(
            "Seconds of unmeasured sequential requests after /health and before t0, "
            f"capped at {WARMUP_MAX_REQUESTS} requests. Default: 0 (45 under --gpu-preset)."
        ),
    ),
    replicas: int = typer.Option(2, help="technique_a_fleet: how many engines the fleet has."),
    fleet_base_port: int = typer.Option(
        FLEET_BASE_PORT, help="technique_a_fleet: replica i listens on this port + i."
    ),
    fleet_placement: str = typer.Option(
        "fleet",
        help=(
            "technique_a_fleet batch placement: 'fleet' (load-aware) or "
            "'pinned' (everything on replica 0 — the control arm)."
        ),
    ),
    diurnal_phase_list: str = typer.Option(
        "",
        help=(
            "technique_a_fleet: comma-separated diurnal phase per replica, e.g. "
            "'0,3.14159'. Default: spread evenly over the period."
        ),
    ),
) -> None:
    """Run one condition end to end and write ``<out>``."""
    engine = resolve_preset(
        gpu_preset=gpu_preset,
        enforce_eager=enforce_eager,
        hf_offline=hf_offline,
        batch_concurrency=batch_concurrency,
        warmup_s=warmup_s,
    )
    cfg = RunConfig(
        condition=condition,
        minutes=minutes,
        online_rps=online_rps,
        batch_items=batch_items,
        arrival=arrival,
        seed=seed,
        model=model,
        port=port,
        online_max_tokens=online_max_tokens,
        batch_max_tokens=batch_max_tokens,
        max_inflight=max_inflight,
        vllm_bin=vllm_bin,
        drain_timeout_s=drain_timeout_s,
        out=out,
        max_model_len=max_model_len,
        tensor_parallel=tensor_parallel,
        enforce_eager=engine["enforce_eager"],
        hf_offline=engine["hf_offline"],
        batch_concurrency=engine["batch_concurrency"],
        warmup_s=engine["warmup_s"],
        replicas=replicas,
        fleet_base_port=fleet_base_port,
        fleet_placement=fleet_placement,
        diurnal_phase_list=diurnal_phase_list,
    )
    result = asyncio.run(run_condition(cfg, log_dir=log_dir or None))

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_jsonable(result), indent=2, default=str))

    online = result["online"]["summary"]
    batch = result["batch"]
    typer.echo(f"[{condition}] wrote {destination}")
    typer.echo(
        f"  online  n={online['count']} p50={online['p50']:.3f}s "
        f"p95={online['p95']:.3f}s p99={online['p99']:.3f}s errors={result['online']['errors']}"
    )
    typer.echo(
        f"  batch   completed={batch['completed']}/{batch['pool_size']} "
        f"in_window={batch['completed_in_window']} "
        f"tok/s={batch['output_tokens_per_s']:.1f} makespan={batch['makespan_s']:.1f}s"
    )
    if batch["drained"]:
        typer.echo(
            "  NOTE: the batch pool drained inside the window — raise --batch-items "
            "so the batch tier is never idle, otherwise throughput is under-reported."
        )
    if batch.get("drain_timed_out"):
        typer.echo(
            "  NOTE: the drain timed out — these are the completions salvaged from a "
            "pool that never finished. In-window throughput stands; makespan does not."
        )
    if result.get("tidal_stats") is not None:
        typer.echo(f"  tidal   {len(result['tidal_stats'])} scheduler telemetry lines")


@app.callback()
def _main() -> None:  # pragma: no cover - typer plumbing
    """Tidal A/B evaluation harness."""


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
