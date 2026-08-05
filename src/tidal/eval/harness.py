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
teardown of the whole process group), because a condition is only meaningful
against a *fresh* engine: prefix cache, KV pool and the latency model's fit
must not carry over between conditions.
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
from collections.abc import Sequence
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
    "BatchCompletion",
    "MetricSample",
    "RunConfig",
    "VllmServer",
    "app",
    "batch_summary",
    "make_batch_items",
    "parse_tidal_stats",
    "percent_of_ceiling",
    "run_condition",
    "summarize_metrics",
    "summarize_tick_reports",
    "write_logging_config",
]

CONDITIONS = ("online_only", "offline_only", "naive", "technique_a", "technique_b")

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

app = typer.Typer(add_completion=False, help="Tidal A/B evaluation harness.")


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
        env_extra: additional environment (``TIDAL_*`` for the scheduler).

    Started with ``start_new_session=True`` so teardown can signal the whole
    process group — vLLM spawns an EngineCore child, and killing only the
    parent leaves a process holding the port.
    """

    port: int
    flavor: str = "stock"
    log_path: str | None = None
    model: str = DEFAULT_MODEL
    max_model_len: int = 4096
    vllm_bin: str = DEFAULT_VLLM_BIN
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
            "--enforce-eager",
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
                # Inductor is a non-starter on this CPU build.
                "TORCHDYNAMO_DISABLE": "1",
                "VLLM_CPU_KVCACHE_SPACE": env.get("VLLM_CPU_KVCACHE_SPACE", "4"),
                "HF_HUB_OFFLINE": env.get("HF_HUB_OFFLINE", "1"),
                "PYTHONUNBUFFERED": "1",
            }
        )
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
    """One batch item's outcome, timed against the run's ``t0``."""

    index: int
    custom_id: str
    started_at: float
    finished_at: float
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return asdict(self)


async def submit_batch_direct(
    base_url: str,
    items: Sequence[dict],
    *,
    t0: float,
    priority: int = BATCH_PRIORITY,
    concurrency: int | None = None,
    timeout_s: float = 900.0,
    client: httpx.AsyncClient | None = None,
) -> list[BatchCompletion]:
    """Fire the whole pool straight at vLLM (the ``naive``/``technique_b`` path).

    ``concurrency=None`` means *all at once*, which is exactly the point of the
    naive condition: no client-side admission control whatsoever.
    """
    owns = client is None
    http = client or httpx.AsyncClient(timeout=timeout_s)
    gate = asyncio.Semaphore(concurrency) if concurrency else None

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
                timeout=timeout_s,
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
        return BatchCompletion(
            index=index,
            custom_id=f"item-{index}",
            started_at=began - t0,
            finished_at=ended - t0,
            latency_s=ended - began,
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            error=error,
        )

    try:
        return list(await asyncio.gather(*(one(i, b) for i, b in enumerate(items))))
    finally:
        if owns:
            await http.aclose()


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

    @property
    def duration_s(self) -> float:
        return self.minutes * 60.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


async def run_condition(cfg: RunConfig, *, log_dir: str | None = None) -> dict:
    """Launch the right engine, run the condition, return the result document."""
    if cfg.condition not in CONDITIONS:
        raise ValueError(f"unknown condition {cfg.condition!r}; want one of {CONDITIONS}")

    flavor = "tidal" if cfg.condition == "technique_b" else "stock"
    log_dir_path = Path(log_dir or tempfile.mkdtemp(prefix="tidal-eval-"))
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / f"vllm-{cfg.condition}.log"
    logging_config = write_logging_config(log_dir_path / "logging.json")

    server = VllmServer(
        port=cfg.port,
        flavor=flavor,
        log_path=str(log_path),
        model=cfg.model,
        vllm_bin=cfg.vllm_bin,
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

    t0 = time.perf_counter()
    epoch0 = time.time()
    samples: list[MetricSample] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(
        sample_metrics(f"{cfg.base_url}/metrics", stop, samples, t0=t0), name="metrics"
    )

    completions: list[BatchCompletion] = []
    tick_reports: list[dict] = []
    online: list[RequestRecord] = []
    try:
        if cfg.condition == "online_only":
            online = await gen.run(offsets, t0=t0)
        elif cfg.condition == "offline_only":
            completions = await submit_batch_direct(cfg.base_url, items, t0=t0)
        elif cfg.condition in ("naive", "technique_b"):
            batch_task = asyncio.create_task(
                submit_batch_direct(cfg.base_url, items, t0=t0), name="batch"
            )
            online = await gen.run(offsets, t0=t0)
            completions = await asyncio.wait_for(batch_task, timeout=cfg.drain_timeout_s)
        else:  # technique_a
            completions, tick_reports = await _run_technique_a(cfg, items, gen, offsets, t0)
            online = list(gen.records)
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await sampler

    window = cfg.duration_s if offsets else max((c.finished_at for c in completions), default=0.0)
    latencies = gen.latencies()
    return {
        "epoch0": epoch0,
        "window_s": window,
        "online": {
            "requests": [r.as_dict() for r in online],
            "summary": summarize_latencies(latencies),
            "errors": sum(1 for r in online if not r.ok),
            "achieved_rps": (len(online) / window) if window > 0 else 0.0,
        },
        "batch": batch_summary(completions, window, pool_size=len(items)),
        "batch_completions": [c.as_dict() for c in completions],
        "metrics": [s.as_dict() for s in samples],
        "metrics_summary": summarize_metrics(samples),
        "tick_reports": tick_reports,
        "tick_summary": summarize_tick_reports(tick_reports),
    }


async def _run_technique_a(
    cfg: RunConfig,
    items: Sequence[dict],
    gen: OnlineLoadGen,
    offsets: Sequence[float],
    t0: float,
) -> tuple[list[BatchCompletion], list[dict]]:
    """Batch through the gateway store + dispatcher; online straight to vLLM.

    The dispatcher is driven in-process rather than through ``tidal serve``: the
    HTTP surface adds nothing to the measurement (it is exercised by the
    integration tests) and this way every ``TickReport`` — the controller's
    entire observable state — lands in the result file.
    """
    from tidal.api.assembler import finalize_if_done
    from tidal.api.jsonl import parse_batch_input
    from tidal.dispatcher.loop import Dispatcher
    from tidal.dispatcher.vllm_client import VllmClient
    from tidal.store.interfaces import BatchStatus, ItemState, make_repository

    workdir = Path(tempfile.mkdtemp(prefix="tidal-eval-store-"))
    tcfg = TidalConfig.from_env()
    tcfg.vllm_base_url = cfg.base_url
    tcfg.vllm_metrics_url = f"{cfg.base_url}/metrics"
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

    client = VllmClient(tcfg)
    dispatcher = Dispatcher(tcfg, repo, client, finalize=lambda r, bid: finalize_if_done(r, bid))
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
                }
            )
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=tcfg.poll_interval_s)

    controller = asyncio.create_task(control(), name="dispatcher")
    try:
        await gen.run(offsets, t0=t0)
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
            )
        )
    completions.sort(key=lambda c: c.index)
    return completions, reports


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
) -> None:
    """Run one condition end to end and write ``<out>``."""
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
    if result.get("tidal_stats") is not None:
        typer.echo(f"  tidal   {len(result['tidal_stats'])} scheduler telemetry lines")


@app.callback()
def _main() -> None:  # pragma: no cover - typer plumbing
    """Tidal A/B evaluation harness."""


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
