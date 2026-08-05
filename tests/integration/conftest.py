"""Live-vLLM fixtures for the end-to-end tests (plan A8 / B4).

These tests boot a real vLLM OpenAI server on this machine and drive real
traffic through it. They are off by default and gated on ``TIDAL_IT``:

    TIDAL_IT=1 .venv/bin/python -m pytest tests/integration -m integration

Ground rules baked into the fixtures, all learned the hard way on a CPU box:

* **One engine at a time.** A 0.5B model on CPU is already the whole machine;
  two servers would make every latency number meaningless. ``engine`` is a
  session-scoped *factory* that owns at most one server and swaps flavors
  (``"stock"`` / ``"tidal"``) on demand, tearing the previous one down first.
* **Kill leaks first.** A run that died mid-test leaves a server holding the
  port; the session start reaps anything listening on the ports we own.
* **Kill the process group.** vLLM V1 forks an EngineCore child. Signalling
  only the parent leaves an orphan pinning the port and the CPU.
* **Be patient.** Model load is 50-90 s here; every timeout is generous, and
  the failure message carries the engine log tail so a timeout is debuggable
  without re-running.

The engine launcher itself lives in :mod:`tidal.eval.harness` — the harness has
to launch the same servers the same way, and two copies of that knowledge would
drift.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from tidal.eval.harness import VllmServer, write_logging_config

#: Ports this suite owns end to end; anything found on them at session start is
#: a leak from an earlier run and is reaped.
OWNED_PORTS = (8397, 8398, 8399)
ENGINE_PORT = 8397

#: Model load on this box is ~50-90 s; leave room for a cold page cache.
READY_TIMEOUT_S = float(os.environ.get("TIDAL_IT_READY_TIMEOUT_S", "300"))


def integration_enabled() -> bool:
    return os.environ.get("TIDAL_IT", "0").lower() not in ("0", "", "false", "no")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("integration") and not integration_enabled():
        pytest.skip("live-vLLM integration tests are off; set TIDAL_IT=1 to run them")


def _reap_owned_ports() -> None:
    """SIGKILL whatever is listening on our ports (a leak from a failed run)."""
    lsof = shutil.which("lsof")
    if lsof is None:  # pragma: no cover - macOS/Linux dev boxes have it
        return
    for port in OWNED_PORTS:
        try:
            found = subprocess.run(
                [lsof, "-ti", f"tcp:{port}"], capture_output=True, text=True, timeout=10
            )
        except (subprocess.SubprocessError, OSError):
            continue
        for pid in found.stdout.split():
            with_pid = int(pid)
            try:
                os.kill(with_pid, 9)
            except (ProcessLookupError, PermissionError):
                continue
    # Give the kernel a moment to release the sockets.
    time.sleep(1.0)


@pytest.fixture(scope="session")
def it_workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("tidal-it")


@pytest.fixture(scope="session")
def engine(it_workdir):
    """Factory: ``engine("stock")`` / ``engine("tidal")`` → a ready server.

    Yields a callable rather than a parameterized fixture so the "one engine at
    a time" invariant is enforced here, in code, instead of relying on pytest's
    fixture-cache eviction order.
    """
    if not integration_enabled():
        pytest.skip("live-vLLM integration tests are off; set TIDAL_IT=1 to run them")

    _reap_owned_ports()
    logging_config = write_logging_config(it_workdir / "logging.json")
    state: dict[str, VllmServer] = {}

    def get(flavor: str = "stock", **env_extra: str) -> VllmServer:
        wanted = {"VLLM_LOGGING_CONFIG_PATH": logging_config, **env_extra}
        current = state.get("server")
        # Config lives in the engine's environment, so a different env is a
        # different engine: reuse only on an exact match.
        if current is not None and current.flavor == flavor and current.env_extra == wanted:
            return current
        if current is not None:
            current.stop()
            state.pop("server", None)
            time.sleep(1.0)
        server = VllmServer(
            port=ENGINE_PORT,
            flavor=flavor,
            log_path=str(it_workdir / f"vllm-{flavor}.log"),
            env_extra=wanted,
        )
        started = time.monotonic()
        server.start()
        server.wait_ready(READY_TIMEOUT_S)
        print(
            f"\n[it] {flavor} vLLM ready on {server.base_url} in {time.monotonic() - started:.0f}s"
        )
        state["server"] = server
        return server

    try:
        yield get
    finally:
        server = state.pop("server", None)
        if server is not None:
            server.stop()
        _reap_owned_ports()


@pytest.fixture
async def engine_http():
    """A long-timeout HTTP client — a CPU decode can genuinely take minutes."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        yield client
