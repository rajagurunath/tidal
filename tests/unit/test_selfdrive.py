"""Tests for the SSH-less CaaS supervisor (``tidal.eval.selfdrive``).

No Docker, no GPU and no child process: the eval script is replaced by a stub
handle whose exit code the test controls, and "the script made progress" is
expressed the way the real script expresses it — by appending ``PROGRESS``
lines to the run log. That is the whole contract between
``deploy/run_gpu_eval.sh`` and this server, so it is what gets tested.
"""

from __future__ import annotations

import asyncio
import signal
import tarfile
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from tidal.eval.selfdrive import (
    DEFAULT_TAIL,
    MAX_TAIL,
    TERMINAL_PHASES,
    SelfDriveConfig,
    Supervisor,
    build_config,
    create_app,
    safe_join,
    tail_lines,
)

BASE = "http://selfdrive.test"
KEY = "s3cret-key"


class StubChild:
    """A ChildHandle the test drives by hand."""

    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.signals: list[int] = []

    def poll(self) -> int | None:
        return self.returncode

    def signal_group(self, sig: int) -> None:
        self.signals.append(sig)

    def exit(self, code: int) -> None:
        self.returncode = code


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    p = tmp_path / "run.log"
    p.write_text("")
    return p


@pytest.fixture
def child() -> StubChild:
    return StubChild()


@pytest.fixture
def cfg(tmp_path: Path, results_dir: Path, log_path: Path) -> SelfDriveConfig:
    return SelfDriveConfig(
        results_dir=results_dir,
        log_path=log_path,
        repo_dir=tmp_path,
        eval_script=tmp_path / "run_gpu_eval.sh",
        api_key=KEY,
    )


@pytest.fixture
def sup(cfg: SelfDriveConfig, child: StubChild) -> Supervisor:
    supervisor = Supervisor(cfg, launcher=lambda: child)
    supervisor.start()
    return supervisor


@pytest_asyncio.fixture
async def client(sup: Supervisor) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(sup)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE) as ac:
        yield ac


def emit(log_path: Path, *lines: str) -> None:
    """Append log lines the way the eval script would."""
    with log_path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


# --------------------------------------------------------------------------
# liveness + index
# --------------------------------------------------------------------------
async def test_healthz_is_200_regardless_of_phase(client, log_path, child):
    child.exit(1)
    emit(log_path, "PROGRESS failed everything is on fire")
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_index_advertises_the_endpoints(client):
    body = (await client.get("/")).json()
    assert any("/results.tar.gz" in e for e in body["endpoints"])
    assert body["phase"] == "booting"


# --------------------------------------------------------------------------
# status derivation
# --------------------------------------------------------------------------
async def test_status_starts_booting(client, sup):
    body = (await client.get("/status")).json()
    assert body["phase"] == "booting"
    assert body["canary"] == "pending"
    assert body["started_at"] is not None
    assert body["elapsed_s"] >= 0
    assert body["pid"] == 4242


async def test_phase_follows_progress_lines(client, log_path):
    emit(log_path, "PROGRESS booting", "[12:00:00] run dir: /results/gpu-x")
    assert (await client.get("/status")).json()["phase"] == "booting"

    emit(log_path, "PROGRESS canary")
    assert (await client.get("/status")).json()["phase"] == "canary"

    emit(log_path, "PROGRESS canary_ok", "PROGRESS probing")
    body = (await client.get("/status")).json()
    assert body["phase"] == "probing"
    assert body["canary"] == "ok"

    emit(log_path, "PROGRESS running:technique_a")
    assert (await client.get("/status")).json()["phase"] == "running:technique_a"

    emit(log_path, "PROGRESS rendering")
    assert (await client.get("/status")).json()["phase"] == "rendering"


async def test_done_only_once_the_child_exits(client, log_path, child):
    emit(log_path, "PROGRESS done")
    # The script printed `done` but has not exited: still packing, maybe.
    assert (await client.get("/status")).json()["phase"] == "done"

    child.exit(0)
    body = (await client.get("/status")).json()
    assert body["phase"] == "done"
    assert body["exit_code"] == 0
    assert body["results_ready"] is True
    assert body["finished_at"] is not None


async def test_nonzero_exit_is_failed_with_an_error(client, child):
    child.exit(137)
    body = (await client.get("/status")).json()
    assert body["phase"] == "failed"
    assert "137" in body["error"]
    assert body["results_ready"] is True


async def test_failed_progress_line_wins_over_a_clean_exit(client, log_path, child):
    emit(log_path, "PROGRESS failed the sizing probe never produced a usable result")
    child.exit(0)
    body = (await client.get("/status")).json()
    assert body["phase"] == "failed"
    assert "sizing probe" in body["error"]


async def test_canary_failure_is_sticky_and_visible(client, log_path, child):
    emit(log_path, "PROGRESS canary", "PROGRESS canary_failed vLLM drifted; technique_b skipped")
    emit(log_path, "PROGRESS running:technique_a")

    body = (await client.get("/status")).json()
    # technique_a keeps running, but the error is already on the wire.
    assert body["phase"] == "running:technique_a"
    assert body["canary"] == "failed"
    assert "technique_b skipped" in body["error"]

    child.exit(1)
    body = (await client.get("/status")).json()
    assert body["phase"] == "failed"
    assert "technique_b skipped" in body["error"]


async def test_condition_counts_come_from_progress_lines(client, log_path):
    emit(
        log_path,
        "PROGRESS total 8",
        "PROGRESS case_done probe_offline",
        "PROGRESS case_done online_only",
        "PROGRESS case_done online_only",  # idempotent
        "PROGRESS case_failed naive",
    )
    conditions = (await client.get("/status")).json()["conditions"]
    assert conditions["total"] == 8
    assert conditions["completed"] == 2
    assert conditions["done"] == ["probe_offline", "online_only"]
    assert conditions["failed"] == ["naive"]


async def test_completed_falls_back_to_result_jsons(client, results_dir):
    (results_dir / "matrix").mkdir()
    for name in ("online_only", "naive"):
        (results_dir / "matrix" / f"{name}.json").write_text('{"condition": "x"}')
    (results_dir / "matrix" / "figures").mkdir()
    (results_dir / "matrix" / "figures" / "plot.json").write_text("{}")

    conditions = (await client.get("/status")).json()["conditions"]
    assert conditions["completed"] == 2  # figures/ does not count


async def test_partial_log_line_is_not_parsed_until_complete(client, log_path):
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("PROGRESS running:tech")  # no newline yet
    assert (await client.get("/status")).json()["phase"] == "booting"

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("nique_b\n")
    assert (await client.get("/status")).json()["phase"] == "running:technique_b"


async def test_engine_shape_read_from_manifest(client, results_dir):
    run = results_dir / "gpu-node7-20260807T101500Z"
    run.mkdir()
    (run / "MANIFEST.txt").write_text(
        "tidal gpu eval\n"
        "host:              node7\n"
        "model:             Qwen/Qwen2.5-7B-Instruct\n"
        "tensor_parallel:   8\n"
        "max_model_len:     8192\n"
        "canary:            ok\n"
        "--- nvidia-smi ---\n"
        "0, NVIDIA H200, 143771 MiB\n"
    )
    engine = (await client.get("/status")).json()["engine"]
    assert engine["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert engine["tensor_parallel"] == "8"
    assert engine["canary"] == "ok"
    assert "MANIFEST.txt" in engine["source"]


async def test_engine_shape_falls_back_to_env(client, monkeypatch):
    monkeypatch.setenv("MODEL", "Qwen/Qwen3-32B")
    monkeypatch.setenv("TENSOR_PARALLEL_SIZE", "4")
    engine = (await client.get("/status")).json()["engine"]
    assert engine["model"] == "Qwen/Qwen3-32B"
    assert engine["tensor_parallel"] == "4"
    assert "not written yet" in engine["source"]


async def test_status_carries_the_last_log_lines(client, log_path):
    emit(log_path, *[f"line {i}" for i in range(40)])
    body = (await client.get("/status")).json()
    assert len(body["log_tail"]) == 20
    assert body["log_tail"][-1] == "line 39"


# --------------------------------------------------------------------------
# /log
# --------------------------------------------------------------------------
async def test_log_tail_defaults_and_caps(client, log_path):
    emit(log_path, *[f"line {i}" for i in range(DEFAULT_TAIL + 120)])

    default_lines = (await client.get("/log")).text.splitlines()
    assert len(default_lines) == DEFAULT_TAIL
    assert default_lines[-1] == f"line {DEFAULT_TAIL + 119}"

    assert len((await client.get("/log", params={"tail": 5})).text.splitlines()) == 5

    # Over the cap is clamped, not rejected; the file is shorter than the cap.
    huge = (await client.get("/log", params={"tail": MAX_TAIL * 10})).text.splitlines()
    assert len(huge) == DEFAULT_TAIL + 121  # +1 for the supervisor's start note


def test_tail_lines_handles_a_missing_file(tmp_path):
    assert tail_lines(tmp_path / "nope.log", 10) == []


# --------------------------------------------------------------------------
# /results
# --------------------------------------------------------------------------
async def test_results_listing(client, results_dir):
    (results_dir / "matrix").mkdir()
    (results_dir / "matrix" / "naive.json").write_text('{"condition": "naive"}')
    (results_dir / "MANIFEST.txt").write_text("tidal gpu eval\n")

    body = (await client.get("/results/")).json()
    names = [f["name"] for f in body["files"]]
    assert names == ["MANIFEST.txt", "matrix/naive.json"]
    assert body["count"] == 2
    assert body["total_bytes"] == sum(f["size"] for f in body["files"])


async def test_results_file_is_served(client, results_dir):
    (results_dir / "MANIFEST.txt").write_text("tidal gpu eval\nhost: node7\n")
    resp = await client.get("/results/MANIFEST.txt")
    assert resp.status_code == 200
    assert "node7" in resp.text


async def test_results_nested_file_is_served(client, results_dir):
    (results_dir / "matrix").mkdir()
    (results_dir / "matrix" / "naive.json").write_text('{"condition": "naive"}')
    resp = await client.get("/results/matrix/naive.json")
    assert resp.status_code == 200
    assert resp.json()["condition"] == "naive"


async def test_results_traversal_is_refused(client, tmp_path, results_dir):
    (tmp_path / "secret.txt").write_text("do not serve me")
    # %2F survives httpx's URL handling and is decoded into the path param, so
    # the handler really does see `../secret.txt` — this is a genuine traversal
    # attempt reaching safe_join, not a request the router normalized away.
    for path in ("/results/..%2Fsecret.txt", "/results/%2e%2e%2fsecret.txt"):
        resp = await client.get(path)
        assert resp.status_code == 404
        assert resp.json()["error"] == "no such result file: ../secret.txt"
        assert "do not serve me" not in resp.text


async def test_results_absolute_path_is_refused(client):
    resp = await client.get("/results//etc/passwd")
    assert resp.status_code == 404
    assert resp.json()["error"] == "no such result file: /etc/passwd"


def test_safe_join_rejects_escapes(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "ok.txt").write_text("x")
    (tmp_path / "secret.txt").write_text("x")

    assert safe_join(root, "ok.txt") == (root / "ok.txt").resolve()
    assert safe_join(root, "../secret.txt") is None
    assert safe_join(root, "a/../../secret.txt") is None
    assert safe_join(root, "/etc/passwd") is None
    assert safe_join(root, "") is None
    assert safe_join(root, ".") is None


# --------------------------------------------------------------------------
# /results.tar.gz
# --------------------------------------------------------------------------
async def test_tarball_404s_with_status_while_running(client, results_dir):
    (results_dir / "MANIFEST.txt").write_text("x\n")
    resp = await client.get("/results.tar.gz")
    assert resp.status_code == 404
    assert resp.json()["status"]["phase"] == "booting"


async def test_tarball_is_built_at_request_time_when_done(client, results_dir, child, tmp_path):
    (results_dir / "matrix").mkdir()
    (results_dir / "matrix" / "naive.json").write_text('{"condition": "naive"}')
    child.exit(0)

    resp = await client.get("/results.tar.gz")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/gzip"
    assert "filename=" in resp.headers["content-disposition"]

    out = tmp_path / "downloaded.tar.gz"
    out.write_bytes(resp.content)
    with tarfile.open(out) as tar:
        assert "results/matrix/naive.json" in tar.getnames()


async def test_tarball_prefers_the_one_the_script_packed(client, results_dir, child):
    packed = results_dir / "tidal-gpu-node7-20260807T101500Z.tar.gz"
    with tarfile.open(packed, "w:gz") as tar:
        member = results_dir / "MANIFEST.txt"
        member.write_text("packed by the script\n")
        tar.add(member, arcname="run/MANIFEST.txt")
    child.exit(0)

    resp = await client.get("/results.tar.gz")
    assert resp.status_code == 200
    assert packed.name in resp.headers["content-disposition"]


async def test_tarball_available_after_an_aborted_run(client, results_dir, child):
    (results_dir / "MANIFEST.txt").write_text("partial\n")
    await client.post("/abort", headers={"X-Tidal-Key": KEY})
    child.exit(143)
    resp = await client.get("/results.tar.gz")
    assert resp.status_code == 200


# --------------------------------------------------------------------------
# /abort
# --------------------------------------------------------------------------
async def test_abort_requires_the_key(client, child):
    assert (await client.post("/abort")).status_code == 401
    assert (await client.post("/abort", headers={"X-Tidal-Key": "wrong"})).status_code == 401
    assert child.signals == []


async def test_abort_signals_the_process_group(client, child):
    resp = await client.post("/abort", headers={"X-Tidal-Key": KEY})
    assert resp.status_code == 200
    assert child.signals == [signal.SIGTERM]
    assert resp.json()["status"]["phase"] == "aborted"
    # Still aborted after the child actually dies, and still serving.
    child.exit(-15)
    assert (await client.get("/status")).json()["phase"] == "aborted"
    assert (await client.get("/healthz")).status_code == 200


async def test_abort_is_refused_when_no_key_is_configured(cfg, child):
    cfg.api_key = ""
    supervisor = Supervisor(cfg, launcher=lambda: child)
    supervisor.start()
    transport = httpx.ASGITransport(app=create_app(supervisor))
    async with httpx.AsyncClient(transport=transport, base_url=BASE) as ac:
        resp = await ac.post("/abort", headers={"X-Tidal-Key": "anything"})
    assert resp.status_code == 401
    assert "TIDAL_API_KEY" in resp.json()["error"]
    assert child.signals == []


# --------------------------------------------------------------------------
# the real child: no stub, an actual process group
# --------------------------------------------------------------------------
async def wait_for_exit(supervisor: Supervisor, timeout: float = 20.0) -> dict:
    """Poll /status until the child is reaped. Racy-by-construction otherwise."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = supervisor.status()
        if st["exit_code"] is not None and st["phase"] in TERMINAL_PHASES:
            return st
        await asyncio.sleep(0.05)
    raise AssertionError(f"child never exited: {supervisor.status()}")


async def test_real_child_runs_and_reports_done(tmp_path, results_dir, log_path):
    script = tmp_path / "fake_eval.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'echo "PROGRESS booting"\n'
        # Proves RESULTS_DIR is exported to the child, which is the one piece of
        # plumbing that decides whether /results/ can see anything at all.
        'printf "host: node7\\n" > "$RESULTS_DIR/MANIFEST.txt"\n'
        'echo "PROGRESS case_done probe_offline"\n'
        'echo "PROGRESS done"\n'
    )
    cfg = SelfDriveConfig(
        results_dir=results_dir, log_path=log_path, repo_dir=tmp_path, eval_script=script
    )
    supervisor = Supervisor(cfg)
    supervisor.start()

    st = await wait_for_exit(supervisor)
    assert st["phase"] == "done"
    assert st["exit_code"] == 0
    assert st["conditions"]["done"] == ["probe_offline"]
    assert (results_dir / "MANIFEST.txt").read_text() == "host: node7\n"
    assert supervisor.ensure_tarball() is not None


async def test_real_child_group_dies_on_abort(tmp_path, results_dir, log_path):
    script = tmp_path / "slow_eval.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'echo "PROGRESS running:technique_a"\n'
        "sleep 120\n"  # a grandchild, so this only dies if the group is signalled
    )
    cfg = SelfDriveConfig(
        results_dir=results_dir, log_path=log_path, repo_dir=tmp_path, eval_script=script
    )
    supervisor = Supervisor(cfg)
    supervisor.start()
    assert supervisor.abort() is True

    st = await wait_for_exit(supervisor)
    assert st["phase"] == "aborted"
    assert st["exit_code"] is not None


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def test_build_config_reads_the_container_env():
    cfg = build_config(
        {
            "RESULTS_DIR": "/results",
            "PORT": "8000",
            "TIDAL_API_KEY": "abc",
            "EVAL_SCRIPT": "/usr/local/bin/tidal-run-gpu-eval",
            "HF_TOKEN": "hf_x",
            "TIDAL_RESULTS_HF_REPO": "me/tidal-results",
        }
    )
    assert cfg.results_dir == Path("/results")
    assert cfg.log_path == Path("/results/selfdrive-run.log")
    assert cfg.port == 8000
    assert cfg.api_key == "abc"
    assert cfg.hf_repo == "me/tidal-results"
    assert cfg.hf_token == "hf_x"


def test_build_config_defaults(tmp_path):
    cfg = build_config({"REPO_DIR": str(tmp_path)})
    assert cfg.results_dir == Path("/results")
    assert cfg.port == 8000
    assert cfg.api_key == ""
    # No installed entrypoint in a checkout: fall back to the repo copy.
    assert cfg.eval_script.name == "run_gpu_eval.sh"
