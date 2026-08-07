"""Self-driving evaluation supervisor for SSH-less container platforms.

io.net CaaS hands you a container and *one* public port. There is no SSH, no
``docker exec``, no volume you can copy results out of afterwards, and the
reservation disappears when ``duration_hours`` expires. So the container has to
do the whole job by itself:

1. On start, launch ``deploy/run_gpu_eval.sh`` as its own process group. That
   script owns the GPU work end to end — compat canary, sizing probe, the
   five-condition matrix, diurnal, deadline pair, figures, manifest, tarball.
   Nothing else needs to be running: the harness launches and kills its own
   ``vllm serve`` per condition.
2. Serve progress and results over ``$PORT`` — the single port CaaS publishes as
   ``public_url`` — for as long as the reservation lasts. Crucially the server
   keeps running *after* the eval finishes (or fails), because that is the only
   window in which the operator can download anything.

Status is derived from two things the script emits, neither of which requires
parsing prose:

* ``PROGRESS <token>`` lines in the run log (``booting``, ``canary``,
  ``probing``, ``running:<condition>``, ``rendering``, ``done``,
  ``case_done <name>``, ``total <n>``, ``canary_failed <msg>``, ``failed <msg>``)
* the result JSONs and ``MANIFEST.txt`` the script writes under ``$RESULTS_DIR``

plus the child's exit status, which is authoritative: a script that dies without
ever printing ``PROGRESS done`` is ``failed`` no matter what the log says.

Endpoints::

    GET  /                 endpoint index
    GET  /healthz          200 always (CaaS liveness)
    GET  /status           phase, progress, engine shape, last log lines
    GET  /log?tail=N       plain-text tail of the run log (default 200, cap 5000)
    GET  /results/         JSON listing of result files
    GET  /results/{name}   one result file (safe-joined under $RESULTS_DIR)
    GET  /results.tar.gz   the tarball, once the run is terminal (404 + status before)
    POST /abort            SIGTERM the child group; needs X-Tidal-Key

Run it with ``python -m tidal.eval.selfdrive``; ``deploy/engine-entrypoint.sh``
does exactly that when ``MODE=selfdrive`` (the default under ``CAAS=1``).
"""

from __future__ import annotations

import contextlib
import hmac
import logging
import os
import signal
import subprocess
import tarfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

__all__ = [
    "DEFAULT_TAIL",
    "MAX_TAIL",
    "STATUS_LOG_LINES",
    "TERMINAL_PHASES",
    "PopenChild",
    "SelfDriveConfig",
    "Supervisor",
    "build_config",
    "create_app",
    "main",
    "safe_join",
    "tail_lines",
]

log = logging.getLogger("tidal.selfdrive")

#: Marker the eval script prints for every machine-readable state change.
PROGRESS_PREFIX = "PROGRESS "

DEFAULT_TAIL = 200
MAX_TAIL = 5000
STATUS_LOG_LINES = 20

#: Phases after which no further work happens and the tarball is servable.
TERMINAL_PHASES = frozenset({"done", "failed", "aborted"})

#: Cases run_gpu_eval.sh executes when nothing is skipped. Only a fallback for
#: the denominator: the script prints ``PROGRESS total <n>`` once it knows
#: whether the canary let technique B through.
DEFAULT_TOTAL_CASES = 10

#: MANIFEST.txt keys worth surfacing in /status as "what engine produced this".
ENGINE_MANIFEST_KEYS = (
    "host",
    "utc",
    "model",
    "tensor_parallel",
    "max_model_len",
    "engine_args",
    "online_rps",
    "minutes",
    "batch_items",
    "items_per_s_probe",
    "canary",
)


def _now() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Child process
# --------------------------------------------------------------------------
class ChildHandle(Protocol):
    """The slice of a running child the supervisor needs.

    A protocol rather than ``subprocess.Popen`` so the tests can drive phase
    transitions with a stub instead of a real process.
    """

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def signal_group(self, sig: int) -> None: ...


class PopenChild:
    """``subprocess.Popen`` in its own session, signalled as a whole group.

    The eval script forks ``vllm serve`` per condition and vLLM V1 forks an
    EngineCore child on top of that, so ``proc.terminate()`` would leave GPU
    processes behind. ``start_new_session=True`` at launch plus ``killpg`` here
    is what makes /abort actually free the card.
    """

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> int | None:
        return self._proc.poll()

    def signal_group(self, sig: int) -> None:
        os.killpg(os.getpgid(self._proc.pid), sig)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class SelfDriveConfig:
    """Everything the supervisor reads from the environment, in one place."""

    results_dir: Path
    log_path: Path
    repo_dir: Path = Path("/opt/tidal")
    eval_script: Path = Path("/usr/local/bin/tidal-run-gpu-eval")
    # Binding all interfaces is the point: CaaS publishes the container port.
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    hf_repo: str = ""
    hf_token: str = ""
    #: Extra env for the child on top of os.environ.
    child_env: dict[str, str] = field(default_factory=dict)


def build_config(env: dict[str, str] | None = None) -> SelfDriveConfig:
    """Build the config from the container's environment.

    ``RESULTS_DIR`` is the contract with ``run_gpu_eval.sh``: the supervisor
    exports it to the child, so both ends always agree on where results land
    even if the image default ever changes.
    """
    env = dict(os.environ if env is None else env)
    results_dir = Path(env.get("RESULTS_DIR") or "/results")
    log_path = Path(env.get("RUN_LOG") or (results_dir / "selfdrive-run.log"))
    repo_dir = Path(env.get("REPO_DIR") or "/opt/tidal")

    script = env.get("EVAL_SCRIPT")
    if script:
        eval_script = Path(script)
    else:
        eval_script = Path("/usr/local/bin/tidal-run-gpu-eval")
        if not eval_script.exists():
            # Dev boxes and CI run straight out of a checkout.
            eval_script = repo_dir / "deploy" / "run_gpu_eval.sh"

    return SelfDriveConfig(
        results_dir=results_dir,
        log_path=log_path,
        repo_dir=repo_dir,
        eval_script=eval_script,
        host=env.get("HOST") or "0.0.0.0",
        port=int(env.get("PORT") or 8000),
        api_key=env.get("TIDAL_API_KEY") or "",
        hf_repo=env.get("TIDAL_RESULTS_HF_REPO") or "",
        hf_token=env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "",
    )


# --------------------------------------------------------------------------
# Small filesystem helpers
# --------------------------------------------------------------------------
def tail_lines(path: Path, n: int) -> list[str]:
    """Last ``n`` lines of ``path``, without reading a multi-GB log into memory."""
    if n <= 0:
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            # 512 bytes/line is generous for this log; the floor keeps small
            # tails from doing a pointlessly tiny read.
            start = max(0, size - max(65536, n * 512))
            fh.seek(start)
            data = fh.read()
    except OSError:
        return []
    text = data.decode("utf-8", "replace")
    if start > 0:
        # The first line is almost certainly cut in half by the seek.
        _, _, text = text.partition("\n")
    return text.splitlines()[-n:]


def safe_join(root: Path, name: str) -> Path | None:
    """Resolve ``name`` under ``root``, or ``None`` if it escapes.

    Rejects ``..`` segments, absolute paths and symlinks pointing outside the
    results directory — this endpoint is on a public URL with no auth.
    """
    if not name or name in {".", "/"}:
        return None
    try:
        root_resolved = root.resolve()
        candidate = (root_resolved / name).resolve()
    except OSError:
        return None
    if candidate == root_resolved or root_resolved not in candidate.parents:
        return None
    return candidate


def _parse_manifest(path: Path) -> dict[str, str]:
    """``key: value`` lines out of MANIFEST.txt; the nvidia-smi tail is ignored."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if line.startswith("---"):
            break
        key, sep, value = line.partition(":")
        if sep and key and not key.startswith(" "):
            out[key.strip()] = value.strip()
    return out


# --------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------
class Supervisor:
    """Owns the eval child process and derives status from its log + results."""

    def __init__(
        self,
        cfg: SelfDriveConfig,
        launcher: Callable[[], ChildHandle] | None = None,
    ) -> None:
        self.cfg = cfg
        self._launcher = launcher or (lambda: self._spawn())
        self._lock = threading.RLock()

        self._child: ChildHandle | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._exit_code: int | None = None
        self._aborted_at: float | None = None
        self._egress_done = False

        # Incremental log parse state.
        self._offset = 0
        self._pending = ""
        self._reset_progress()

        self._tar_lock = threading.Lock()
        self._built_tarball: Path | None = None

    # -- lifecycle ---------------------------------------------------------
    def _reset_progress(self) -> None:
        self._phase = "booting"
        self._error: str | None = None
        self._canary: str | None = None
        self._total: int | None = None
        self._done_cases: list[str] = []
        self._failed_cases: list[str] = []

    def _spawn(self) -> ChildHandle:
        cfg = self.cfg
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.update(cfg.child_env)
        env["RESULTS_DIR"] = str(cfg.results_dir)
        env["RESULTS_BASE"] = str(cfg.results_dir)
        env["PYTHONUNBUFFERED"] = "1"

        # Append, so a container restart keeps the earlier attempt's log.
        handle = cfg.log_path.open("ab", buffering=0)
        try:
            # Fixed argv from config; no shell, and stdin is /dev/null so the
            # child can never block on a prompt.
            proc = subprocess.Popen(
                ["/bin/bash", str(cfg.eval_script)],
                cwd=str(cfg.repo_dir if cfg.repo_dir.exists() else Path.cwd()),
                env=env,
                stdin=subprocess.DEVNULL,  # no TTY, ever
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            handle.close()
        log.info("launched %s (pid %d)", cfg.eval_script, proc.pid)
        return PopenChild(proc)

    def start(self) -> None:
        """Launch the eval child. Idempotent."""
        with self._lock:
            if self._child is not None:
                return
            self._started_at = _now()
            self._note(f"[selfdrive] starting {self.cfg.eval_script}")
            self._child = self._launcher()

    def _note(self, message: str) -> None:
        """Append a supervisor line to the run log so /log tells the whole story."""
        try:
            self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.log_path.open("a", encoding="utf-8") as fh:
                fh.write(message.rstrip("\n") + "\n")
        except OSError:  # pragma: no cover — a read-only log is not fatal
            log.warning("could not append to %s", self.cfg.log_path)

    # -- log ingestion -----------------------------------------------------
    def _drain_log(self) -> None:
        path = self.cfg.log_path
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._offset:  # truncated out from under us
            self._offset = 0
            self._pending = ""
            self._reset_progress()
        if size == self._offset:
            return
        try:
            with path.open("rb") as fh:
                fh.seek(self._offset)
                data = fh.read()
        except OSError:
            return
        self._offset += len(data)
        chunk = self._pending + data.decode("utf-8", "replace")
        lines = chunk.split("\n")
        self._pending = lines.pop()  # incomplete last line, re-read next time
        for line in lines:
            self._ingest(line)

    def _ingest(self, line: str) -> None:
        idx = line.find(PROGRESS_PREFIX)
        if idx < 0:
            return
        token = line[idx + len(PROGRESS_PREFIX) :].strip()
        if not token:
            return
        head, _, rest = token.partition(" ")
        rest = rest.strip()

        if head in {"booting", "canary", "probing", "rendering", "done"} or head.startswith(
            "running:"
        ):
            self._phase = head
        elif head == "total":
            with contextlib.suppress(ValueError):
                self._total = int(rest)
        elif head == "case_done":
            if rest and rest not in self._done_cases:
                self._done_cases.append(rest)
        elif head == "case_failed":
            if rest and rest not in self._failed_cases:
                self._failed_cases.append(rest)
        elif head == "canary_ok":
            self._canary = "ok"
        elif head == "canary_failed":
            self._canary = "failed"
            # Sticky: the run continues (technique A still runs) but every later
            # status must keep saying that technique B is not trustworthy here.
            self._error = rest or "compat canary failed"
        elif head == "failed":
            self._phase = "failed"
            self._error = rest or self._error or "eval script reported failure"
        elif head == "error":
            self._error = rest or self._error

    # -- derived state -----------------------------------------------------
    def _refresh(self) -> None:
        self._drain_log()
        child = self._child
        if child is None or self._exit_code is not None:
            return
        rc = child.poll()
        if rc is None:
            return
        self._exit_code = rc
        self._finished_at = _now()
        self._drain_log()  # anything written between the last read and exit
        log.info("eval child exited with %s", rc)
        self._maybe_egress()

    def _resolved_phase(self) -> str:
        if self._aborted_at is not None:
            return "aborted"
        if self._exit_code is None:
            return self._phase
        if self._exit_code == 0:
            return "done" if self._phase != "failed" else "failed"
        return "failed"

    def _result_json_count(self) -> int:
        try:
            return sum(
                1
                for p in self.cfg.results_dir.rglob("*.json")
                if p.is_file() and p.parent.name != "figures"
            )
        except OSError:
            return 0

    def manifest(self) -> tuple[dict[str, str], Path | None]:
        """Newest MANIFEST.txt under the results dir, parsed."""
        try:
            candidates = sorted(
                (p for p in self.cfg.results_dir.rglob("MANIFEST.txt") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return {}, None
        if not candidates:
            return {}, None
        newest = candidates[-1]
        return _parse_manifest(newest), newest

    def _engine_shape(self) -> dict[str, Any]:
        parsed, path = self.manifest()
        if parsed:
            shape = {k: parsed[k] for k in ENGINE_MANIFEST_KEYS if k in parsed}
            shape["source"] = str(path)
            return shape
        # No manifest yet — echo the shape the child was asked for, so /status
        # is useful during the two hours before MANIFEST.txt exists.
        env = os.environ
        return {
            "model": env.get("MODEL", ""),
            "tensor_parallel": env.get("TENSOR_PARALLEL_SIZE", ""),
            "max_model_len": env.get("MAX_MODEL_LEN", ""),
            "source": "env (MANIFEST.txt not written yet)",
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh()
            phase = self._resolved_phase()
            now = _now()
            started = self._started_at
            end = self._finished_at or (self._aborted_at if phase == "aborted" else None)
            elapsed = (end or now) - started if started is not None else 0.0

            error = self._error
            if phase == "failed" and not error:
                if self._exit_code not in (None, 0):
                    error = f"eval script exited with code {self._exit_code}"
                else:
                    error = "eval script reported failure"

            completed = max(len(self._done_cases), self._result_json_count())
            return {
                "phase": phase,
                "error": error,
                "canary": self._canary or ("unknown" if phase in TERMINAL_PHASES else "pending"),
                "conditions": {
                    "completed": completed,
                    "total": self._total or DEFAULT_TOTAL_CASES,
                    "done": list(self._done_cases),
                    "failed": list(self._failed_cases),
                },
                "started_at": _iso(started),
                "finished_at": _iso(self._finished_at),
                "elapsed_s": round(elapsed, 1),
                "exit_code": self._exit_code,
                "pid": self._child.pid if self._child is not None else None,
                "results_dir": str(self.cfg.results_dir),
                "results_ready": phase in TERMINAL_PHASES,
                "engine": self._engine_shape(),
                "log_tail": tail_lines(self.cfg.log_path, STATUS_LOG_LINES),
            }

    # -- actions -----------------------------------------------------------
    def check_key(self, presented: str | None) -> bool:
        if not self.cfg.api_key:
            return False
        if not presented:
            return False
        return hmac.compare_digest(presented, self.cfg.api_key)

    def abort(self) -> bool:
        """SIGTERM the child's whole process group. Returns False if nothing ran."""
        with self._lock:
            self._refresh()
            child = self._child
            if child is None:
                return False
            self._aborted_at = _now()
            if self._exit_code is None:
                try:
                    child.signal_group(signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError) as exc:
                    log.warning("abort: could not signal group: %s", exc)
            self._note("[selfdrive] abort requested; SIGTERM sent to the eval process group")
            return True

    # -- results -----------------------------------------------------------
    def list_results(self) -> list[dict[str, Any]]:
        root = self.cfg.results_dir
        out: list[dict[str, Any]] = []
        try:
            paths: Iterable[Path] = sorted(root.rglob("*"))
        except OSError:
            return out
        for path in paths:
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
            except OSError:
                continue
            out.append(
                {
                    "name": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "modified": _iso(stat.st_mtime),
                }
            )
        return out

    def existing_tarball(self) -> Path | None:
        """The tarball run_gpu_eval.sh packed, if it got that far."""
        try:
            balls = [p for p in self.cfg.results_dir.glob("*.tar.gz") if p.is_file()]
        except OSError:
            return None
        if not balls:
            return None
        return max(balls, key=lambda p: p.stat().st_mtime)

    def ensure_tarball(self) -> Path | None:
        """The tarball to serve: the script's if present, otherwise built now.

        Building at request time is what makes an aborted or crashed run still
        retrievable — the script only packs on the happy path.
        """
        found = self.existing_tarball()
        if found is not None:
            return found
        with self._tar_lock:
            found = self.existing_tarball()
            if found is not None:
                return found
            if self._built_tarball is not None and self._built_tarball.exists():
                return self._built_tarball
            root = self.cfg.results_dir
            if not root.is_dir():
                return None
            stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            # Outside results_dir, or the archive would try to contain itself.
            target = root.parent / f"tidal-results-{stamp}.tar.gz"
            try:
                with tarfile.open(target, "w:gz") as tar:
                    for path in sorted(root.rglob("*")):
                        if path.is_file() and path.suffix != ".gz":
                            tar.add(path, arcname=f"results/{path.relative_to(root).as_posix()}")
            except OSError as exc:  # pragma: no cover — disk full etc.
                log.warning("could not build tarball: %s", exc)
                return None
            self._built_tarball = target
            return target

    # -- optional backup egress -------------------------------------------
    def _maybe_egress(self) -> None:
        if self._egress_done or not (self.cfg.hf_repo and self.cfg.hf_token):
            return
        self._egress_done = True
        threading.Thread(target=self._egress, name="tidal-egress", daemon=True).start()

    def _egress(self) -> None:  # pragma: no cover — needs network + a token
        """Push the tarball to a HF dataset repo. Best effort, log-only."""
        try:
            tarball = self.ensure_tarball()
            if tarball is None:
                log.warning("egress: no tarball to upload")
                return
            from huggingface_hub import HfApi

            api = HfApi(token=self.cfg.hf_token)
            api.create_repo(self.cfg.hf_repo, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(tarball),
                path_in_repo=tarball.name,
                repo_id=self.cfg.hf_repo,
                repo_type="dataset",
            )
            log.info("egress: uploaded %s to %s", tarball.name, self.cfg.hf_repo)
            self._note(f"[selfdrive] uploaded {tarball.name} to {self.cfg.hf_repo}")
        except Exception as exc:  # egress must never take the server down
            log.warning("egress to %s failed: %s", self.cfg.hf_repo, exc)
            self._note(f"[selfdrive] egress to {self.cfg.hf_repo} failed: {exc}")


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------
def create_app(sup: Supervisor) -> FastAPI:
    app = FastAPI(title="tidal selfdrive", docs_url=None, redoc_url=None)
    app.state.supervisor = sup

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        # Deliberately does not consult the child: this is CaaS's liveness
        # probe, and a failed eval must not get the container restarted out
        # from under the results it just produced.
        return {"ok": True, "service": "tidal-selfdrive"}

    @app.get("/")
    def index() -> dict[str, Any]:
        return {
            "service": "tidal-selfdrive",
            "endpoints": [
                "GET /status",
                "GET /log?tail=N",
                "GET /results/",
                "GET /results/{name}",
                "GET /results.tar.gz",
                "POST /abort  (header X-Tidal-Key)",
                "GET /healthz",
            ],
            "phase": sup.status()["phase"],
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        return sup.status()

    @app.get("/log", response_class=PlainTextResponse)
    def get_log(tail: int = DEFAULT_TAIL) -> PlainTextResponse:
        n = max(1, min(int(tail), MAX_TAIL))
        lines = tail_lines(sup.cfg.log_path, n)
        return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""))

    @app.get("/results.tar.gz")
    def results_tarball() -> Any:
        st = sup.status()
        if st["phase"] not in TERMINAL_PHASES:
            return JSONResponse(
                {"error": "run is still in progress", "status": st},
                status_code=404,
            )
        path = sup.ensure_tarball()
        if path is None:
            return JSONResponse(
                {"error": "no results to pack", "status": st},
                status_code=404,
            )
        return FileResponse(path, media_type="application/gzip", filename=path.name)

    @app.get("/results/")
    def results_index() -> dict[str, Any]:
        files = sup.list_results()
        return {
            "results_dir": str(sup.cfg.results_dir),
            "count": len(files),
            "total_bytes": sum(int(f["size"]) for f in files),
            "files": files,
        }

    @app.get("/results/{name:path}")
    def results_file(name: str) -> Any:
        path = safe_join(sup.cfg.results_dir, name)
        if path is None or not path.is_file():
            return JSONResponse({"error": f"no such result file: {name}"}, status_code=404)
        return FileResponse(path, filename=path.name)

    @app.post("/abort")
    def abort(x_tidal_key: str | None = Header(default=None, alias="X-Tidal-Key")) -> Any:
        if not sup.check_key(x_tidal_key):
            detail = (
                "abort is disabled: TIDAL_API_KEY is not set on the container"
                if not sup.cfg.api_key
                else "missing or wrong X-Tidal-Key header"
            )
            return JSONResponse({"error": detail}, status_code=401)
        if not sup.abort():
            return JSONResponse({"error": "nothing is running"}, status_code=409)
        return {"aborted": True, "status": sup.status()}

    return app


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=os.environ.get("TIDAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = build_config()
    sup = Supervisor(cfg)
    sup.start()
    app = create_app(sup)

    import uvicorn

    log.info("serving progress + results on %s:%d", cfg.host, cfg.port)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info", access_log=False)


if __name__ == "__main__":  # pragma: no cover
    main()
