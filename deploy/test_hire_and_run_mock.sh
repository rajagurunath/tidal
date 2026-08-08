#!/usr/bin/env bash
#
# test_hire_and_run_mock.sh — exercise deploy/hire_and_run.sh end to end against
# an inline stub of the io.net CaaS marketplace API.
#
#   deploy/test_hire_and_run_mock.sh
#
# No GPU, no network, no money. The stub implements the six endpoints the driver
# actually touches (/hardware, /price, /deploy, /deployment/{id},
# /deployment/{id}/containers, DELETE /deployment/{id}) *and* doubles as the
# container's own public_url (/status, /log, /results.tar.gz), because that is
# exactly what CaaS does: the containers endpoint hands you back a URL that
# speaks the selfdrive supervisor's protocol.
#
# The catalogue is rigged so that exactly one row is the right answer, and it is
# only reachable if all three selection filters work:
#
#   gpu_1x_l40_16    $0.10/hr  16GB  available  matches PREFER  -> too small
#   gpu_1x_a6000_out $0.10/hr  48GB  SOLD OUT   matches PREFER  -> unavailable
#   gpu_1x_rtx4000   $0.25/hr  16GB  available  no PREFER match -> filtered
#   gpu_1x_a6000     $0.80/hr  48GB  available  matches PREFER  -> THE ANSWER
#   gpu_1x_a100      $1.50/hr  80GB  available  matches PREFER  -> more expensive
#   gpu_1x_h100      $9.00/hr  80GB  available  matches PREFER  -> over budget
#
# What is asserted: the driver hires the right card, writes every evidence
# artefact, walks the phase machine, retrieves a real tarball, destroys the
# deployment, never leaks the API key into evidence, and refuses to hire at all
# when the price estimate blows the budget.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/hire_and_run.sh"

command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
command -v jq      >/dev/null 2>&1 || { echo "jq is required" >&2; exit 1; }
[ -x "$DRIVER" ] || { echo "driver not executable: $DRIVER" >&2; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/tidal-hire-mock.XXXXXX")"
STUB_PY="${WORK}/caas_stub.py"
STUB_PID=""
FAILURES=0

# The key the driver must never write anywhere we can read.
MOCK_KEY="stub-S3CR3T-9f2a4c"
MOCK_ABORT_KEY="mock-abort-key-0000"

# shellcheck disable=SC2317,SC2329  # invoked from the EXIT trap
cleanup() {
  stop_stub
  if [ "${KEEP_WORK:-0}" = "1" ]; then
    echo "workdir kept: $WORK"
  else
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT

pass() { printf '  ok   %s\n' "$*"; }
fail() { printf '  FAIL %s\n' "$*"; FAILURES=$(( FAILURES + 1 )); }

assert_nonempty_file() {
  if [ -s "$1" ]; then pass "$2"; else fail "$2 — missing or empty: $1"; fi
}

assert_eq() {
  if [ "$1" = "$2" ]; then pass "$3"; else fail "$3 — expected '$2', got '$1'"; fi
}

assert_contains() {
  if printf '%s' "$1" | grep -q -- "$2"; then pass "$3"; else fail "$3 — '$2' not found in: $(printf '%s' "$1" | head -c 200)"; fi
}

# --------------------------------------------------------------------------
# The stub
# --------------------------------------------------------------------------
cat >"$STUB_PY" <<'PYEOF'
"""Inline stub of the io.net CaaS marketplace API + a selfdrive container.

Binds an ephemeral port and writes it to $STUB_DIR/port. Everything the test
needs to assert is written to $STUB_DIR as it happens, so the assertions read
files instead of racing the server.

Env knobs:
  STUB_DIR        where to record (required)
  STUB_EXPENSIVE  1 -> /price returns a number that must trip the budget abort
  STUB_PENDING    number of /containers calls answered "pending" before running
"""

import gzip
import io
import json
import os
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

STUB_DIR = os.environ["STUB_DIR"]
EXPENSIVE = os.environ.get("STUB_EXPENSIVE") == "1"
PENDING_CALLS = int(os.environ.get("STUB_PENDING", "1"))

LOCK = threading.Lock()
COUNTERS = {"containers": 0, "status": 0}
DEPLOY_ID = "dep-mock-0001"

HARDWARE = [
    {"hardware_id": "gpu_1x_l40_16", "hardware_name": "NVIDIA L40 16GB",
     "available": 4, "price": 0.10, "max_gpus_per_container": 8, "location": "US"},
    {"hardware_id": "gpu_1x_a6000_out", "hardware_name": "NVIDIA RTX A6000 48GB",
     "available": 0, "price": 0.10, "max_gpus_per_container": 8, "location": "US"},
    {"hardware_id": "gpu_1x_rtx4000", "hardware_name": "NVIDIA RTX 4000 16GB",
     "available": 6, "price": 0.25, "max_gpus_per_container": 8, "location": "US"},
    {"hardware_id": "gpu_1x_a6000", "hardware_name": "NVIDIA RTX A6000 48GB",
     "available": 3, "price": 0.80, "max_gpus_per_container": 8, "location": "US"},
    {"hardware_id": "gpu_1x_a100", "hardware_name": "NVIDIA A100 80GB",
     "available": 2, "price": 1.50, "max_gpus_per_container": 8, "location": "US"},
    {"hardware_id": "gpu_1x_h100", "hardware_name": "NVIDIA H100 80GB",
     "available": 1, "price": 9.00, "max_gpus_per_container": 8, "location": "US"},
]

# The phase walk the driver has to survive: booting -> one running case -> done.
STATUS_SCRIPT = [
    {"phase": "booting", "completed": 0},
    {"phase": "running:online_only", "completed": 1},
    {"phase": "done", "completed": 6},
]


def record(name, text, mode="a"):
    with LOCK, open(os.path.join(STUB_DIR, name), mode) as fh:
        fh.write(text)


def make_tarball():
    payload = b"model: Qwen/Qwen2.5-7B-Instruct\ncanary: ok\n"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        info = tarfile.TarInfo("gpu-mock/MANIFEST.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return gzip.compress(raw.getvalue())


TARBALL = make_tarball()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the test output readable
        record("access.log", "%s %s\n" % (self.command, self.path))

    # -- helpers ---------------------------------------------------------
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj))

    def _note_key(self, path):
        """Record `METHOD PATH KEY` so the test can assert per-route.

        The marketplace routes must always carry x-api-key; the container's own
        surface (/status, /log, /results.tar.gz) must never see it, because
        public_url is an unauthenticated third-party URL.
        """
        key = self.headers.get("x-api-key")
        record("api-keys.txt", "%s %s %s\n" % (self.command, path, key or "-"))
        return key

    def _self_url(self):
        return "http://127.0.0.1:%d" % self.server.server_address[1]

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        self._note_key(path)

        if path == "/hardware":
            return self._json(200, {"data": HARDWARE})

        if path == "/price":
            total = 999.99 if EXPENSIVE else 1.60
            return self._json(200, {"data": {"total_price": total, "currency": "USD"}})

        if path.endswith("/containers"):
            with LOCK:
                COUNTERS["containers"] += 1
                n = COUNTERS["containers"]
            if n <= PENDING_CALLS:
                worker = {"public_url": "", "status": "pending"}
            else:
                worker = {"public_url": self._self_url(), "status": "running"}
            return self._json(200, {"data": {"workers": [worker]}})

        if path.startswith("/deployment/"):
            return self._json(200, {"data": {"id": DEPLOY_ID, "status": "running"}})

        # ---- the container's own surface (public_url points back here) ----
        if path == "/status":
            with LOCK:
                i = min(COUNTERS["status"], len(STATUS_SCRIPT) - 1)
                COUNTERS["status"] += 1
            step = STATUS_SCRIPT[i]
            return self._json(200, {
                "phase": step["phase"],
                "error": None,
                "canary": "ok",
                "conditions": {"completed": step["completed"], "total": 6,
                               "done": [], "failed": []},
                "elapsed_s": 10.0 * (i + 1),
                "results_ready": step["phase"] == "done",
                "engine": {"model": "Qwen/Qwen2.5-7B-Instruct"},
                "log_tail": ["PROGRESS %s" % step["phase"]],
            })

        if path == "/log":
            lines = "\n".join("mock run log line %d" % i for i in range(1, 51))
            return self._send(200, lines + "\n", "text/plain")

        if path == "/results.tar.gz":
            return self._send(200, TARBALL, "application/gzip")

        if path in ("/", "/healthz"):
            return self._json(200, {"ok": True})

        return self._json(404, {"error": "no route", "path": path})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        self._note_key(path)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode() if length else ""

        if path == "/deploy":
            record("deploy-request.json", body, mode="w")
            return self._json(200, {"data": {"deployment_id": DEPLOY_ID,
                                             "status": "pending"}})
        if path.endswith("/extend"):
            record("extend.txt", body + "\n")
            return self._json(200, {"data": {"ok": True}})
        return self._json(404, {"error": "no route", "path": path})

    def do_DELETE(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        self._note_key(path)
        if path.startswith("/deployment/"):
            record("deleted.txt", path.rsplit("/", 1)[-1] + "\n")
            return self._json(200, {"data": {"deleted": True}})
        return self._json(404, {"error": "no route", "path": path})


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    tmp = os.path.join(STUB_DIR, "port.tmp")
    with open(tmp, "w") as fh:
        fh.write(str(port))
    os.rename(tmp, os.path.join(STUB_DIR, "port"))
    server.serve_forever()


if __name__ == "__main__":
    main()
PYEOF

start_stub() {
  local tag="$1"
  shift
  STUB_DIR="${WORK}/${tag}"
  mkdir -p "$STUB_DIR"
  STUB_DIR="$STUB_DIR" env "$@" python3 "$STUB_PY" >"${STUB_DIR}/stub.stderr" 2>&1 &
  STUB_PID=$!
  local i=0
  while [ ! -s "${STUB_DIR}/port" ]; do
    i=$(( i + 1 ))
    if [ "$i" -gt 100 ]; then
      cat "${STUB_DIR}/stub.stderr" >&2 || true
      echo "stub failed to start" >&2
      exit 1
    fi
    sleep 0.1
  done
  STUB_PORT="$(cat "${STUB_DIR}/port")"
  STUB_BASE="http://127.0.0.1:${STUB_PORT}"
}

stop_stub() {
  if [ -n "$STUB_PID" ]; then
    kill "$STUB_PID" 2>/dev/null || true
    wait "$STUB_PID" 2>/dev/null || true
    STUB_PID=""
  fi
}

# Everything the driver needs to be fast and hermetic. TIDAL_SHA is pinned so
# the test does not depend on the checkout's HEAD.
run_driver() {
  local evidence="$1"
  shift
  env -i \
    PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" \
    CAAS_API_KEY="$MOCK_KEY" \
    CAAS_BASE="$STUB_BASE" \
    EVIDENCE_DIR="$evidence" \
    TIDAL_SHA="0123456789abcdef0123456789abcdef01234567" \
    TIDAL_API_KEY="$MOCK_ABORT_KEY" \
    BUDGET_USD=10 INITIAL_HOURS=2 \
    POLL_BOOT_S=1 POLL_STATUS_S=1 \
    BOOT_TIMEOUT_S=30 EVAL_TIMEOUT_S=60 HTTP_TIMEOUT_S=10 \
    FETCH_RETRIES=2 \
    "$@" \
    "$DRIVER"
}

# ==========================================================================
echo "== case 1: happy path =="
# ==========================================================================
start_stub happy
EV1="${WORK}/evidence-happy"
RC=0
run_driver "$EV1" >"${WORK}/driver-happy.out" 2>&1 || RC=$?

assert_eq "$RC" "0" "driver exits 0 on phase=done"
if [ "$RC" != "0" ]; then
  echo "---- driver output ----" >&2
  cat "${WORK}/driver-happy.out" >&2
fi

for f in driver.log hardware.json hardware-choice.json price.json \
         entrypoint-cmd.txt deploy-request.redacted.json deploy-response.json \
         containers-log.jsonl status-log.jsonl full-log.txt results.tar.gz \
         delete-response.json; do
  assert_nonempty_file "${EV1}/${f}" "evidence: ${f}"
done

CHOICE="$(jq -r '.choice.id' "${EV1}/hardware-choice.json")"
assert_eq "$CHOICE" "gpu_1x_a6000" "picks the cheapest available >=24GB card matching PREFER"

PHASES="$(jq -r '.phase' "${EV1}/status-log.jsonl" | tr '\n' ' ')"
assert_contains "$PHASES" "booting" "status log records the booting phase"
assert_contains "$PHASES" "running:online_only" "status log records a running phase"
assert_contains "$PHASES" "done" "status log records the terminal phase"

if tar -tzf "${EV1}/results.tar.gz" 2>/dev/null | grep -q 'MANIFEST.txt'; then
  pass "results.tar.gz is a real gzip tarball with a MANIFEST"
else
  fail "results.tar.gz is not a readable tarball"
fi

assert_nonempty_file "${STUB_DIR}/deleted.txt" "stub recorded a DELETE (no leaked deployment)"
assert_eq "$(tr -d '\n' <"${STUB_DIR}/deleted.txt")" "dep-mock-0001" "DELETE targeted the right deployment id"

KEYS="${STUB_DIR}/api-keys.txt"
MARKET_KEYS="$(awk '$2 ~ /^\/(hardware|price|deploy|deployment)/ {print $3}' "$KEYS" | sort -u | tr '\n' ' ')"
assert_eq "${MARKET_KEYS% }" "$MOCK_KEY" "every marketplace API call carried x-api-key, and only that key"
CONTAINER_KEYS="$(awk '$2 ~ /^\/(status|log|results)/ {print $3}' "$KEYS" | sort -u | tr '\n' ' ')"
assert_eq "${CONTAINER_KEYS% }" "-" "the CaaS key is never sent to the container public_url"

if grep -rq -- "$MOCK_KEY" "$EV1" 2>/dev/null; then
  fail "the API key leaked into the evidence directory"
else
  pass "the API key never appears in evidence"
fi
if grep -rq -- "$MOCK_ABORT_KEY" "$EV1" 2>/dev/null; then
  fail "the container secret leaked into the evidence directory"
else
  pass "secret_env_variables are REDACTED in the saved payload"
fi

REDACTED="$(jq -r '.container_config.secret_env_variables.TIDAL_API_KEY' "${EV1}/deploy-request.redacted.json")"
assert_eq "$REDACTED" "REDACTED" "saved payload redacts TIDAL_API_KEY"

REQ="${STUB_DIR}/deploy-request.json"
assert_nonempty_file "$REQ" "stub captured the deploy payload"
assert_eq "$(jq -r '.container_config.entrypoint[0:2] | join(" ")' "$REQ")" "bash -lc" "entrypoint is bash -lc"
assert_eq "$(jq -r 'has("args") or (.container_config | has("args"))' "$REQ")" "false" "no args field is sent (the endpoint drops it silently)"
assert_eq "$(jq -r '.location_ids | join(",")' "$REQ")" "US" "location_ids is present and non-empty"
assert_eq "$(jq -r '.hardware_id' "$REQ")" "gpu_1x_a6000" "payload carries the selected hardware_id"
assert_eq "$(jq -r '.duration_hours' "$REQ")" "2" "payload carries INITIAL_HOURS"
ENTRY="$(jq -r '.container_config.entrypoint[2]' "$REQ")"
assert_contains "$ENTRY" "0123456789abcdef0123456789abcdef01234567" "entrypoint pins TIDAL_SHA"
assert_contains "$ENTRY" "exec python3 -m tidal.eval.selfdrive" "entrypoint execs the selfdrive supervisor"
assert_contains "$ENTRY" "EVAL_SCRIPT=/opt/tidal/deploy/run_gpu_eval.sh" "entrypoint exports EVAL_SCRIPT"
if printf '%s' "$ENTRY" | grep -q "'"; then
  fail "entrypoint contains a single quote (must stay safe to wrap in '...')"
else
  pass "entrypoint is single-quote-safe"
fi

assert_contains "$(cat "${WORK}/driver-happy.out")" "duration_hours" "driver prints the extend curl on the way out"

stop_stub

# ==========================================================================
echo "== case 2: budget abort =="
# ==========================================================================
start_stub expensive STUB_EXPENSIVE=1
EV2="${WORK}/evidence-expensive"
RC=0
run_driver "$EV2" >"${WORK}/driver-expensive.out" 2>&1 || RC=$?

assert_eq "$RC" "5" "an over-budget /price estimate exits 5"
assert_contains "$(cat "${WORK}/driver-expensive.out")" "budget abort" "the abort says why"
if [ -f "${STUB_DIR}/deploy-request.json" ]; then
  fail "budget abort still hired something"
else
  pass "budget abort hired nothing (no POST /deploy)"
fi
if [ -f "${STUB_DIR}/deleted.txt" ]; then
  fail "budget abort issued a DELETE for a deployment that never existed"
else
  pass "budget abort issued no DELETE"
fi
assert_nonempty_file "${EV2}/hardware.json" "budget abort still leaves evidence"

stop_stub

# ==========================================================================
echo
if [ "$FAILURES" -eq 0 ]; then
  echo "ALL ASSERTIONS PASSED"
  exit 0
fi
echo "${FAILURES} ASSERTION(S) FAILED"
exit 1
