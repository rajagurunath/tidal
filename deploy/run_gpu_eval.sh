#!/usr/bin/env bash
#
# Unattended GPU evaluation for Tidal. Runs inside the engine container, where
# the `vllm` binary lives.
#
#   docker compose -f deploy/docker-compose.gpu.yml stop engine    # free the GPU
#   docker run --rm --gpus all --ipc host --shm-size 32g \
#     -v tidal_hf-cache:/root/.cache/huggingface \
#     -v "$PWD/results:/opt/tidal/results" \
#     -e MODEL=Qwen/Qwen2.5-7B-Instruct \
#     --entrypoint tidal-run-gpu-eval tidal-engine:fast
#
# WHY THE SERVING STACK MUST BE DOWN: `tidal.eval.harness` owns its engine. It
# launches a fresh `vllm serve` per condition (stock or --scheduler-cls tidal),
# waits for /health, drives the load, and tears the process group down. A
# long-lived serving engine on the same GPU would make every latency number
# meaningless. The "wait for engine health" step below is therefore a *probe*
# run: it pays the weight-download and first-load cost once, proves the GPU
# path works end to end, and measures the offline throughput ceiling used to
# size every later pool.
#
# What it produces under results/gpu-<host>-<utc>/:
#   matrix/     the five conditions: online_only offline_only naive
#               technique_a technique_b, plus figures/
#   diurnal/    technique_a + technique_b under sinusoidal online load
#   deadline/   one control-vs-laxity pair at a deliberately tight window
#   probe/      the sizing run
#   logs/       per-case stdout and per-case vLLM engine logs
#   MANIFEST.txt, and a tarball next to the directory.
#
# Robustness contract:
#   * set -euo pipefail
#   * every case is failure-isolated — one bad condition does not abort the run
#   * resumable — a case whose result JSON already parses is skipped, so a
#     re-invocation after an OOM or a dropped SSH session picks up where it left
#     off. Delete the JSON to force a re-run.

set -euo pipefail

# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/opt/tidal}"
RESULTS_BASE="${RESULTS_BASE:-${REPO_DIR}/results}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EVAL_PORT="${EVAL_PORT:-8400}"

# GPU-scale load. 20 rps peak is the target the plan calls for; the CPU testbed
# in the paper topped out near 3.
ONLINE_RPS="${ONLINE_RPS:-20}"
MINUTES="${MINUTES:-15}"
ONLINE_MAX_TOKENS="${ONLINE_MAX_TOKENS:-64}"
BATCH_MAX_TOKENS="${BATCH_MAX_TOKENS:-256}"
# TidalConfig's default of 4 is a Mac-CPU number; the field comment says 64 on GPU.
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S:-900}"
SEED="${SEED:-7}"

# Sizing probe: an offline_only run that measures the batch ceiling. Its length
# is governed by PROBE_ITEMS, not by --minutes: offline_only builds no arrival
# schedule and simply drains the pool. Note the harness fires the whole pool at
# once through one httpx.AsyncClient, whose default pool caps at 100 concurrent
# connections — so this measures the ceiling at ~100-way batch concurrency,
# which is the same ceiling every co-located condition is bounded by.
PROBE_ITEMS="${PROBE_ITEMS:-600}"
# Pool multiple over "how many items the ceiling could do in the window". >1
# guarantees the pool never drains, which is what makes batch throughput a
# measurement of the scheduler rather than of the pool size. The harness prints
# a NOTE when a pool drains; that NOTE means raise this.
POOL_SAFETY="${POOL_SAFETY:-2.0}"
MIN_BATCH_ITEMS="${MIN_BATCH_ITEMS:-500}"

# Diurnal: two cycles inside the window (loadgen's default period), peak = ONLINE_RPS.
DIURNAL_MINUTES="${DIURNAL_MINUTES:-20}"

# Deadline stress: one feasible-but-tight pair. The paper could not construct a
# discriminating regime on CPU; TIGHTNESS is the dial to sweep here. It is the
# fraction of the *offline ceiling* the pool is sized to, so 0.55 means "batch
# needs a bit over half the GPU for the whole window to make its deadline" —
# infeasible without co-scheduling headroom, feasible with it.
DEADLINE_MINUTES="${DEADLINE_MINUTES:-18}"
DEADLINE_TIGHTNESS="${DEADLINE_TIGHTNESS:-0.55}"
# Completion window as a multiple of the run length. 1.15 leaves a short drain.
DEADLINE_WINDOW_SLACK="${DEADLINE_WINDOW_SLACK:-1.15}"

# Hard cap per case, so a wedged engine cannot hang an overnight run.
CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-5400}"

PREWARM="${PREWARM:-1}"
MAKE_TARBALL="${MAKE_TARBALL:-1}"
# RESUME=1 reuses the newest gpu-<host>-* run directory instead of starting a
# fresh one, which is what makes "just re-run it" pick up after a crash. Set
# RESUME=0 (or pass an explicit RUN_DIR) for a clean run.
RESUME="${RESUME:-1}"

# The harness defaults this to a path on the author's Mac.
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"

# VllmServer.environment() defaults HF_HUB_OFFLINE to "1" when unset, which
# would make a cold node fail to fetch weights. Force it off unless told
# otherwise; after the prewarm step it is a no-op either way.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TIDAL_EVAL_MODEL="$MODEL"
export TIDAL_EVAL_HEALTH_TIMEOUT_S="${TIDAL_EVAL_HEALTH_TIMEOUT_S:-1800}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
HOSTNAME_TAG="$(hostname -s 2>/dev/null || hostname || echo unknown)"
mkdir -p "$RESULTS_BASE"

if [ -n "${RUN_DIR:-}" ]; then
  RUN_NAME="$(basename "$RUN_DIR")"
else
  RUN_NAME=""
  if [ "$RESUME" = "1" ]; then
    # Newest existing run for this host, if any. Without this, every retry would
    # mint a fresh timestamped directory and re-run everything. The glob sorts
    # lexically ascending and the stamps are ISO-8601, so the last match is the
    # newest; a glob rather than `ls` keeps paths with spaces intact.
    newest_run=""
    for candidate in "${RESULTS_BASE}/gpu-${HOSTNAME_TAG}-"*; do
      [ -d "$candidate" ] || continue
      newest_run="$candidate"
    done
    if [ -n "$newest_run" ]; then
      RUN_NAME="$(basename "$newest_run")"
      echo "resuming existing run ${RUN_NAME} (RESUME=0 to start fresh)" >&2
    fi
  fi
  if [ -z "$RUN_NAME" ]; then
    RUN_NAME="gpu-${HOSTNAME_TAG}-${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
  fi
  RUN_DIR="${RESULTS_BASE}/${RUN_NAME}"
fi
STAMP="${RUN_NAME##*-}"

MATRIX_DIR="${RUN_DIR}/matrix"
DIURNAL_DIR="${RUN_DIR}/diurnal"
DEADLINE_DIR="${RUN_DIR}/deadline"
PROBE_DIR="${RUN_DIR}/probe"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "$MATRIX_DIR" "$DIURNAL_DIR" "$DEADLINE_DIR" "$PROBE_DIR" "$LOG_DIR"

FAILED_CASES=()
SKIPPED_CASES=()
RAN_CASES=()

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

die() {
  log "FATAL: $*"
  exit 1
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
preflight() {
  log "run dir: ${RUN_DIR}"
  [ -n "$VLLM_BIN" ] || die "no 'vllm' on PATH; set VLLM_BIN. This script must run inside the engine image."
  [ -x "$VLLM_BIN" ] || die "VLLM_BIN=${VLLM_BIN} is not executable"

  cd "$REPO_DIR" || die "REPO_DIR=${REPO_DIR} does not exist"

  "$PYTHON_BIN" -c "import tidal.eval.harness" \
    || die "cannot import tidal.eval.harness (is PYTHONPATH=${REPO_DIR}/src set?)"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
      | tee "${LOG_DIR}/nvidia-smi.txt" >&2
  else
    log "WARNING: no nvidia-smi — this looks like a CPU box; numbers will not be GPU-scale"
  fi
}

# Prewarm weights so the first condition's health timeout is not competing with
# a multi-GB download. Best effort: if the CLI is missing, the probe run pays
# the cost instead (its health timeout is 30 min by default).
prewarm_weights() {
  [ "$PREWARM" = "1" ] || { log "prewarm disabled"; return 0; }
  local dl=""
  if command -v hf >/dev/null 2>&1; then
    dl="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    dl="huggingface-cli"
  else
    log "no hf CLI; letting the probe run download ${MODEL}"
    return 0
  fi
  log "prewarming ${MODEL} via ${dl} download"
  if ! "$dl" download "$MODEL" >"${LOG_DIR}/prewarm.log" 2>&1; then
    log "WARNING: prewarm failed (see logs/prewarm.log); continuing — the probe will retry"
  fi
}

# SIGKILL anything still holding the eval port. vLLM V1 forks an EngineCore
# child; a case that died between start and teardown leaves it pinning the port.
reap_port() {
  local port="$1" raw="" pids=""
  if command -v lsof >/dev/null 2>&1; then
    raw="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  elif command -v fuser >/dev/null 2>&1; then
    raw="$(fuser -n tcp "$port" 2>/dev/null || true)"
  fi
  # Keep only bare integers. `fuser` without -n support prints its usage text
  # on stdout, and feeding that to kill is how you take out the wrong process.
  for token in $raw; do
    case "$token" in
      ''|*[!0-9]*) continue ;;
      *) pids="${pids} ${token}" ;;
    esac
  done
  if [ -n "${pids// /}" ]; then
    log "reaping stale listeners on :${port} ->${pids}"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 2
  fi
}

# A result counts as done only if it parses and carries a condition.
result_ok() {
  local path="$1"
  [ -s "$path" ] || return 1
  "$PYTHON_BIN" - "$path" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1]) as fh:
    doc = json.load(fh)
sys.exit(0 if doc.get("condition") else 1)
PY
}

# run_case <case-name> <out-json> <env-assignments-csv> [harness args...]
#
# env-assignments-csv is a ';'-separated K=V list applied to this case only —
# that is how the deadline pair differentiates control from laxity, since both
# arms are `technique_a` with identical harness flags and differ purely in the
# TIDAL_* config the dispatcher reads from the environment.
run_case() {
  local name="$1" out="$2" envs="$3"
  shift 3

  if result_ok "$out"; then
    log "SKIP ${name} (result exists: ${out})"
    SKIPPED_CASES+=("$name")
    return 0
  fi

  reap_port "$EVAL_PORT"

  local case_log="${LOG_DIR}/${name}.log"
  local engine_logs="${LOG_DIR}/${name}-engine"
  mkdir -p "$engine_logs"

  local -a env_args=()
  if [ -n "$envs" ]; then
    # IFS scoped to the read alone — a function-scoped `local IFS` would
    # silently change word-splitting for the rest of this function.
    IFS=';' read -r -a env_args <<<"$envs"
  fi

  local -a timeout_prefix=()
  if command -v timeout >/dev/null 2>&1; then
    timeout_prefix=(timeout --signal=TERM --kill-after=120 "${CASE_TIMEOUT_S}")
  fi

  log "RUN  ${name} -> ${out}"

  # Failure isolation. `cmd || rc=$?` rather than `if ! cmd`, because inside an
  # `if !` branch $? is the negated status (always 0) — the real exit code
  # would be lost.
  local rc=0
  ${timeout_prefix[@]+"${timeout_prefix[@]}"} \
    env ${env_args[@]+"${env_args[@]}"} \
      "$PYTHON_BIN" -m tidal.eval.harness run \
        --model "$MODEL" \
        --port "$EVAL_PORT" \
        --vllm-bin "$VLLM_BIN" \
        --log-dir "$engine_logs" \
        --out "$out" \
        "$@" \
      >"$case_log" 2>&1 || rc=$?

  if [ "$rc" -ne 0 ]; then
    log "FAIL ${name} (exit ${rc}); tail of ${case_log}:"
    tail -n 25 "$case_log" >&2 || true
    FAILED_CASES+=("${name}(exit ${rc})")
    reap_port "$EVAL_PORT"
    return 0
  fi

  tail -n 4 "$case_log" >&2 || true
  RAN_CASES+=("$name")
  reap_port "$EVAL_PORT"
}

# --------------------------------------------------------------------------
# Sizing probe — the offline-only ceiling in items/second.
# --------------------------------------------------------------------------
# Pure stdout. Kept separate from the run_case call because run_case appends to
# the FAILED/RAN arrays, and those writes would be lost inside a $( ) subshell.
read_items_per_s() {
  local probe_out="$1"
  if ! result_ok "$probe_out"; then
    echo "${PROBE_FALLBACK_ITEMS_PER_S:-5.0}"
    return 0
  fi

  "$PYTHON_BIN" - "$probe_out" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    doc = json.load(fh)
batch = doc.get("batch") or {}
completed = float(batch.get("completed") or 0.0)
makespan = float(batch.get("makespan_s") or 0.0)
rate = completed / makespan if completed > 0 and makespan > 0 else 0.0
# A ceiling of zero means the probe served nothing; keep the run going with a
# conservative number rather than sizing every later pool to zero.
print(f"{rate if rate > 0 else 5.0:.4f}")
PY
}

# ceil(a * b * c), floored at MIN_BATCH_ITEMS
size_pool() {
  local rate="$1" seconds="$2" factor="$3" floor="${4:-$MIN_BATCH_ITEMS}"
  "$PYTHON_BIN" - "$rate" "$seconds" "$factor" "$floor" <<'PY'
import math, sys
rate, seconds, factor, floor = (float(a) for a in sys.argv[1:5])
print(max(int(floor), int(math.ceil(rate * seconds * factor))))
PY
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
preflight
prewarm_weights

# The probe doubles as the "wait for engine health" gate: it is the first thing
# that boots vLLM on this GPU, so it is where a bad image, a missing driver, or
# an un-downloadable model surfaces — before 90 minutes of matrix runs.
log "sizing probe: offline_only, pool ${PROBE_ITEMS}"
PROBE_OUT="${PROBE_DIR}/offline_probe.json"
run_case "probe_offline" "$PROBE_OUT" "" \
  --condition offline_only \
  --minutes 1 \
  --batch-items "$PROBE_ITEMS" \
  --batch-max-tokens "$BATCH_MAX_TOKENS" \
  --online-max-tokens "$ONLINE_MAX_TOKENS" \
  --drain-timeout-s "$DRAIN_TIMEOUT_S" \
  --seed "$SEED"

if ! result_ok "$PROBE_OUT"; then
  die "the sizing probe never produced a usable result — the engine did not come up on this box. See ${LOG_DIR}/probe_offline.log and ${LOG_DIR}/probe_offline-engine/."
fi

ITEMS_PER_S="$(read_items_per_s "$PROBE_OUT")"
log "offline ceiling ~= ${ITEMS_PER_S} items/s"

WINDOW_S="$(awk -v m="$MINUTES" 'BEGIN{printf "%.0f", m*60}')"
BATCH_ITEMS="$(size_pool "$ITEMS_PER_S" "$WINDOW_S" "$POOL_SAFETY")"
log "matrix pool size: ${BATCH_ITEMS} items (${POOL_SAFETY}x the ceiling over ${MINUTES} min)"

DIURNAL_WINDOW_S="$(awk -v m="$DIURNAL_MINUTES" 'BEGIN{printf "%.0f", m*60}')"
DIURNAL_ITEMS="$(size_pool "$ITEMS_PER_S" "$DIURNAL_WINDOW_S" "$POOL_SAFETY")"

DEADLINE_WINDOW_S="$(awk -v m="$DEADLINE_MINUTES" 'BEGIN{printf "%.0f", m*60}')"
DEADLINE_ITEMS="$(size_pool "$ITEMS_PER_S" "$DEADLINE_WINDOW_S" "$DEADLINE_TIGHTNESS" 1)"
DEADLINE_WINDOW_HOURS="$(awk -v s="$DEADLINE_WINDOW_S" -v k="$DEADLINE_WINDOW_SLACK" \
  'BEGIN{printf "%.6f", (s*k)/3600}')"
log "deadline pair: ${DEADLINE_ITEMS} items, window ${DEADLINE_WINDOW_HOURS} h"

# -- 1. the five-condition matrix ------------------------------------------
# Identical flags across conditions; the condition name is the only variable.
# Same seed everywhere, so every arm sees the same arrival trace.
for condition in online_only offline_only naive technique_a technique_b; do
  run_case "$condition" "${MATRIX_DIR}/${condition}.json" "" \
    --condition "$condition" \
    --minutes "$MINUTES" \
    --online-rps "$ONLINE_RPS" \
    --batch-items "$BATCH_ITEMS" \
    --online-max-tokens "$ONLINE_MAX_TOKENS" \
    --batch-max-tokens "$BATCH_MAX_TOKENS" \
    --max-inflight "$MAX_INFLIGHT" \
    --drain-timeout-s "$DRAIN_TIMEOUT_S" \
    --arrival poisson \
    --seed "$SEED"
done

# -- 2. diurnal ------------------------------------------------------------
# The tide-filling claim: batch completions should anti-correlate with online
# arrivals. Run both placements so the correlation can be compared.
for condition in technique_a technique_b; do
  run_case "diurnal_${condition}" "${DIURNAL_DIR}/diurnal_${condition}.json" "" \
    --condition "$condition" \
    --minutes "$DIURNAL_MINUTES" \
    --online-rps "$ONLINE_RPS" \
    --batch-items "$DIURNAL_ITEMS" \
    --online-max-tokens "$ONLINE_MAX_TOKENS" \
    --batch-max-tokens "$BATCH_MAX_TOKENS" \
    --max-inflight "$MAX_INFLIGHT" \
    --drain-timeout-s "$DRAIN_TIMEOUT_S" \
    --arrival diurnal \
    --seed "$SEED"
done

# -- 3. deadline stress: control vs laxity ---------------------------------
# Both arms are technique_a with byte-identical harness flags. What differs is
# the dispatcher's TIDAL_* config:
#   control — escalation horizon collapsed to 1 s (a batch effectively never
#             escalates out of the lowest band) and the progress floor pushed
#             out of reach (floor_urgency > 1 can never be met). Best-effort
#             AIMD co-location, which is the thing laxity has to beat.
#   laxity  — escalation over the last 10 min of slack, floor at the default.
# Admission feasibility is disabled on both arms (margin 0), otherwise the
# tight job is refused at submission and there is nothing to measure.
DEADLINE_COMMON=(
  --condition technique_a
  --minutes "$DEADLINE_MINUTES"
  --online-rps "$ONLINE_RPS"
  --batch-items "$DEADLINE_ITEMS"
  --online-max-tokens "$ONLINE_MAX_TOKENS"
  --batch-max-tokens "$BATCH_MAX_TOKENS"
  --max-inflight "$MAX_INFLIGHT"
  --drain-timeout-s "$DRAIN_TIMEOUT_S"
  --arrival diurnal
  --seed "$SEED"
)

run_case "deadline_control" "${DEADLINE_DIR}/deadline_control.json" \
  "TIDAL_COMPLETION_WINDOW_HOURS=${DEADLINE_WINDOW_HOURS};TIDAL_ESCALATION_HORIZON_S=1;TIDAL_FLOOR_URGENCY=2.0;TIDAL_ADMISSION_FEASIBILITY_MARGIN=0" \
  "${DEADLINE_COMMON[@]}"

run_case "deadline_laxity" "${DEADLINE_DIR}/deadline_laxity.json" \
  "TIDAL_COMPLETION_WINDOW_HOURS=${DEADLINE_WINDOW_HOURS};TIDAL_ESCALATION_HORIZON_S=600;TIDAL_FLOOR_URGENCY=0.9;TIDAL_ADMISSION_FEASIBILITY_MARGIN=0" \
  "${DEADLINE_COMMON[@]}"

# -- 4. figures ------------------------------------------------------------
# Only the matrix directory: plots.load_results keys by the payload's
# `condition`, so the diurnal and deadline runs (all technique_a/technique_b)
# would collide with the matrix arms if they shared a directory.
log "rendering figures"
if ! "$PYTHON_BIN" -m tidal.eval.plots render \
      --results-dir "$MATRIX_DIR" \
      --out-dir "${MATRIX_DIR}/figures" >"${LOG_DIR}/plots.log" 2>&1; then
  log "WARNING: figure rendering failed; see logs/plots.log"
  FAILED_CASES+=("plots")
fi

# -- 5. manifest + tarball -------------------------------------------------
{
  echo "tidal gpu eval"
  echo "host:              ${HOSTNAME_TAG}"
  echo "utc:               ${STAMP}"
  echo "model:             ${MODEL}"
  echo "vllm_bin:          ${VLLM_BIN}"
  echo "online_rps:        ${ONLINE_RPS}"
  echo "minutes:           ${MINUTES}"
  echo "online_max_tokens: ${ONLINE_MAX_TOKENS}"
  echo "batch_max_tokens:  ${BATCH_MAX_TOKENS}"
  echo "max_inflight:      ${MAX_INFLIGHT}"
  echo "items_per_s_probe: ${ITEMS_PER_S}"
  echo "batch_items:       ${BATCH_ITEMS}"
  echo "diurnal_items:     ${DIURNAL_ITEMS}"
  echo "deadline_items:    ${DEADLINE_ITEMS}"
  echo "deadline_window_h: ${DEADLINE_WINDOW_HOURS}"
  echo "ran:               ${RAN_CASES[*]:-none}"
  echo "skipped(resumed):  ${SKIPPED_CASES[*]:-none}"
  echo "failed:            ${FAILED_CASES[*]:-none}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
  fi
} >"${RUN_DIR}/MANIFEST.txt"

cat "${RUN_DIR}/MANIFEST.txt" >&2

if [ "$MAKE_TARBALL" = "1" ]; then
  TARBALL="${RESULTS_BASE}/tidal-${RUN_NAME}.tar.gz"
  log "packing ${TARBALL}"
  tar -czf "$TARBALL" -C "$RESULTS_BASE" "$RUN_NAME"
  log "wrote ${TARBALL}"
fi

if [ "${#FAILED_CASES[@]}" -gt 0 ]; then
  log "completed with ${#FAILED_CASES[@]} failed case(s): ${FAILED_CASES[*]}"
  log "re-run this script to retry only the failures (successful results are skipped)"
  exit 1
fi

log "all cases completed"
