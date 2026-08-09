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
#     -e TENSOR_PARALLEL_SIZE=8 -e MAX_MODEL_LEN=8192 \
#     --entrypoint tidal-run-gpu-eval tidal-engine:fast
#
# Every case runs the harness with `--gpu-preset` plus `--tensor-parallel` and
# `--max-model-len` from the environment. Without those the harness would run
# its Mac-CPU defaults on the GPU: eager-only, torchdynamo disabled, one card,
# 4096 context, and batch concurrency capped by httpx at ~100.
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
# CASES trims that list — see the "Case selector" block below. The canary and
# the sizing probe are outside it and always run. WARMUP_S buys unmeasured
# requests between /health and t0 so first-condition warm-up costs are not
# reported as scheduler-induced latency.
#
# Robustness contract:
#   * set -euo pipefail
#   * every case is failure-isolated — one bad condition does not abort the run
#   * resumable — a case whose result JSON already parses is skipped, so a
#     re-invocation after an OOM or a dropped SSH session picks up where it left
#     off. Delete the JSON to force a re-run.

set -euo pipefail

# No TTY is ever required: stdin is closed up front, so this runs identically
# under `docker run` without -it, under nohup, and as the selfdrive supervisor's
# child. Anything that tried to prompt gets EOF instead of hanging forever.
exec </dev/null

# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/opt/tidal}"
# RESULTS_DIR is the name the selfdrive supervisor (src/tidal/eval/selfdrive.py)
# exports; RESULTS_BASE is the older name this script used. Either works, and
# the supervisor always sets both to the same path so the HTTP result endpoints
# and this script can never disagree about where things landed.
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_BASE:-/results}}"
RESULTS_BASE="$RESULTS_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EVAL_PORT="${EVAL_PORT:-8400}"

# Engine shape. The harness defaults describe the Mac CPU testbed it was
# written on; --gpu-preset flips the three that are actively wrong on a GPU
# (eager-only, torchdynamo off, offline hub, a ~100-connection batch pool).
# TP and context length are node-specific, so they are passed explicitly and
# share names with docker-compose.gpu.yml / engine-entrypoint.sh.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# Empty means "whatever --gpu-preset says" (512). Set it to override.
BATCH_CONCURRENCY="${BATCH_CONCURRENCY:-}"
# Measurement warm-up: unmeasured sequential requests fired after the engine's
# /health goes green and before t0, so CUDA-graph capture, torch.compile and
# the cold prefix cache are not charged to the first condition's latency
# numbers. Empty means "whatever --gpu-preset says" (45 s); 0 disables it.
WARMUP_S="${WARMUP_S:-}"

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
# schedule and simply drains the pool. The harness fires the whole pool at once
# through one httpx.AsyncClient, so the ceiling is measured at
# --batch-concurrency-way concurrency — the same bound every co-located
# condition gets, because every case below is passed the same value.
PROBE_ITEMS="${PROBE_ITEMS:-600}"
# Pool multiple over "how many items the ceiling could do in the window". >1
# guarantees the pool never drains, which is what makes batch throughput a
# measurement of the scheduler rather than of the pool size. The harness prints
# a NOTE when a pool drains; that NOTE means raise this.
POOL_SAFETY="${POOL_SAFETY:-1.15}"
# 1.15x: never drains inside the window (probe error margin), while keeping the
# post-window drain of direct-submission arms under a minute — a 2.0x pool live-
# wedged offline_only for 80+ min (23k in-flight futures vs a 6-min window).
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
CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-$(( MINUTES * 60 + ${DRAIN_TIMEOUT_S:-900} + 900 ))}"

# --------------------------------------------------------------------------
# Case selector
# --------------------------------------------------------------------------
# CASES is a comma-separated subset of ALL_CASES; anything not listed is not
# run at all, and `PROGRESS total <n>` counts only what will actually run. An
# unknown name is fatal in preflight, before a second of GPU time is spent —
# a typo that silently ran the full 3.6 h matrix would be the expensive
# failure mode.
#
# Two things are NOT selectable and always run:
#   * the compat canary (RUN_CANARY=0 is its own switch)
#   * the sizing probe — it is the engine-health gate *and* the divisor every
#     pool size is computed from. In particular, dropping `offline_only` from
#     CASES does not drop the probe: the probe is its own `offline_only` run
#     against the same engine shape, so the pools stay correctly sized even
#     when the measured offline_only condition is not part of this invocation.
#
# Order is fixed by the script (matrix, then diurnal, then deadline), not by
# the order names appear in CASES. Duplicates and surrounding whitespace are
# tolerated. Figures are rendered only when at least one matrix case is
# selected — `plots.load_results` reads the matrix directory, and there is
# nothing to re-render for a deadline-only invocation.
ALL_CASES="online_only,offline_only,naive,technique_a,technique_b,diurnal_technique_a,diurnal_technique_b,deadline_control,deadline_laxity"
CASES="${CASES:-$ALL_CASES}"
SELECTED_CASES=()

PREWARM="${PREWARM:-1}"
MAKE_TARBALL="${MAKE_TARBALL:-1}"
# The 10-minute compat canary, run before any GPU time is spent. Set to 0 only
# if you have already run it against this exact image.
RUN_CANARY="${RUN_CANARY:-1}"
# RESUME=1 reuses the newest gpu-<host>-* run directory instead of starting a
# fresh one, which is what makes "just re-run it" pick up after a crash. Set
# RESUME=0 (or pass an explicit RUN_DIR) for a clean run.
RESUME="${RESUME:-1}"

# The harness defaults this to a path on the author's Mac.
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"

# HF_HUB_OFFLINE is no longer exported here: --gpu-preset implies
# --no-hf-offline, and the harness now sets the variable for the engine from
# the flag rather than inheriting it. Pass --hf-offline explicitly (see
# HARNESS_ENGINE_ARGS below) if you want a pinned, air-gapped run.
export TIDAL_EVAL_MODEL="$MODEL"
export TIDAL_EVAL_HEALTH_TIMEOUT_S="${TIDAL_EVAL_HEALTH_TIMEOUT_S:-1800}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Applied to every case, including the sizing probe — the ceiling is only a
# valid divisor for the conditions it sizes if it was measured on the same
# engine. Append to this to force an individual flag: it is read left to right
# by typer, and an explicit flag beats --gpu-preset regardless of position.
HARNESS_ENGINE_ARGS=(
  --gpu-preset
  --tensor-parallel "$TENSOR_PARALLEL_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
)
if [ -n "$BATCH_CONCURRENCY" ]; then
  HARNESS_ENGINE_ARGS+=(--batch-concurrency "$BATCH_CONCURRENCY")
fi
if [ -n "$WARMUP_S" ]; then
  HARNESS_ENGINE_ARGS+=(--warmup-s "$WARMUP_S")
fi

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

# Machine-readable state changes, for src/tidal/eval/selfdrive.py's /status.
# One token per line, stderr like log() so ordering in a merged log is sane:
#   booting | canary | probing | running:<case> | rendering | done
#   total <n> | case_done <name> | case_failed <name>
#   canary_ok | canary_failed <msg> | failed <msg>
# A human reading the log loses nothing; a supervisor parsing it needs no regex
# against prose that might get reworded later.
progress() {
  printf 'PROGRESS %s\n' "$*" >&2
}

die() {
  log "FATAL: $*"
  # The supervisor's /status shows this verbatim; it is often the only thing an
  # operator on the far end of a public URL will ever see.
  progress "failed $*"
  exit 1
}

# --------------------------------------------------------------------------
# Case selection
# --------------------------------------------------------------------------
# Membership test against the parsed selection. Used as `if case_selected x`
# rather than `case_selected x && ...`: under `set -e` a bare `&&` whose left
# side is false is a failing compound, and as the last statement of a loop body
# that aborts the script.
case_selected() {
  local needle="$1" name
  for name in ${SELECTED_CASES[@]+"${SELECTED_CASES[@]}"}; do
    [ "$name" = "$needle" ] && return 0
  done
  return 1
}

# CASES -> SELECTED_CASES, or die with the valid list. Every unknown name is
# reported at once: fixing one typo only to hit the next one on the following
# run is a bad trade when a run costs GPU-hours.
parse_cases() {
  local raw="$1" name wanted=","
  local -a requested=() bad=()
  SELECTED_CASES=()
  IFS=',' read -r -a requested <<<"$raw"
  for name in ${requested[@]+"${requested[@]}"}; do
    name="$(printf '%s' "$name" | tr -d '[:space:]')"
    [ -n "$name" ] || continue
    case ",${ALL_CASES}," in
      *",${name},"*) wanted="${wanted}${name}," ;;
      *) bad+=("$name") ;;
    esac
  done
  # Rebuilt in ALL_CASES order rather than in the order the operator typed, so
  # `cases:` in MANIFEST.txt reads in the order the run will actually execute,
  # and duplicates collapse for free.
  for name in ${ALL_CASES//,/ }; do
    case "$wanted" in
      *",${name},"*) SELECTED_CASES+=("$name") ;;
    esac
  done
  if [ "${#bad[@]}" -gt 0 ]; then
    die "unknown case(s) in CASES: ${bad[*]} -- valid cases are: ${ALL_CASES//,/ } (the compat canary and the sizing probe always run and are not selectable)"
  fi
  if [ "${#SELECTED_CASES[@]}" -eq 0 ]; then
    die "CASES selected nothing -- valid cases are: ${ALL_CASES//,/ }"
  fi
  log "cases: ${SELECTED_CASES[*]}"
}

# The /status denominator: the always-run sizing probe plus everything CASES
# selected, less the technique-B cases a failed canary takes off the table.
compute_total_cases() {
  local total=$((1 + ${#SELECTED_CASES[@]}))
  if [ "$CANARY_OK" != "1" ]; then
    if case_selected technique_b; then total=$((total - 1)); fi
    if case_selected diurnal_technique_b; then total=$((total - 1)); fi
  fi
  printf '%s\n' "$total"
}

# True when the figure step has something to render.
any_matrix_case_selected() {
  local name
  for name in online_only offline_only naive technique_a technique_b; do
    if case_selected "$name"; then return 0; fi
  done
  return 1
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
preflight() {
  # First, before anything else can cost time: a bad CASES must fail in
  # seconds, not after the canary and a multi-GB weight download.
  parse_cases "$CASES"

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

# The 10-minute compat canary. It exercises TidalScheduler against *this
# image's* vLLM using real Scheduler machinery and real KV block accounting with
# no weights loaded, so it costs minutes rather than an evaluation's worth of
# GPU hours. Running it here rather than by hand is the whole point of the
# SSH-less flow: nobody is around to run it first.
#
# On failure the run does NOT stop. Only technique_b depends on the plugin, so
# the technique_a matrix, the diurnal technique_a arm and the deadline pair are
# still worth the GPU time. What changes:
#   * CANARY_OK=0 -> both technique_b cases are skipped
#   * PROGRESS canary_failed <msg> -> /status carries the error from here on
#   * the failure joins FAILED_CASES, so the script exits non-zero and the
#     supervisor's terminal phase is `failed` rather than `done`
#   * MANIFEST.txt records it, so a downloaded tarball is self-describing
#
# `python3 -m pytest` rather than bare `pytest` is required, not stylistic:
# tests/ has no __init__.py, so `from tests.unit.engine_fixtures import ...`
# only resolves because -m puts the working directory on sys.path.
CANARY_OK=1
CANARY_NOTE="ok"

run_compat_canary() {
  progress "canary"
  if [ "$RUN_CANARY" != "1" ]; then
    CANARY_NOTE="skipped (RUN_CANARY=0)"
    log "compat canary disabled (RUN_CANARY=0)"
    return 0
  fi

  local out="${LOG_DIR}/compat_canary.log"
  local rc=0
  log "compat canary: tests/unit/test_tidal_scheduler.py"
  (cd "$REPO_DIR" && "$PYTHON_BIN" -m pytest tests/unit/test_tidal_scheduler.py -q) \
    >"$out" 2>&1 || rc=$?

  # Three distinguishable outcomes, because the fix differs for each:
  #   exit 5 / no passing tests -> vLLM did not import, the image is broken
  #   any other non-zero        -> upstream drifted away from the plan-B pin
  #   green                     -> proceed
  local msg=""
  if [ "$rc" -eq 5 ] || { [ "$rc" -eq 0 ] && ! grep -qE '[0-9]+ passed' "$out"; }; then
    rc=5
    msg="compat canary exercised nothing (exit 5 / all skipped): vLLM did not import in this image, so TidalScheduler was never tested. Rebuild the image. technique_b is skipped; see logs/compat_canary.log"
  elif [ "$rc" -ne 0 ]; then
    msg="compat canary FAILED (exit ${rc}): this image's vLLM has drifted from the TidalScheduler interface points. technique_b is not trustworthy here and is skipped; see logs/compat_canary.log"
  fi

  if [ "$rc" -ne 0 ]; then
    CANARY_OK=0
    CANARY_NOTE="FAILED - ${msg}"
    progress "canary_failed ${msg}"
    log "$msg"
    tail -n 20 "$out" >&2 || true
    FAILED_CASES+=("compat_canary(exit ${rc})")
    return 0
  fi

  progress "canary_ok"
  log "compat canary green"
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
    # A resumed case is a completed case as far as progress is concerned: its
    # result JSON is in the tarball either way.
    progress "case_done ${name}"
    return 0
  fi

  # The sizing probe gets its own phase name because it is not one of the
  # measured conditions — it is the engine-health gate in front of them.
  if [ "$name" = "probe_offline" ]; then
    progress "probing"
  else
    progress "running:${name}"
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
        "${HARNESS_ENGINE_ARGS[@]}" \
        "$@" \
      >"$case_log" 2>&1 || rc=$?

  if [ "$rc" -ne 0 ]; then
    log "FAIL ${name} (exit ${rc}); tail of ${case_log}:"
    tail -n 25 "$case_log" >&2 || true
    FAILED_CASES+=("${name}(exit ${rc})")
    progress "case_failed ${name}"
    reap_port "$EVAL_PORT"
    return 0
  fi

  tail -n 4 "$case_log" >&2 || true
  RAN_CASES+=("$name")
  progress "case_done ${name}"
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
# Test seam
# --------------------------------------------------------------------------
# `TIDAL_EVAL_SOURCE_ONLY=1 . run_gpu_eval.sh` stops here with every function
# defined and nothing executed, so tests/unit/test_run_gpu_eval_cases.py can
# call parse_cases / compute_total_cases directly instead of re-implementing
# the selector in the test and asserting against a copy. Unset (the only way it
# is ever run for real) this block does nothing.
if [ "${TIDAL_EVAL_SOURCE_ONLY:-0}" = "1" ]; then
  # `return` when sourced; `exit` if someone sets the variable while executing
  # the script normally, where a top-level `return` is an error. shellcheck
  # cannot see the sourced case, so the fallback looks unreachable to it.
  # shellcheck disable=SC2317
  return 0 2>/dev/null || exit 0
fi

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
progress "booting"
preflight
# Before the weight download, not after: the canary takes minutes, prewarm can
# take half an hour, and an operator watching /status wants the verdict early.
run_compat_canary

# Denominator for /status. The always-run probe plus whatever CASES selected
# (by default 5 matrix + 2 diurnal + 2 deadline = 10 in total), less the two
# technique_b cases a failed canary takes off the table.
TOTAL_CASES="$(compute_total_cases)"
progress "total ${TOTAL_CASES}"

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
  if ! case_selected "$condition"; then
    log "SKIP ${condition} (not in CASES)"
    continue
  fi
  if [ "$condition" = "technique_b" ] && [ "$CANARY_OK" != "1" ]; then
    log "SKIP technique_b — the compat canary failed; the plugin's numbers would be meaningless"
    SKIPPED_CASES+=("technique_b(canary)")
    continue
  fi
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
  if ! case_selected "diurnal_${condition}"; then
    log "SKIP diurnal_${condition} (not in CASES)"
    continue
  fi
  if [ "$condition" = "technique_b" ] && [ "$CANARY_OK" != "1" ]; then
    SKIPPED_CASES+=("diurnal_technique_b(canary)")
    continue
  fi
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

if case_selected deadline_control; then
  run_case "deadline_control" "${DEADLINE_DIR}/deadline_control.json" \
    "TIDAL_COMPLETION_WINDOW_HOURS=${DEADLINE_WINDOW_HOURS};TIDAL_ESCALATION_HORIZON_S=1;TIDAL_FLOOR_URGENCY=2.0;TIDAL_ADMISSION_FEASIBILITY_MARGIN=0" \
    "${DEADLINE_COMMON[@]}"
else
  log "SKIP deadline_control (not in CASES)"
fi

if case_selected deadline_laxity; then
  run_case "deadline_laxity" "${DEADLINE_DIR}/deadline_laxity.json" \
    "TIDAL_COMPLETION_WINDOW_HOURS=${DEADLINE_WINDOW_HOURS};TIDAL_ESCALATION_HORIZON_S=600;TIDAL_FLOOR_URGENCY=0.9;TIDAL_ADMISSION_FEASIBILITY_MARGIN=0" \
    "${DEADLINE_COMMON[@]}"
else
  log "SKIP deadline_laxity (not in CASES)"
fi

# -- 4. figures ------------------------------------------------------------
# Only the matrix directory: plots.load_results keys by the payload's
# `condition`, so the diurnal and deadline runs (all technique_a/technique_b)
# would collide with the matrix arms if they shared a directory.
progress "rendering"
if any_matrix_case_selected; then
  log "rendering figures"
  if ! "$PYTHON_BIN" -m tidal.eval.plots render \
        --results-dir "$MATRIX_DIR" \
        --out-dir "${MATRIX_DIR}/figures" >"${LOG_DIR}/plots.log" 2>&1; then
    log "WARNING: figure rendering failed; see logs/plots.log"
    FAILED_CASES+=("plots")
  fi
else
  # Nothing in this invocation wrote to matrix/, so there is nothing new to
  # draw, and a render over an empty directory would only manufacture a
  # failure for a run that did exactly what it was asked to.
  log "skipping figures — CASES selected no matrix case"
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
  echo "engine_args:       ${HARNESS_ENGINE_ARGS[*]}"
  echo "tensor_parallel:   ${TENSOR_PARALLEL_SIZE}"
  echo "max_model_len:     ${MAX_MODEL_LEN}"
  echo "cases:             ${SELECTED_CASES[*]}"
  echo "cases_env:         ${CASES}"
  echo "canary:            ${CANARY_NOTE}"
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
  # Results are packed and servable regardless — the supervisor keeps serving
  # after a non-zero exit, so `failed` here means "download it and read the
  # manifest", not "nothing to see".
  progress "failed ${#FAILED_CASES[@]} case(s) failed: ${FAILED_CASES[*]}"
  exit 1
fi

progress "done"
log "all cases completed"
