#!/usr/bin/env bash
#
# Technique A across a fleet: the treatment run and its control, back to back.
#
#   scripts/run_fleet_experiment.sh
#
# Two runs, identical in every respect except where batch work is allowed to go:
#
#   results/fleet_a.json       --fleet-placement fleet   (load-aware placement)
#   results/fleet_pinned.json  --fleet-placement pinned  (all batch on replica 0)
#
# The pinned arm is the control the fleet policy has to beat. Both drive the
# same N engines with the same phase-shifted diurnal load, so any difference in
# online latency or batch throughput is placement and nothing else.
#
# THIS SCRIPT BOOTS NOTHING ITSELF. `tidal.eval.harness` owns its engines: it
# launches one stock `vllm serve` per replica on FLEET_BASE_PORT + i, waits for
# each /health, runs the window, and tears the whole process group down. Any
# vLLM already listening on those ports will make the run fail at launch — kill
# it first.
#
# macOS core isolation, stated plainly: `taskset` does not exist here, and
# neither does user-level CPU affinity (vLLM's own CPU-list builder says so:
# "For MacOS, no user-level CPU affinity and SMT, return all CPUs"). The
# harness still sets a disjoint VLLM_CPU_OMP_THREADS_BIND range per replica,
# because vLLM derives OMP_NUM_THREADS from the width of that range — so on a
# Mac this is *thread-count* isolation, and on Linux it is thread-count and
# core-pinning both. See `replica_env` in src/tidal/eval/harness.py.

set -euo pipefail
exec </dev/null  # no TTY is ever required

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"

# -- run shape (every one is an env override) -------------------------------
REPLICAS="${REPLICAS:-2}"
FLEET_BASE_PORT="${FLEET_BASE_PORT:-8399}"
MINUTES="${MINUTES:-20}"
ONLINE_RPS="${ONLINE_RPS:-1.0}"
BATCH_ITEMS="${BATCH_ITEMS:-4000}"
ARRIVAL="${ARRIVAL:-diurnal}"
SEED="${SEED:-7}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"
DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S:-300}"
# Empty = spread the replicas evenly over the period (two replicas ⇒ half a
# period apart, which is the arrangement the whole fleet thesis rests on).
DIURNAL_PHASE_LIST="${DIURNAL_PHASE_LIST:-}"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/fleet-logs}"

# -- Mac CPU engine environment ---------------------------------------------
# The same set tests/integration/conftest.py and the harness launch vLLM with:
# eager only (inductor is a non-starter on the CPU build), the hub offline, and
# unbuffered stdout so the engine log is readable while the run is in flight.
# The per-replica core/KV split is applied by the harness itself, per engine.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TIDAL_EVAL_VLLM_BIN="${TIDAL_EVAL_VLLM_BIN:-/Users/gurunathlunkupalivenugopal/ionet/repos/vllm-experiments/.venv/bin/vllm}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

run_case() {
  local placement="$1" out="$2"
  echo "=== technique_a_fleet [$placement] -> $out"
  "$PYTHON" -m tidal.eval.harness run \
    --condition technique_a_fleet \
    --replicas "$REPLICAS" \
    --fleet-base-port "$FLEET_BASE_PORT" \
    --fleet-placement "$placement" \
    --diurnal-phase-list "$DIURNAL_PHASE_LIST" \
    --arrival "$ARRIVAL" \
    --minutes "$MINUTES" \
    --online-rps "$ONLINE_RPS" \
    --batch-items "$BATCH_ITEMS" \
    --max-inflight "$MAX_INFLIGHT" \
    --drain-timeout-s "$DRAIN_TIMEOUT_S" \
    --model "$MODEL" \
    --seed "$SEED" \
    --log-dir "$LOG_DIR" \
    --out "$out"
}

# The treatment first: if the box cannot run the fleet at all, that is cheaper
# to learn from the arm the experiment is actually about.
run_case fleet "$RESULTS_DIR/fleet_a.json"
run_case pinned "$RESULTS_DIR/fleet_pinned.json"

echo
echo "wrote:"
echo "  $RESULTS_DIR/fleet_a.json       (load-aware placement)"
echo "  $RESULTS_DIR/fleet_pinned.json  (control: all batch on replica 0)"
echo "engine logs: $LOG_DIR"
