#!/usr/bin/env bash
# Entrypoint for the Tidal GPU image. Two jobs behind one binary.
#
# MODE=selfdrive (the default under CAAS=1)
#   exec `python -m tidal.eval.selfdrive`: a supervisor that runs the ENTIRE
#   evaluation unattended and serves progress + results over $PORT. This is the
#   only mode that works on io.net CaaS, where there is no SSH, no exec, and no
#   way to retrieve a volume — the container has to hand its own results back
#   over the single published port. Nothing else needs to be running: the eval
#   harness launches and kills its own `vllm serve` per condition.
#
# MODE=serve (the default outside CaaS)
#   Translate a small, CaaS-friendly set of environment variables into a
#   `vllm serve` command line, for docker-compose.gpu.yml and dev boxes. Env
#   rather than args because a CaaS payload's `args` is a flat JSON array that
#   has to be rewritten for every knob.
#
#   MODE             selfdrive | serve  what this container is for
#   CAAS             1                  shorthand for MODE=selfdrive
#   PORT             int                bind port (CaaS traffic_port); both modes
#
# selfdrive mode additionally reads (see src/tidal/eval/selfdrive.py and
# deploy/run_gpu_eval.sh for the full list):
#
#   RESULTS_DIR      path               where results land (default /results)
#   TIDAL_API_KEY    str                required by POST /abort
#   MODEL / TENSOR_PARALLEL_SIZE / MAX_MODEL_LEN / ONLINE_RPS / MINUTES /
#   BATCH_CONCURRENCY                   the eval parameterization
#
# serve mode reads:
#
#   SCHEDULER_MODE   stock | tidal      technique A vs technique B (default stock)
#   MODEL            HF model id        what to serve
#   MAX_MODEL_LEN    int                --max-model-len
#   TBT_SLO          ms                 TidalScheduler's time-between-tokens SLO
#   TENSOR_PARALLEL_SIZE int            --tensor-parallel-size
#   PORT             int                bind port (CaaS traffic_port)
#   VLLM_API_KEY     str                --api-key, omitted when empty
#   SCHEDULING_POLICY priority|fcfs     both techniques need `priority`
#   EXTRA_VLLM_ARGS  str                appended verbatim, word-split
#
# SCHEDULER_MODE=tidal is the only mode that requires the vLLM in this image to
# be API-compatible with the TidalScheduler plugin. The scheduler probes for the
# upstream hooks it needs at startup and degrades mechanism-by-mechanism rather
# than crashing, so a drifted vLLM shows up as a downgraded `capabilities=` line
# in the log, not as a failed boot. Grep it — see deploy/README.md.

set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# `tidal` is pip-installed in the image, but --scheduler-cls is resolved inside
# the EngineCore subprocess; keeping /opt/tidal/src on PYTHONPATH makes the
# import work even if the install layer is ever shadowed or overlaid.
export PYTHONPATH="/opt/tidal/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------
# Mode selection
# --------------------------------------------------------------------------
# CaaS gives one container and one public port and takes everything else away,
# so selfdrive is the right default there and only there: docker-compose still
# wants a serving engine, and it does not set CAAS.
if [ -z "${MODE:-}" ]; then
  if [ "${CAAS:-0}" = "1" ]; then
    MODE=selfdrive
  else
    MODE=serve
  fi
fi

case "$MODE" in
  selfdrive)
    export PORT HOST
    export RESULTS_DIR="${RESULTS_DIR:-/results}"
    export REPO_DIR="${REPO_DIR:-/opt/tidal}"
    mkdir -p "$RESULTS_DIR"
    echo "engine-entrypoint: mode=selfdrive port=${PORT} results=${RESULTS_DIR} model=${MODEL:-<image default>}" >&2
    echo "engine-entrypoint: exec python3 -m tidal.eval.selfdrive" >&2
    # Not from /opt/tidal: a local `vllm/` or `tidal/` directory would shadow
    # the installed package (learned the hard way on the dev box). The
    # supervisor cds the eval child into REPO_DIR itself.
    cd /
    exec python3 -m tidal.eval.selfdrive
    ;;
  serve|engine)
    ;;
  *)
    echo "engine-entrypoint: MODE must be 'selfdrive' or 'serve', got '${MODE}'" >&2
    exit 2
    ;;
esac

# --------------------------------------------------------------------------
# MODE=serve — vllm serve, for docker-compose.gpu.yml and dev boxes
# --------------------------------------------------------------------------
SCHEDULER_MODE="${SCHEDULER_MODE:-stock}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TBT_SLO="${TBT_SLO:-200}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-priority}"

# TidalScheduler reads TidalConfig.from_env(); TBT_SLO is the one knob worth a
# short alias because it is the headline dial of technique B.
export TIDAL_TBT_SLO_MS="${TIDAL_TBT_SLO_MS:-$TBT_SLO}"

args=(
  serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --scheduling-policy "$SCHEDULING_POLICY"
)

case "$SCHEDULER_MODE" in
  stock)
    ;;
  tidal)
    args+=(--scheduler-cls tidal.engine.scheduler.TidalScheduler)
    ;;
  *)
    echo "engine-entrypoint: SCHEDULER_MODE must be 'stock' or 'tidal', got '${SCHEDULER_MODE}'" >&2
    exit 2
    ;;
esac

if [ -n "${VLLM_API_KEY:-}" ]; then
  args+=(--api-key "$VLLM_API_KEY")
fi

if [ -n "${EXTRA_VLLM_ARGS:-}" ]; then
  # Deliberate word-splitting: EXTRA_VLLM_ARGS is an operator-supplied arg list.
  read -r -a extra_args <<<"$EXTRA_VLLM_ARGS"
  args+=("${extra_args[@]}")
fi

# Anything after the entrypoint (CaaS `args`, `docker run ... -- --foo`) wins.
if [ "$#" -gt 0 ]; then
  args+=("$@")
fi

echo "engine-entrypoint: mode=serve scheduler=${SCHEDULER_MODE} model=${MODEL} tbt_slo_ms=${TIDAL_TBT_SLO_MS}" >&2
echo "engine-entrypoint: exec vllm ${args[*]}" >&2

# Never exec from a source directory: a local vllm/ or tidal/ would shadow the
# installed package (learned the hard way on the dev box).
cd /
exec vllm "${args[@]}"
