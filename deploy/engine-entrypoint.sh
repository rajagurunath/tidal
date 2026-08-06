#!/usr/bin/env bash
# Engine entrypoint for the Tidal GPU image.
#
# Translates a small, CaaS-friendly set of environment variables into a
# `vllm serve` command line. The io.net CaaS payload can set env vars far more
# comfortably than it can set args (args are a flat JSON array that has to be
# rewritten for every knob), so everything routine lives in env here.
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

SCHEDULER_MODE="${SCHEDULER_MODE:-stock}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TBT_SLO="${TBT_SLO:-200}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-priority}"

# TidalScheduler reads TidalConfig.from_env(); TBT_SLO is the one knob worth a
# short alias because it is the headline dial of technique B.
export TIDAL_TBT_SLO_MS="${TIDAL_TBT_SLO_MS:-$TBT_SLO}"

# `tidal` is pip-installed in the image, but --scheduler-cls is resolved inside
# the EngineCore subprocess; keeping /opt/tidal/src on PYTHONPATH makes the
# import work even if the install layer is ever shadowed or overlaid.
export PYTHONPATH="/opt/tidal/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

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

echo "engine-entrypoint: mode=${SCHEDULER_MODE} model=${MODEL} tbt_slo_ms=${TIDAL_TBT_SLO_MS}" >&2
echo "engine-entrypoint: exec vllm ${args[*]}" >&2

# Never exec from a source directory: a local vllm/ or tidal/ would shadow the
# installed package (learned the hard way on the dev box).
cd /
exec vllm "${args[@]}"
