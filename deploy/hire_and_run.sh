#!/usr/bin/env bash
#
# hire_and_run.sh — one-shot io.net CaaS *marketplace* hire that runs the whole
# Tidal GPU evaluation and hands the results back, then destroys the rental.
#
# This is the marketplace sibling of the deploy-to-specific-device flow in
# deploy/README.md. There, you already own a node and you know its device_id.
# Here you own nothing: the script shops the /hardware catalogue, picks a card
# that fits the budget, pays for INITIAL_HOURS up front, boots a stock
# `vllm/vllm-openai` image, installs Tidal into it from a pinned SHA, lets
# `tidal.eval.selfdrive` drive the entire evaluation, downloads the tarball,
# and DELETEs the deployment.
#
#   export CAAS_API_KEY=...            # never printed, never in argv
#   deploy/hire_and_run.sh
#
# Every decision, request and poll is timestamped into $EVIDENCE_DIR/driver.log,
# and the raw JSON of every API call is kept next to it. If this run turns into
# a paper number, the evidence directory is the receipt.
#
# WHAT THIS COSTS YOU IF IT GOES WRONG: the marketplace charges the full
# `duration_hours` up front and does not refund an early DELETE. A crashed
# driver that leaks a deployment therefore costs the same as a successful one —
# which is why teardown runs from an EXIT/INT/TERM trap and not from the happy
# path. `KEEP=1` is the only way to keep a deployment alive past this script.
#
# EXIT CODES
#   0  the eval reached phase=done and the tarball was fetched
#   2  hire failed (no API reachability, no eligible hardware, deploy rejected)
#   3  boot timeout (no running container with a public_url inside BOOT_TIMEOUT_S)
#   4  the eval failed (phase=failed/aborted, or the eval timed out) — results
#      are still fetched, best effort, before the deployment is destroyed
#   5  budget abort (the /price estimate exceeds the spend cap; nothing hired)
#
# SECRET DISCIPLINE
#   $CAAS_API_KEY is written once, by a shell builtin, into a 0600 curl config
#   file and passed to curl as `--config`. It never appears in this script's
#   argv, in curl's argv (so not in `ps`), in the driver log, or in the saved
#   payloads. `set -x` is disabled at the top and is re-disabled around the one
#   function that touches the key.

set -euo pipefail
set +x

# --------------------------------------------------------------------------
# Env contract
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# The one required input.
: "${CAAS_API_KEY:=}"

# API surface. DEV is VPN-gated; the guard below says so in as many words when
# Cloudflare answers instead of the API.
: "${CAAS_BASE:=https://api-dev.io.solutions/enterprise/v1/io-cloud/caas}"

# Money. BUDGET_FRACTION of BUDGET_USD may be spent on the initial hire; the
# remainder is deliberately left unspent so a run that needs another hour can
# be extended without a second approval round.
: "${BUDGET_USD:=10}"
: "${BUDGET_FRACTION:=0.6}"
: "${INITIAL_HOURS:=2}"

# What to run.
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${CASES:=online_only,offline_only,naive,technique_a,technique_b}"
: "${MINUTES:=6}"
: "${ONLINE_RPS:=20}"
: "${MAX_MODEL_LEN:=8192}"
: "${WARMUP_S:=45}"
: "${BATCH_CONCURRENCY:=512}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${RUN_CANARY:=1}"

# What to run it on. PREFER is a case-insensitive regex over hardware_name.
: "${PREFER:=a6000|a100|4090|l40}"
: "${MIN_VRAM_GB:=24}"
: "${LOCATION_IDS:=US}"
: "${IMAGE_URL:=vllm/vllm-openai:v0.26.0}"

# What code to run. Default: whatever this checkout is at, which is the only
# way the container and the operator can be talking about the same experiment.
: "${TIDAL_REPO_URL:=https://github.com/rajagurunath/tidal}"
if [ -z "${TIDAL_SHA:-}" ]; then
  TIDAL_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
fi

# Bookkeeping.
: "${EVIDENCE_DIR:=./evidence/gpu-$(date +%Y%m%d-%H%M)}"
: "${RESOURCE_PRIVATE_NAME:=tidal-eval-$(date -u +%Y%m%d%H%M%S)}"
: "${CREATED_BY:=hire_and_run}"

# Optional secrets forwarded to the container (never written to evidence).
: "${HF_TOKEN:=}"
: "${TIDAL_API_KEY:=}"

# Timing. Overridden wholesale by the mock test.
: "${POLL_BOOT_S:=30}"
: "${POLL_STATUS_S:=60}"
: "${BOOT_TIMEOUT_S:=1200}"
: "${HTTP_TIMEOUT_S:=60}"
: "${EVAL_TIMEOUT_S:=0}"     # 0 = derive from INITIAL_HOURS
: "${FETCH_RETRIES:=5}"
: "${INSTALL_RETRIES:=3}"

# Safety valves.
: "${KEEP:=0}"               # 1 = do not DELETE on exit (you are then paying)
: "${DRY_RUN:=0}"            # 1 = select + estimate, then stop before POST /deploy
: "${HF_TRANSFER:=0}"        # stock vllm images do not ship hf_transfer

DEPLOY_ID=""
PUBLIC_URL=""
CURLRC=""
DRIVER_LOG=""
FINAL_PHASE="unknown"
DESTROYED=0

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log() {
  local line
  line="$(ts) $*"
  printf '%s\n' "$line" >&2
  if [ -n "$DRIVER_LOG" ]; then printf '%s\n' "$line" >>"$DRIVER_LOG"; fi
}

die() {
  local rc="$1"; shift
  log "FATAL($rc): $*"
  exit "$rc"
}

# --------------------------------------------------------------------------
# Teardown. Runs from a trap so a crash, a Ctrl-C and a clean finish all take
# the same path — a leaked deployment costs real money and nothing here is
# refundable.
# --------------------------------------------------------------------------
# shellcheck disable=SC2317,SC2329  # invoked from traps
destroy_deployment() {
  [ -n "$DEPLOY_ID" ] || return 0
  [ "$DESTROYED" = "0" ] || return 0
  DESTROYED=1
  local code out
  out="${EVIDENCE_DIR}/delete-response.json"
  log "destroying deployment $DEPLOY_ID"
  code="$(api DELETE "/deployment/${DEPLOY_ID}" "$out")"
  if [ "$code" = "000" ] || [ "$code" -ge 400 ] 2>/dev/null; then
    log "WARNING: DELETE returned HTTP $code — verify by hand that $DEPLOY_ID is gone"
  else
    log "deployment $DEPLOY_ID destroyed (HTTP $code)"
  fi
}

# shellcheck disable=SC2317,SC2329  # invoked from traps
print_extend_hint() {
  [ -n "$DEPLOY_ID" ] || return 0
  cat >&2 <<EOF

--------------------------------------------------------------------------
Extension is NOT automatic. If you want more time on THIS deployment, run
this *before* the teardown above (extend is additive, and it costs money):

  curl -sS -X POST "${CAAS_BASE}/deployment/${DEPLOY_ID}/extend" \\
    -H "Content-Type: application/json" \\
    -H "x-api-key: \$CAAS_API_KEY" \\
    -d '{"duration_hours":1}'

--------------------------------------------------------------------------
EOF
}

# shellcheck disable=SC2317,SC2329  # invoked from traps
on_exit() {
  local rc=$?
  trap - INT TERM EXIT
  print_extend_hint
  if [ -n "$DEPLOY_ID" ]; then
    if [ "$KEEP" = "1" ]; then
      log "KEEP=1 — leaving deployment $DEPLOY_ID RUNNING. You are being charged; DELETE it yourself."
    else
      destroy_deployment || true
    fi
  fi
  if [ -n "$CURLRC" ]; then rm -f "$CURLRC"; fi
  log "exit rc=$rc phase=$FINAL_PHASE evidence=$EVIDENCE_DIR"
  exit "$rc"
}

# shellcheck disable=SC2317,SC2329  # invoked from traps
on_signal() {
  log "signal $1 received — tearing down"
  exit 130
}

# --------------------------------------------------------------------------
# HTTP. One helper, one guard, and the key lives only in $CURLRC.
# --------------------------------------------------------------------------
write_curlrc() {
  { set +x; } 2>/dev/null
  local old_umask
  old_umask="$(umask)"
  umask 077
  CURLRC="$(mktemp "${TMPDIR:-/tmp}/hire-and-run-curlrc.XXXXXX")"
  # printf is a bash builtin: the key never becomes an argv of a real process.
  printf 'header = "x-api-key: %s"\n' "$CAAS_API_KEY" >"$CURLRC"
  umask "$old_umask"
}

# api METHOD PATH OUTFILE [extra curl args...] -> echoes the HTTP status code
api() {
  local method="$1" path="$2" out="$3"
  shift 3
  local code
  code="$(curl -sS -X "$method" \
            --config "$CURLRC" \
            --max-time "$HTTP_TIMEOUT_S" \
            -H 'Accept: application/json' \
            -o "$out" -w '%{http_code}' \
            "$@" \
            "${CAAS_BASE}${path}" 2>>"${DRIVER_LOG:-/dev/null}")" || code="000"
  printf '%s' "$code"
}

looks_like_html() {
  head -c 1024 "$1" 2>/dev/null \
    | tr '[:upper:]' '[:lower:]' \
    | grep -q '<!doctype html\|<html\|cloudflare\|attention required\|just a moment'
}

# guard CODE OUTFILE WHAT EXITCODE
guard() {
  local code="$1" out="$2" what="$3" rc="$4"
  if [ "$code" = "000" ]; then
    die "$rc" "$what: no HTTP response from ${CAAS_BASE}. Is the io.net VPN up?"
  fi
  if looks_like_html "$out"; then
    log "response head: $(head -c 200 "$out" | tr -d '\n')"
    die "$rc" "$what: got an HTML page (HTTP $code), not JSON. The DEV API is VPN-gated and Cloudflare is answering instead. Connect the io.net VPN and retry."
  fi
  if [ "$code" = "401" ] || [ "$code" = "403" ]; then
    die "$rc" "$what: HTTP $code. Either CAAS_API_KEY is wrong/unscoped for ${CAAS_BASE}, or you are outside the VPN. Body: $(head -c 400 "$out" | tr -d '\n')"
  fi
  if [ "$code" -ge 400 ] 2>/dev/null; then
    die "$rc" "$what: HTTP $code. Body: $(head -c 800 "$out" | tr -d '\n')"
  fi
  if ! jq -e . "$out" >/dev/null 2>&1; then
    die "$rc" "$what: HTTP $code but the body is not JSON. Body: $(head -c 400 "$out" | tr -d '\n')"
  fi
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
for tool in curl jq awk tar; do
  command -v "$tool" >/dev/null 2>&1 || { printf 'missing required tool: %s\n' "$tool" >&2; exit 2; }
done

[ -n "$CAAS_API_KEY" ] || {
  printf 'CAAS_API_KEY is not set. Export it (it is never printed by this script) and retry.\n' >&2
  exit 2
}
[ -n "$TIDAL_SHA" ] || {
  printf 'TIDAL_SHA is empty and %s is not a git checkout. Set TIDAL_SHA=<sha> explicitly.\n' "$REPO_ROOT" >&2
  exit 2
}

mkdir -p "$EVIDENCE_DIR"
EVIDENCE_DIR="$(cd -- "$EVIDENCE_DIR" && pwd)"
DRIVER_LOG="${EVIDENCE_DIR}/driver.log"
: >>"$DRIVER_LOG"

trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

write_curlrc

if [ "$EVAL_TIMEOUT_S" = "0" ]; then
  EVAL_TIMEOUT_S="$(awk -v h="$INITIAL_HOURS" 'BEGIN{printf "%d", h*3600 + 900}')"
fi
SPEND_CAP="$(awk -v b="$BUDGET_USD" -v f="$BUDGET_FRACTION" 'BEGIN{printf "%.4f", b*f}')"

log "hire_and_run start"
log "  base=${CAAS_BASE}"
log "  budget=\$${BUDGET_USD} cap=\$${SPEND_CAP} (${BUDGET_FRACTION}) initial_hours=${INITIAL_HOURS}"
log "  model=${MODEL} cases=${CASES} minutes=${MINUTES} rps=${ONLINE_RPS} max_model_len=${MAX_MODEL_LEN}"
log "  tidal_sha=${TIDAL_SHA} image=${IMAGE_URL}"
log "  evidence=${EVIDENCE_DIR}"

# --------------------------------------------------------------------------
# Step 1 — Shop the catalogue
# --------------------------------------------------------------------------
HW_RAW="${EVIDENCE_DIR}/hardware.json"
log "GET /hardware"
HW_CODE="$(api GET "/hardware" "$HW_RAW")"
guard "$HW_CODE" "$HW_RAW" "GET /hardware" 2

# The catalogue's envelope has moved before (bare array, {data:[...]}, nested).
# Rather than pin a shape, collect every object anywhere in the document that
# carries a hardware_id — that field is the row identity by definition.
# shellcheck disable=SC2016  # jq program: $prefer/$cap/$hours are jq args, not shell vars
HW_SELECT_JQ='
def n($v): ($v | tostring | tonumber?) // null;
[ .. | objects | select(has("hardware_id")) ]
| map({
    hardware_id_raw: .hardware_id,
    id:   (.hardware_id | tostring),
    name: ((.hardware_name // .name // "") | tostring),
    available: ((n(.available) // n(.available_count) // n(.availability) // 0)),
    price: ((n(.price) // n(.price_per_hour) // n(.hourly_price) // n(.price_per_hour_usd) // 0)),
    maxg:  ((n(.max_gpus_per_container) // 1)),
    location: ((.location // .location_id // "") | tostring),
    vram: (
      ( [ to_entries[]
          | select(.key | test("vram|gpu_memory|memory_gb|gpu_ram"; "i"))
          | n(.value) | select(. != null) ] | first )
      // ( ((.hardware_name // .name // "") | tostring
           | capture("(?<g>[0-9]+)[ ]?[Gg][Bb]")? | .g | tonumber?) )
      // null
    ),
    rank: (
      ((.hardware_name // .name // "") | tostring | ascii_downcase) as $n
      | if ($n | test("a6000")) then 0
        elif ($n | test("a100")) then 1
        elif ($n | test("h100|h200")) then 2
        else 3 end
    )
  })
| map(select(.available > 0))
| map(select(.name | test($prefer; "i")))
| map(select((.price * $hours) <= $cap))
| . as $elig
| ([ $elig[] | select(.vram != null and .vram >= $minvram) ] | sort_by(.price))   as $big
| ([ $elig[] | select(.vram == null) ] | sort_by([.rank, .price]))                as $unknown
| {
    eligible: ($elig | length),
    with_vram: ($elig | map(select(.vram != null)) | length),
    too_small: ([ $elig[] | select(.vram != null and .vram < $minvram) ] | length),
    choice: (if ($big | length) > 0 then $big[0]
             elif ($unknown | length) > 0 then $unknown[0]
             else null end)
  }
'

HW_PICK="${EVIDENCE_DIR}/hardware-choice.json"
jq --arg prefer "$PREFER" \
   --argjson hours "$INITIAL_HOURS" \
   --argjson cap "$SPEND_CAP" \
   --argjson minvram "$MIN_VRAM_GB" \
   "$HW_SELECT_JQ" "$HW_RAW" >"$HW_PICK"

ELIGIBLE="$(jq -r '.eligible' "$HW_PICK")"
TOO_SMALL="$(jq -r '.too_small' "$HW_PICK")"
if [ "$(jq -r '.choice == null' "$HW_PICK")" = "true" ]; then
  log "catalogue rows: $(jq '[ .. | objects | select(has("hardware_id"))] | length' "$HW_RAW")"
  log "eligible after availability+PREFER+budget filters: ${ELIGIBLE} (of which ${TOO_SMALL} were under ${MIN_VRAM_GB}GB VRAM)"
  die 2 "no hardware matches PREFER='${PREFER}' with available>0 and ${INITIAL_HOURS}h under \$${SPEND_CAP}. Widen PREFER, raise BUDGET_USD, or lower INITIAL_HOURS. Catalogue saved to ${HW_RAW}."
fi

HW_ID_JSON="$(jq -c '.choice.hardware_id_raw' "$HW_PICK")"
HW_ID="$(jq -r '.choice.id' "$HW_PICK")"
HW_NAME="$(jq -r '.choice.name' "$HW_PICK")"
HW_PRICE="$(jq -r '.choice.price' "$HW_PICK")"
HW_VRAM="$(jq -r '.choice.vram // "unknown"' "$HW_PICK")"
HW_LOC="$(jq -r '.choice.location' "$HW_PICK")"
HW_COST="$(awk -v p="$HW_PRICE" -v h="$INITIAL_HOURS" 'BEGIN{printf "%.2f", p*h}')"

log "chose hardware_id=${HW_ID} name='${HW_NAME}' price=\$${HW_PRICE}/hr vram=${HW_VRAM}GB location='${HW_LOC}'"
log "  ${INITIAL_HOURS}h at that rate = \$${HW_COST} (cap \$${SPEND_CAP}, ${ELIGIBLE} eligible rows)"

# --------------------------------------------------------------------------
# Step 2 — Price it before hiring it
# --------------------------------------------------------------------------
PRICE_RAW="${EVIDENCE_DIR}/price.json"
log "GET /price"
PRICE_CODE="$(api GET "/price" "$PRICE_RAW" \
  -G \
  --data-urlencode "location_ids=${LOCATION_IDS}" \
  --data-urlencode "hardware_id=${HW_ID}" \
  --data-urlencode "duration_hours=${INITIAL_HOURS}" \
  --data-urlencode "gpus_per_container=1" \
  --data-urlencode "replica_count=1")"
guard "$PRICE_CODE" "$PRICE_RAW" "GET /price" 2

# shellcheck disable=SC2016  # jq program: $re is a jq function parameter
PRICE_JQ='
def pick($re): [ .. | objects | to_entries[]
                 | select(.key | test($re; "i"))
                 | (.value | tostring | tonumber?) | select(. != null) ] | max;
(pick("^total.*(price|cost|amount)$") // pick("estimat") // pick("^(total|price|cost|amount)")) // null
'
ESTIMATE="$(jq -r "$PRICE_JQ" "$PRICE_RAW")"
if [ "$ESTIMATE" = "null" ] || [ -z "$ESTIMATE" ]; then
  ESTIMATE="$HW_COST"
  log "WARNING: could not read an estimate out of /price (saved to ${PRICE_RAW}); falling back to catalogue price x hours = \$${ESTIMATE}"
else
  log "price estimate for ${INITIAL_HOURS}h = \$${ESTIMATE}"
fi

if [ "$(awk -v e="$ESTIMATE" -v c="$SPEND_CAP" 'BEGIN{print (e>c) ? 1 : 0}')" = "1" ]; then
  die 5 "budget abort: estimate \$${ESTIMATE} exceeds the spend cap \$${SPEND_CAP} (BUDGET_USD=${BUDGET_USD} x BUDGET_FRACTION=${BUDGET_FRACTION}). Nothing was hired."
fi

# --------------------------------------------------------------------------
# Step 3 — The container's whole life, as one bash -lc string
# --------------------------------------------------------------------------
# /deploy SILENTLY DROPS container_config.args. Everything — the install, the
# checkout, the exports, the supervisor — therefore has to live in the
# entrypoint string. Assembled from an array and joined with ' && ' so each
# step is independently readable and any failure stops the chain. No single
# quotes anywhere in it, so the whole thing stays safe to wrap in '...' when
# you paste it into a shell by hand.
CMD_PARTS=()
CMD_PARTS+=("set -euo pipefail")
CMD_PARTS+=("mkdir -p /results")
CMD_PARTS+=("export MODE=selfdrive CAAS=1 REPO_DIR=/opt/tidal EVAL_SCRIPT=/opt/tidal/deploy/run_gpu_eval.sh RESULTS_DIR=/results TIDAL_EVAL_VLLM_BIN=vllm PORT=8000 PYTHONPATH=/opt/tidal/src PYTHONUNBUFFERED=1")
CMD_PARTS+=("export MODEL=${MODEL} TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN} ONLINE_RPS=${ONLINE_RPS} MINUTES=${MINUTES} CASES=${CASES} WARMUP_S=${WARMUP_S} BATCH_CONCURRENCY=${BATCH_CONCURRENCY} RUN_CANARY=${RUN_CANARY}")
# git is not guaranteed in a stock vllm image, and both the pip install and the
# asset checkout need it.
CMD_PARTS+=("( command -v git >/dev/null 2>&1 || ( apt-get update -qq && apt-get install -y -qq --no-install-recommends git ) )")
CMD_PARTS+=("( for i in \$(seq 1 ${INSTALL_RETRIES}); do pip install -q --no-cache-dir git+${TIDAL_REPO_URL}.git@${TIDAL_SHA} && exit 0; sleep \$(( i * 15 )); done; exit 1 )")
# The eval extras (matplotlib, openai) are a separate line rather than a
# [eval] marker so the install URL above stays quote-free.
CMD_PARTS+=("( for i in \$(seq 1 ${INSTALL_RETRIES}); do pip install -q --no-cache-dir matplotlib openai && exit 0; sleep \$(( i * 15 )); done; exit 1 )")
# The pip install gives us the importable package; the clone gives us the
# script assets (run_gpu_eval.sh and friends) that live outside the wheel.
CMD_PARTS+=("( for i in \$(seq 1 ${INSTALL_RETRIES}); do git clone --depth 1 ${TIDAL_REPO_URL} /opt/tidal && exit 0; rm -rf /opt/tidal; sleep \$(( i * 15 )); done; exit 1 )")
CMD_PARTS+=("cd /opt/tidal")
# --depth 1 only fetches the default branch tip, so an arbitrary SHA needs a
# targeted fetch before it can be checked out.
CMD_PARTS+=("( git checkout ${TIDAL_SHA} || ( git fetch --depth 1 origin ${TIDAL_SHA} && git checkout FETCH_HEAD ) )")
CMD_PARTS+=("chmod +x /opt/tidal/deploy/run_gpu_eval.sh")
CMD_PARTS+=("exec python3 -m tidal.eval.selfdrive")

CMD=""
for part in "${CMD_PARTS[@]}"; do
  if [ -z "$CMD" ]; then CMD="$part"; else CMD="${CMD} && ${part}"; fi
done
printf '%s\n' "$CMD" >"${EVIDENCE_DIR}/entrypoint-cmd.txt"

# --------------------------------------------------------------------------
# Step 4 — POST /deploy
# --------------------------------------------------------------------------
PAYLOAD="$(mktemp "${TMPDIR:-/tmp}/hire-and-run-payload.XXXXXX")"
jq -n \
  --arg name "$RESOURCE_PRIVATE_NAME" \
  --argjson hours "$INITIAL_HOURS" \
  --argjson hw "$HW_ID_JSON" \
  --arg locs "$LOCATION_IDS" \
  --arg cmd "$CMD" \
  --arg image "$IMAGE_URL" \
  --arg created_by "$CREATED_BY" \
  --arg model "$MODEL" \
  --arg tp "$TENSOR_PARALLEL_SIZE" \
  --arg mml "$MAX_MODEL_LEN" \
  --arg rps "$ONLINE_RPS" \
  --arg minutes "$MINUTES" \
  --arg cases "$CASES" \
  --arg warmup "$WARMUP_S" \
  --arg batch "$BATCH_CONCURRENCY" \
  --arg canary "$RUN_CANARY" \
  --arg hftransfer "$HF_TRANSFER" \
  --arg hf_token "$HF_TOKEN" \
  --arg tidal_key "$TIDAL_API_KEY" \
  '{
     resource_private_name: $name,
     duration_hours: $hours,
     gpus_per_container: 1,
     hardware_id: $hw,
     location_ids: ($locs | split(",") | map(select(length > 0))),
     container_config: {
       replica_count: 1,
       traffic_port: 8000,
       entrypoint: ["bash", "-lc", $cmd],
       env_variables: ({
         created_by: $created_by,
         MODE: "selfdrive",
         CAAS: "1",
         PORT: "8000",
         RESULTS_DIR: "/results",
         REPO_DIR: "/opt/tidal",
         EVAL_SCRIPT: "/opt/tidal/deploy/run_gpu_eval.sh",
         TIDAL_EVAL_VLLM_BIN: "vllm",
         MODEL: $model,
         TENSOR_PARALLEL_SIZE: $tp,
         MAX_MODEL_LEN: $mml,
         ONLINE_RPS: $rps,
         MINUTES: $minutes,
         CASES: $cases,
         WARMUP_S: $warmup,
         BATCH_CONCURRENCY: $batch,
         RUN_CANARY: $canary,
         VLLM_LOG_STATS_INTERVAL: "1",
         VLLM_RPC_TIMEOUT: "600000",
         VLLM_ENGINE_READY_TIMEOUT_S: "3600",
         HF_HUB_ENABLE_HF_TRANSFER: $hftransfer,
         internal_logging_enabled: "true"
       }),
       secret_env_variables: (
         {}
         + (if $hf_token  != "" then {HF_TOKEN: $hf_token, HUGGING_FACE_HUB_TOKEN: $hf_token} else {} end)
         + (if $tidal_key != "" then {TIDAL_API_KEY: $tidal_key} else {} end)
       )
     },
     registry_config: { image_url: $image }
   }' >"$PAYLOAD"

# The evidence copy has every secret value replaced, so the directory is safe
# to attach to a writeup.
jq '.container_config.secret_env_variables |= with_entries(.value = "REDACTED")' \
  "$PAYLOAD" >"${EVIDENCE_DIR}/deploy-request.redacted.json"

if [ "$DRY_RUN" = "1" ]; then
  rm -f "$PAYLOAD"
  log "DRY_RUN=1 — selection and estimate done, nothing hired. Payload: ${EVIDENCE_DIR}/deploy-request.redacted.json"
  FINAL_PHASE="dry-run"
  exit 0
fi

DEPLOY_RESP="${EVIDENCE_DIR}/deploy-response.json"
log "POST /deploy (${INITIAL_HOURS}h on ${HW_NAME}, charged up front, no refund on early destroy)"
DEPLOY_CODE="$(api POST "/deploy" "$DEPLOY_RESP" \
  -H 'Content-Type: application/json' \
  --data-binary "@${PAYLOAD}")"
rm -f "$PAYLOAD"
guard "$DEPLOY_CODE" "$DEPLOY_RESP" "POST /deploy" 2

# deployment_id is the documented name; cluster_id and a bare id are the shapes
# the endpoint has answered with before. Probed in that order of confidence so a
# nested, unrelated `id` can never win.
DEPLOY_ID="$(jq -r '
  def hunt($key): [ .. | objects | to_entries[]
                    | select(.key == $key)
                    | .value | select(type == "string" or type == "number")
                    | tostring | select(length > 0) ] | first;
  (hunt("deployment_id") // hunt("cluster_id") // hunt("id")) // ""' "$DEPLOY_RESP")"
if [ -z "$DEPLOY_ID" ]; then
  log "deploy response body: $(head -c 1200 "$DEPLOY_RESP" | tr -d '\n')"
  die 2 "POST /deploy returned HTTP ${DEPLOY_CODE} but no deployment id could be found in the body (saved to ${DEPLOY_RESP})"
fi
log "deployment_id=${DEPLOY_ID} — teardown is now armed (trap on EXIT/INT/TERM)"

# --------------------------------------------------------------------------
# Step 5 — Wait for a running container with a public_url
# --------------------------------------------------------------------------
CONTAINERS_LOG="${EVIDENCE_DIR}/containers-log.jsonl"
BOOT_START="$(date +%s)"
LAST_WORKER_STATUS="none"
CONT_TMP="${EVIDENCE_DIR}/.containers.json"

while :; do
  ELAPSED=$(( $(date +%s) - BOOT_START ))
  if [ "$ELAPSED" -ge "$BOOT_TIMEOUT_S" ]; then
    log "boot timeout after ${ELAPSED}s; last worker status='${LAST_WORKER_STATUS}'"
    FINAL_PHASE="boot-timeout"
    die 3 "no running container with a public_url after ${BOOT_TIMEOUT_S}s (last status: ${LAST_WORKER_STATUS}). Deployment will be destroyed."
  fi

  CONT_CODE="$(api GET "/deployment/${DEPLOY_ID}/containers" "$CONT_TMP")"
  if [ "$CONT_CODE" != "000" ] && ! looks_like_html "$CONT_TMP" && jq -e . "$CONT_TMP" >/dev/null 2>&1; then
    jq -c --arg t "$(ts)" '{t: $t, containers: .}' "$CONT_TMP" >>"$CONTAINERS_LOG"
    PUBLIC_URL="$(jq -r '[ .. | objects | select(has("public_url")) | .public_url
                           | select(type == "string" and length > 0) ] | first // ""' "$CONT_TMP")"
    LAST_WORKER_STATUS="$(jq -r '[ .. | objects | select(has("public_url")) | (.status // "")
                                   | select(type == "string" and length > 0) ] | first // "unknown"' "$CONT_TMP")"
    if [ -n "$PUBLIC_URL" ] && printf '%s' "$LAST_WORKER_STATUS" | grep -qi 'running\|ready\|healthy'; then
      PUBLIC_URL="${PUBLIC_URL%/}"
      log "container running after ${ELAPSED}s: public_url=${PUBLIC_URL}"
      break
    fi
  else
    LAST_WORKER_STATUS="http_${CONT_CODE}"
  fi
  log "booting… t=${ELAPSED}s status=${LAST_WORKER_STATUS} public_url=${PUBLIC_URL:-none}"
  sleep "$POLL_BOOT_S"
done

# --------------------------------------------------------------------------
# Step 6 — Watch the eval through the supervisor's own /status
# --------------------------------------------------------------------------
STATUS_LOG="${EVIDENCE_DIR}/status-log.jsonl"
STATUS_TMP="${EVIDENCE_DIR}/.status.json"
EVAL_START="$(date +%s)"
PHASE="unknown"
STATUS_MISSES=0

log "polling ${PUBLIC_URL}/status every ${POLL_STATUS_S}s (eval timeout ${EVAL_TIMEOUT_S}s)"
while :; do
  ELAPSED=$(( $(date +%s) - EVAL_START ))
  if [ "$ELAPSED" -ge "$EVAL_TIMEOUT_S" ]; then
    log "eval timeout after ${ELAPSED}s at phase=${PHASE}"
    PHASE="timeout"
    break
  fi

  if curl -sS --max-time "$HTTP_TIMEOUT_S" -o "$STATUS_TMP" "${PUBLIC_URL}/status" 2>>"$DRIVER_LOG" \
     && jq -e . "$STATUS_TMP" >/dev/null 2>&1; then
    STATUS_MISSES=0
    jq -c --arg t "$(ts)" --argjson elapsed "$ELAPSED" '. + {driver_t: $t, driver_elapsed_s: $elapsed}' \
      "$STATUS_TMP" >>"$STATUS_LOG"
    PHASE="$(jq -r '.phase // "unknown"' "$STATUS_TMP")"
    DONE_N="$(jq -r '.conditions.completed // 0' "$STATUS_TMP")"
    TOTAL_N="$(jq -r '.conditions.total // 0' "$STATUS_TMP")"
    ERR="$(jq -r '.error // ""' "$STATUS_TMP")"
    log "t=${ELAPSED}s phase=${PHASE} conditions=${DONE_N}/${TOTAL_N}${ERR:+ error=${ERR}}"
    case "$PHASE" in
      done|failed|aborted) break ;;
    esac
  else
    STATUS_MISSES=$(( STATUS_MISSES + 1 ))
    log "t=${ELAPSED}s /status unreachable (miss ${STATUS_MISSES})"
  fi
  sleep "$POLL_STATUS_S"
done

FINAL_PHASE="$PHASE"

# --------------------------------------------------------------------------
# Step 7 — Egress. This is the only window; the container dies with the rental.
# --------------------------------------------------------------------------
TARBALL="${EVIDENCE_DIR}/results.tar.gz"
FETCH_OK=0
i=1
while [ "$i" -le "$FETCH_RETRIES" ]; do
  if curl -fsS --max-time 900 -o "$TARBALL" "${PUBLIC_URL}/results.tar.gz" 2>>"$DRIVER_LOG" \
     && tar -tzf "$TARBALL" >/dev/null 2>&1; then
    log "fetched results.tar.gz ($(wc -c <"$TARBALL" | tr -d ' ') bytes) -> ${TARBALL}"
    FETCH_OK=1
    break
  fi
  log "results.tar.gz fetch attempt ${i}/${FETCH_RETRIES} failed"
  rm -f "$TARBALL"
  i=$(( i + 1 ))
  [ "$i" -le "$FETCH_RETRIES" ] && sleep 15
done
[ "$FETCH_OK" = "1" ] || log "WARNING: could not retrieve results.tar.gz — see ${EVIDENCE_DIR}/full-log.txt"

if curl -fsS --max-time "$HTTP_TIMEOUT_S" -o "${EVIDENCE_DIR}/full-log.txt" "${PUBLIC_URL}/log?tail=5000" 2>>"$DRIVER_LOG"; then
  log "fetched run log -> ${EVIDENCE_DIR}/full-log.txt"
else
  log "WARNING: could not retrieve ${PUBLIC_URL}/log"
fi

curl -fsS --max-time "$HTTP_TIMEOUT_S" -o "${EVIDENCE_DIR}/status-final.json" "${PUBLIC_URL}/status" 2>>"$DRIVER_LOG" || true
rm -f "$STATUS_TMP" "$CONT_TMP"

# --------------------------------------------------------------------------
# Verdict. Teardown happens in the EXIT trap either way.
# --------------------------------------------------------------------------
case "$PHASE" in
  done)
    log "eval finished: phase=done. Evidence in ${EVIDENCE_DIR}"
    exit 0
    ;;
  *)
    log "eval did not reach done: phase=${PHASE}"
    exit 4
    ;;
esac
