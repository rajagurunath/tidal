# Tidal GPU deployment kit (io.net CaaS)

Build one image, deploy it to a GPU box on io.net CaaS, and get the full A/B
evaluation back over HTTP. The container drives the entire run by itself and
hands the results out through its one published port.

> **Unofficial.** This is a test deployment kit, not a production fleet
> procedure. The io.net CaaS deploy endpoint is approval-gated for good reason —
> every step that costs money or touches a shared box is called out below and
> should be run by hand, with eyes on it.

## Why the container is self-driving

CaaS gives you a container and takes away every way you would normally babysit
one:

| You would normally… | On CaaS |
|---|---|
| `ssh` in and run the eval under tmux | no SSH |
| `docker exec` to check on it | no exec |
| mount a volume and copy results off | no volume retrieval |
| leave it running and come back | the reservation ends at `duration_hours` |
| reach several ports | exactly one `public_url`, for `traffic_port` |

So the image does all of it from `docker run` with no operator present:
`engine-entrypoint.sh` (mode `selfdrive`) execs `python -m tidal.eval.selfdrive`,
which launches `run_gpu_eval.sh` as a child process group and serves progress
and results on `$PORT` for the life of the reservation — **including after the
eval finishes or fails**, because that window is the only chance to download
anything.

Nothing else needs to be running in the container. The eval harness owns its
engine: for each condition it launches a fresh `vllm serve` (stock, or with
`--scheduler-cls`), waits for `/health`, drives the load, and kills the process
group. There is deliberately **no** long-lived engine alongside it — a second
engine on the same GPU makes every latency number meaningless.

## Files

| File | What it is |
|---|---|
| `Dockerfile.engine` | vLLM GPU image + Tidal. `BUILD_MODE=fast` (default) or `pinned`. |
| `engine-entrypoint.sh` | `MODE=selfdrive` → the supervisor; `MODE=serve` → a `vllm serve` command line. Defaults to selfdrive when `CAAS=1`. |
| `caas-payload.template.json` | io.net CaaS deploy payload for the self-driving eval, `${PLACEHOLDER}` form. |
| `run_gpu_eval.sh` | The evaluation itself: canary → probe → 5-condition matrix → diurnal → deadline pair → figures → tarball. Emits `PROGRESS` lines for the supervisor. |
| `../src/tidal/eval/selfdrive.py` | The supervisor: runs the script, serves `/status`, `/log`, `/results*`, `/abort`. |
| `docker-compose.gpu.yml` | The *serving* topology (engine + gateway) for a dev box. Not used by the eval. |
| `Dockerfile.gateway` | Slim Python image running `tidal serve`. No CUDA. |
| `Dockerfile.*.dockerignore` | BuildKit-scoped ignore rules, so the kit does not need a repo-root `.dockerignore`. |

---

## Step 0 — Pick the vLLM tag, and read the pin caveat

`Dockerfile.engine` defaults to `--build-arg VLLM_IMAGE=vllm/vllm-openai:v0.26.0`.
**Verify that tag exists before you build** — the tag list moves weekly and this
default was chosen offline:

```bash
curl -s 'https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=25' \
  | jq -r '.results[].name'
```

### The version-pin caveat

`docs/plans/2026-08-06-plan-B-engine.md` verified TidalScheduler's interface
points against vLLM at commit **`e6d67fdd`** — `schedule()`,
`SchedulingPolicy.PRIORITY` preemption, `_preempt_request()`,
`update_from_output()`, `SchedulerConfig.scheduler_cls`. Upstream moves fast and
none of those are public API.

The fast path does *not* pin to that commit, and that is a deliberate trade:
`TidalScheduler` probes for each hook at startup and degrades mechanism by
mechanism instead of crashing, logging exactly what it found. So a drifted vLLM
shows up as a **downgraded capability line, not a failed boot** — which is far
worse than a crash if nobody looks.

On CaaS nobody *can* look at the container's logs mid-run, which is why the
compat canary is now the first thing the eval does (see below) and why its
verdict is on `/status` and in `MANIFEST.txt`. Two ways to pin if the canary
comes back red:

1. **Preferred.** Build vLLM's own `docker/Dockerfile` at `e6d67fdd`, tag it,
   and feed it to the fast path:
   ```bash
   git clone https://github.com/vllm-project/vllm && cd vllm
   git checkout e6d67fdd
   docker build -f docker/Dockerfile -t vllm-pinned:e6d67fdd --target vllm-openai .
   # then, from the tidal repo root:
   docker build -f deploy/Dockerfile.engine \
     --build-arg VLLM_IMAGE=vllm-pinned:e6d67fdd -t tidal-engine:pinned .
   ```
2. **Convenience, unverified.** `--build-arg BUILD_MODE=pinned` compiles vLLM
   from source inside `Dockerfile.engine`. Budget **30–60 minutes** on a
   many-core builder, several GB of ccache, and expect to adjust
   `TORCH_INDEX_URL` / `VLLM_BUILD_BASE` if the pinned commit wants a different
   torch. Narrow `TORCH_CUDA_ARCH_LIST` (`9.0` for H100/H200, `10.0` for B200) —
   every extra arch is another full kernel compile.

---

## Step 1 — Build and push

**On a Mac (arm64), you must cross-build or build on a Linux box.** The vLLM
base images are `linux/amd64` only.

```bash
# From the repo root. VERIFY_IMPORTS=0 because importing torch under QEMU
# emulation is pathologically slow.
docker build --platform linux/amd64 \
  -f deploy/Dockerfile.engine \
  --build-arg VERIFY_IMPORTS=0 \
  -t <registry>/tidal-engine:fast .

docker push <registry>/tidal-engine:fast
```

CaaS needs a registry it can pull from. A public tag is simplest; otherwise put
the credentials in the payload's `registry_config`.

`INSTALL_EVAL_EXTRAS=1` (the default) is required for the self-driving image:
the supervisor needs `fastapi`/`uvicorn` (both already in the vLLM base) and the
eval needs `matplotlib`, `openai`, `typer` and `pytest` — `pytest` because the
compat canary runs *inside* the container now.

---

## Step 2 — Pick a node and read its identity

Full procedure: the **`caas-model-deploy`** skill, Step 2. This is the one part
of the flow that still wants a shell, and it is on the *provisioning* side, not
the eval side — you are choosing where to spend the money:

```bash
ssh "$SSH_TARGET" "cat /etc/ionet_device_id"                                    # device_id
ssh "$SSH_TARGET" "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv"
ssh "$SSH_TARGET" "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'"
```

Two things to establish before going further:

- **The GPUs are actually free.** Cross-reference `nvidia-smi` against running
  containers. If something LLM-shaped is on the cards, the skill's Step 3 covers
  the teardown — and note that a Nomad-managed CaaS container will *not* stay
  down after `docker stop`; it needs the CaaS `DELETE` path.
- **Idle 1 GB hugepages.** Some provider images reserve terabytes of hugepages
  that nothing uses. The skill documents the check and the safe runtime release.

Weights: pre-staging `MODEL` into the fleet's shared HF cache
(`/ephemeral/hf-cache/<deployer-uuid>/`) before deploying turns a ~30 min cold
start into ~5 min, and the eval's wall-clock budget is dominated by exactly that
kind of thing. The skill's Step 4.5 has the exact `hf download` container.

`node_pool_id` and `hardware_id` come from **Prod RDS (Preset `database_id=4`)**
— do not guess them. The `caas-model-deploy` skill's Step 4 has the queries; the
shape is `node_pool_devices` joined to `node_pools` filtered by `device_id` (the
column is **`pool_id`**, not `node_pool_id` — the wrong name silently returns
zero rows), and `devices` filtered by `device_id` for `hardware_id` (a *numeric
string*: keep it quoted).

---

## Step 3 — Fill the payload

`caas-payload.template.json` is `envsubst`-shaped and carries the **full eval
parameterization** — there is no second chance to set a knob once the container
is running.

```bash
export RESOURCE_PRIVATE_NAME=tidal-eval-h200-1
export DEVICE_ID=...            # /etc/ionet_device_id      (Step 2)
export HARDWARE_ID=...          # devices.hardware_id       (Step 2, quoted string)
export NODE_POOL_ID=...         # node_pool_devices.pool_id (Step 2)
export CREATED_BY=<your-handle>
export ENGINE_IMAGE=<registry>/tidal-engine:fast
export REGISTRY_USERNAME= REGISTRY_SECRET=       # empty for a public image

# The evaluation itself
export MODEL=Qwen/Qwen2.5-7B-Instruct
export TENSOR_PARALLEL_SIZE=1                    # must equal gpus_per_container
export MAX_MODEL_LEN=8192
export ONLINE_RPS=20 MINUTES=15
export BATCH_CONCURRENCY=                        # empty = the GPU preset's 512
export DIURNAL_MINUTES=20 DEADLINE_MINUTES=18 DEADLINE_TIGHTNESS=0.55
export MAX_INFLIGHT=64 CASE_TIMEOUT_S=5400

# Secrets
export HF_TOKEN=...                              # gated models only; never commit
export TIDAL_API_KEY="$(openssl rand -hex 16)"   # the only auth on POST /abort
export TIDAL_RESULTS_HF_REPO=                    # optional backup egress, see below

envsubst < deploy/caas-payload.template.json > /tmp/tidal-payload.json
jq . /tmp/tidal-payload.json    # must parse
```

Field by field:

| Field | Where it comes from |
|---|---|
| `resource_private_name` | You pick it. Fleet convention `<model-slug>-<backend>-<gpu-tier>-<idx>`; take the next free index from `resources.resource_private_name` in Prod RDS. |
| `device_id` | `/etc/ionet_device_id` on the box (Step 2). |
| `duration_hours` | **The deadline for the whole thing.** See the sizing formula below; the template ships `12`. |
| `gpus_per_container` | **Must equal `TENSOR_PARALLEL_SIZE`.** Left as a literal `1` because JSON numbers cannot hold a placeholder — edit it by hand if you change TP. |
| `hardware_id` / `node_pool_id` | Prod RDS (Step 2). |
| `container_config.env_variables` | The eval parameterization. `MODE=selfdrive` and `CAAS=1` are what make the container self-driving; `PORT` and `traffic_port` must agree. |
| `container_config.secret_env_variables` | `HF_TOKEN` (both spellings, because the hub CLI and `huggingface_hub` read different ones) and `TIDAL_API_KEY`. Kept out of `env_variables` so they are not echoed back by the status endpoint. |
| `container_config.entrypoint` | `/usr/local/bin/tidal-engine-entrypoint`. Everything else is env-driven, which is why `args` is empty. |
| `container_config.traffic_port` | `8000` — **the supervisor's port**, not vLLM's. The per-condition engines live on `EVAL_PORT` (8400) and are never exposed. |
| `registry_config.image_url` | Your pushed engine tag. |

### Sizing `duration_hours`

The reservation has to cover the eval **plus the time it takes you to download
the results afterwards**, because the container disappears with the reservation
and nothing was written to a volume.

```
eval_hours ≈ ( 20                                  # boot, weight prewarm, canary
             + 5                                   # the sizing probe
             + 4 × cases                           # ~4 min of engine start/stop per case
             + 5 × MINUTES                         # the matrix
             + 2 × DIURNAL_MINUTES
             + 2 × DEADLINE_MINUTES ) / 60
```

At the defaults (`MINUTES=15`, `DIURNAL_MINUTES=20`, `DEADLINE_MINUTES=18`, 10
cases) that is ≈ 3.6 h. Then:

- **+1 h** if the model is not already in the node's HF cache (cold download).
- **+1–2 h** of retrieval margin. A tarball with figures runs to hundreds of MB
  and you may not be at your desk when it finishes.

`duration_hours = 12` covers the defaults comfortably. Scale it if you raise
`MINUTES` or add conditions, and remember `CASE_TIMEOUT_S` (5400 s) is the
worst case *per case* — a run where several cases hang costs far more than the
formula.

---

## Step 4 — Deploy (approval-gated)

Follow the `caas-model-deploy` skill's Steps 6–7: write the curl into a script,
show the payload, get an explicit yes, *then* run it.

```bash
# Prod. Staging is https://api.staging-prod.io.solutions/engineering/v1/...
# Use whichever environment your IOCLOUD_API_KEY is scoped for.
[ -n "${IOCLOUD_API_KEY:-}" ] || { echo "IOCLOUD_API_KEY not set"; exit 1; }

curl --location 'https://api.io.solutions/engineering/v1/io-cloud/caas/deploy-to-specific-device' \
  --header 'Content-Type: application/json' \
  --header "x-api-key: $IOCLOUD_API_KEY" \
  --data @/tmp/tidal-payload.json
```

The response's `deployment_id` **is** the `caas_cluster_id`. Keep it.

`traffic_port` is not published to a host port — CaaS containers live in a Nomad
network namespace. The address you actually use comes from the containers
endpoint:

```bash
export CLUSTER_ID=<deployment_id>
export PUBLIC_URL="$(curl -s \
  "https://api.io.solutions/enterprise/v1/io-cloud/caas/deployment/${CLUSTER_ID}/containers" \
  -H "x-api-key: $IOCLOUD_API_KEY" | jq -r '.data.workers[0].public_url')"
echo "$PUBLIC_URL"
```

It can take a few minutes to appear while the image pulls. `GET $PUBLIC_URL/healthz`
answers as soon as the supervisor is up — well before the first GPU work starts,
because the supervisor binds the port *before* the eval gets anywhere.

---

## Step 5 — Watch it

```bash
watch -n 60 "curl -s $PUBLIC_URL/status | jq '{phase, canary, error, conditions, elapsed_s, engine: .engine.model}'"
```

or the same thing without `watch`:

```bash
while :; do
  curl -s "$PUBLIC_URL/status" \
    | jq -c '{t: .elapsed_s, phase, done: .conditions.completed, total: .conditions.total, error}'
  sleep 60
done
```

A typical `/status`:

```json
{
  "phase": "running:technique_a",
  "error": null,
  "canary": "ok",
  "conditions": {"completed": 4, "total": 10,
                 "done": ["probe_offline", "online_only", "offline_only", "naive"],
                 "failed": []},
  "started_at": "2026-08-07T09:14:02Z",
  "finished_at": null,
  "elapsed_s": 4831.2,
  "exit_code": null,
  "results_ready": false,
  "engine": {"model": "Qwen/Qwen2.5-7B-Instruct", "tensor_parallel": "1",
             "max_model_len": "8192", "source": "…/MANIFEST.txt"},
  "log_tail": ["…", "…"]
}
```

### How `phase` is derived

Two sources, and the second one always wins:

1. **`PROGRESS <token>` lines** the eval script prints to its log. The
   supervisor tails the log incrementally and maps tokens to state:
   `booting`, `canary`, `probing`, `running:<condition>`, `rendering`, `done`,
   plus `total <n>`, `case_done <name>`, `case_failed <name>`, `canary_ok`,
   `canary_failed <msg>`, `failed <msg>`. Prose in the log is never parsed, so
   reworded log messages cannot break `/status`.
2. **The child's exit status.** A script that dies without printing
   `PROGRESS done` is `failed` no matter what the log said; a non-zero exit is
   `failed` with the exit code in `error`. `POST /abort` pins the phase to
   `aborted`.

`conditions.completed` is the larger of the `case_done` count and the number of
result JSONs actually on disk, so a resumed run (results already present) still
reports honestly. `engine` comes from `MANIFEST.txt` once the run writes it, and
falls back to the container's env before that.

The terminal phases are `done`, `failed` and `aborted`. **In all three the
supervisor keeps serving** — the run is over, the download window is not.

### What is exposed

`public_url` is not a secret and nothing except `/abort` is authenticated:
anyone with the URL can read `/status`, `/log` and the results. That is a
deliberate trade for retrievability on a platform with no other egress, but it
means **do not run this with an input dataset you would not publish**, and note
that `/status` echoes the model name and the tail of the run log. Only `/abort`
requires `X-Tidal-Key`, and only because it destroys work.

---

## Step 6 — Fetch the results

```bash
curl -s "$PUBLIC_URL/results/" | jq '{count, total_bytes, files: [.files[].name]}'   # what is there
curl -OJ "$PUBLIC_URL/results.tar.gz"                                                # everything
tar -xzf tidal-gpu-*.tar.gz
```

`-OJ` is what picks up the server's filename (`tidal-gpu-<host>-<utc>.tar.gz`).
Before the run is terminal, `/results.tar.gz` returns **404 with the status JSON
in the body** rather than a partial archive:

```json
{"error": "run is still in progress", "status": {"phase": "running:naive", …}}
```

If the script never got as far as packing (a crash, or `POST /abort`), the
supervisor builds a tarball from whatever is in `$RESULTS_DIR` at request time,
so a half-finished run is still retrievable.

Single files, for when you want one number without pulling a 300 MB archive:

```bash
curl -s "$PUBLIC_URL/results/MANIFEST.txt"
curl -s "$PUBLIC_URL/results/gpu-<host>-<utc>/matrix/technique_a.json" | jq '.summary'
curl -s "$PUBLIC_URL/log?tail=2000" | tail -50
```

Read `MANIFEST.txt` first — it carries the canary verdict, the probe's measured
ceiling, every derived pool size, and the ran/skipped/failed case lists.

### Optional backup egress

Set `TIDAL_RESULTS_HF_REPO` (e.g. `you/tidal-gpu-results`) and `HF_TOKEN`, and
the supervisor uploads the tarball to that HF dataset repo as soon as the run
ends. It is **best effort and log-only**: a failed upload is a line in `/log`,
never a failed run, and it does not replace downloading the tarball yourself.
Useful when the reservation might expire before you get back to it.

### Aborting

```bash
curl -X POST "$PUBLIC_URL/abort" -H "X-Tidal-Key: $TIDAL_API_KEY"
```

SIGTERMs the eval's whole process group (the script, its `vllm serve`, and vLLM
V1's EngineCore grandchild — `killpg`, not `terminate`, or GPU processes survive
the abort). The phase becomes `aborted`, the server keeps running, and
`/results.tar.gz` starts serving whatever completed. Without a correct
`X-Tidal-Key` you get 401; if `TIDAL_API_KEY` was never set on the container,
abort is disabled entirely and every request gets 401.

**Teardown** is the `caas-model-deploy` skill's Step 3 procedure —
`DELETE /enterprise/v1/io-cloud/caas/deployment/<CLUSTER_ID>`, one explicit
approval per cluster, then `io-infra-fleet-sync` to close the books. Download
the tarball *first*.

---

## Endpoint reference

| Endpoint | What it returns |
|---|---|
| `GET /healthz` | `200` always. CaaS liveness — deliberately does not report the eval's health, because a failed eval must not get the container restarted out from under its results. |
| `GET /` | The endpoint list and the current phase. |
| `GET /status` | Phase, canary verdict, error, conditions completed/total, `started_at`, `elapsed_s`, engine shape from `MANIFEST.txt`, last 20 log lines. |
| `GET /log?tail=N` | Plain-text tail of the run log. Default 200 lines, capped at 5000. |
| `GET /results/` | JSON listing: every result file with size and mtime. |
| `GET /results/{name}` | One file, safe-joined under `$RESULTS_DIR` (`..`, absolute paths and escaping symlinks are refused with 404). |
| `GET /results.tar.gz` | The tarball once the run is terminal; 404 + status JSON before that. |
| `POST /abort` | SIGTERM the eval process group. Requires `X-Tidal-Key: $TIDAL_API_KEY`. |

---

## What the container actually runs

`run_gpu_eval.sh`, in order, under `$RESULTS_DIR` (`/results`):

1. **Compat canary** *(new: it used to be a manual pre-flight)* —
   `python3 -m pytest tests/unit/test_tidal_scheduler.py -q` inside the
   container, before any GPU time is spent. It exercises `TidalScheduler`
   against *this image's* vLLM using real `Scheduler` machinery and real KV
   block accounting with no weights loaded, so it costs minutes rather than an
   evaluation's worth of GPU hours. It is the single highest-value check on a
   new image and nobody is around to run it by hand any more.

   On failure the run does **not** stop:

   - both `technique_b` cases are skipped — the plugin's numbers would be
     meaningless — while the `technique_a` matrix, the diurnal `technique_a` arm
     and the deadline pair still run and are still worth the GPU time;
   - `/status` carries `canary: "failed"` and a clear `error` from that moment
     on, and it is sticky: it is still there at the end of the run;
   - the failure joins the failed-case list, so the script exits non-zero and
     the **terminal phase is `failed`**, not `done`;
   - `MANIFEST.txt` records it, so a downloaded tarball is self-describing.

   A canary that reports no *passing* tests is treated the same way: the test
   file skips itself when vLLM does not import, which means the image is broken
   in exactly the way that matters. Set `RUN_CANARY=0` only if you have already
   run it against this exact image.
2. **Preflight** — `vllm` on PATH, `tidal.eval.harness` importable, `nvidia-smi`.
3. **Prewarm** — `hf download $MODEL`, so the first condition's health timeout is
   not racing a multi-GB download.
4. **Sizing probe** — a short `offline_only` run. Both the engine-health gate (a
   bad image or missing driver fails here, before hours of matrix runs) and the
   measurement of the offline throughput ceiling in items/s. It is **not**
   selectable through `CASES` and runs even when the `offline_only` *condition*
   is deselected: every pool size downstream is computed from its number.
5. **Matrix** — `online_only`, `offline_only`, `naive`, `technique_a`,
   `technique_b` at `ONLINE_RPS`, same seed, pool sized to
   `POOL_SAFETY × ceiling × window` so it **never drains** — a drained pool
   under-reports batch throughput, and the harness prints a `NOTE:` when it
   happens. Raise `POOL_SAFETY` if you see it.
6. **Diurnal** — `technique_a` and `technique_b` under sinusoidal load, for the
   tide-filling correlation.
7. **Deadline stress** — one control-vs-laxity pair. Both arms are
   `technique_a` with byte-identical flags; they differ only in the dispatcher's
   environment (`TIDAL_ESCALATION_HORIZON_S=1` + an unreachable floor for the
   control, defaults for the laxity arm), with a completion window sized to
   `DEADLINE_TIGHTNESS × ceiling`. The paper could not construct a
   discriminating regime on CPU — **`DEADLINE_TIGHTNESS` is the dial to sweep**,
   and one pair at one setting is a starting point, not an answer.
8. **Figures** — rendered from the matrix directory only, because
   `plots.load_results` keys by the payload's `condition` and the diurnal and
   deadline runs would otherwise collide with the matrix arms. Skipped when
   `CASES` selected no matrix case: there would be nothing new to draw, and a
   render over an empty directory would fail a run that did what it was asked.
9. **Manifest + tarball** — `$RESULTS_DIR/tidal-gpu-<host>-<utc>.tar.gz`.

Every one of steps 5–7 boots its own engine, and every one of them is warmed
first — see **Measurement warm-up** below.

Layout under `$RESULTS_DIR/gpu-<host>-<utc>/`: `matrix/` (+ `figures/`),
`diurnal/`, `deadline/`, `probe/`, `logs/` (per-case stdout, per-case vLLM
engine logs, `compat_canary.log`), `MANIFEST.txt`.

Robustness: `set -euo pipefail`; stdin is closed up front so no step can ever
want a TTY; each case is failure-isolated so one bad condition does not abort
the run; each case is capped by `CASE_TIMEOUT_S`; stale listeners on the eval
port are reaped between cases. **It is resumable** — a case whose result JSON
already parses is skipped, and a bare re-invocation reuses the newest
`gpu-<host>-*` directory. Delete a JSON to force a re-run, or set `RESUME=0`.

Knobs (all settable from `env_variables`): `CASES` `ONLINE_RPS` `MINUTES`
`ONLINE_MAX_TOKENS` `BATCH_MAX_TOKENS` `MAX_INFLIGHT` `POOL_SAFETY`
`PROBE_ITEMS` `DIURNAL_MINUTES` `DEADLINE_MINUTES` `DEADLINE_TIGHTNESS`
`CASE_TIMEOUT_S` `SEED` `EVAL_PORT` `RESUME` `PREWARM` `RUN_CANARY`
`MAKE_TARBALL` `RESULTS_DIR`, plus the engine shape: `TENSOR_PARALLEL_SIZE`
`MAX_MODEL_LEN` `BATCH_CONCURRENCY` (empty = the preset's 512) `WARMUP_S`
(empty = the preset's 45 s).

---

## `CASES` — running less than everything

A full run is ten cases and the better part of four GPU-hours. `CASES` is a
comma-separated subset; anything not listed is not run at all.

```
CASES=online_only,offline_only,naive,technique_a,technique_b,diurnal_technique_a,diurnal_technique_b,deadline_control,deadline_laxity
```

That is also the default, so an unset `CASES` is exactly the run this kit
always did. Whitespace around names is tolerated, duplicates collapse, and the
order you write them in does not matter — the script always runs matrix, then
diurnal, then deadline.

* **An unknown name is fatal in preflight**, before the canary, the weight
  download and any GPU time, and the error lists every valid name. A typo that
  silently fell back to "run everything" is the expensive failure mode this
  exists to prevent, so there is no lenient path.
* **`PROGRESS total <n>` counts the trimmed selection**, so `/status` shows
  `3/3` and not `3/10` — the supervisor's progress bar stays honest. A failed
  canary still discounts the two `technique_b` cases, but only if they were
  selected in the first place.
* **The compat canary and the sizing probe are not selectable and always run.**
  In particular, dropping `offline_only` does *not* drop the probe: the probe
  is its own `offline_only` run against the same engine shape, and it is the
  ceiling divisor every pool size is computed from. `RUN_CANARY=0` remains the
  separate switch for the canary.
* **`MANIFEST.txt` records both** the resolved list (`cases:`, in run order)
  and the raw string you passed (`cases_env:`), so a downloaded tarball says
  which subset produced it — and, by omission, which numbers it does not
  contain.

`CASES` composes with resumability rather than replacing it: a case whose
result JSON already parses is skipped either way, so `CASES=technique_b` after
a canary fix re-runs exactly that arm into the existing run directory.

---

## Measurement warm-up (`WARMUP_S`)

`/health` answering 200 means the weights are loaded, not that the engine is at
steady state. The first requests of a run still pay CUDA-graph capture,
`torch.compile`, kernel autotuning and a cold prefix cache — and unwarmed, that
cost lands inside the measured window of whichever condition the loop happens to
reach first, where it is indistinguishable from latency the scheduler caused.

`WARMUP_S` (default: the preset's 45 s; `0` disables) buys unmeasured
sequential requests fired **after** `/health` and **before** `t0`, capped at 30
requests, built by the same online request builder the measured window uses.
Results, timings and errors are all discarded.

It is *engine* warm-up, not load shape, so it applies to every condition that
boots an engine — including `offline_only` and the sizing probe, whose ceiling
would otherwise be measured against a colder engine than the conditions it is
the divisor for. The engine command line is untouched by it. Each condition
prints one line, `warmup: <n> requests in <x>s`, and `warmup_s` is serialized
into every result JSON's `config`: a warmed run and an unwarmed one are not the
same measurement, and the result file says which one it is.

---

## Harness engine flags

`src/tidal/eval/harness.py` was written for a Mac CPU box, and for a while this
kit worked around that. It no longer needs to: the CPU-specific choices are now
flags on `run`, and `run_gpu_eval.sh` passes them for every case.

**The defaults did not move.** A bare `run` still produces the same command
line, environment and connection pool it always did, so every CPU number in the
paper remains comparable. GPU behaviour is opt-in, and `--gpu-preset` is the
one-flag way to opt in.

| Was | Now | Default | Under `--gpu-preset` |
|---|---|---|---|
| `--enforce-eager` always → no CUDA graphs | `--enforce-eager` / `--no-enforce-eager` | eager | **off** |
| `TORCHDYNAMO_DISABLE=1` always → no `torch.compile` | same flag; the two always travel together | disabled | **enabled** |
| `max_model_len` fixed at 4096, no CLI flag | `--max-model-len N` | `4096` | `4096` (pass it) |
| No `--tensor-parallel-size` → 8×H200 evaluated as 1×H200 | `--tensor-parallel N` | `1` | `1` (pass it) |
| `HF_HUB_OFFLINE=1`, worked around by exporting `0` | `--hf-offline` / `--no-hf-offline` | offline | **online** |
| `httpx` default pool capped batch at ~100 | `--batch-concurrency N` → explicit `httpx.Limits` | `100` | **512** |
| Measurement started the instant `/health` went green | `--warmup-s N` → unmeasured requests before `t0` | `0` | **45** |

`--gpu-preset` only moves *defaults*: an explicit flag beats it in either
direction and at any position, so `--gpu-preset --enforce-eager` is a legitimate
way to ask for CUDA-graph-free numbers on a GPU. It deliberately does **not**
touch `--tensor-parallel` or `--max-model-len` — those are properties of the
node and the run, not of "is this a GPU", so the script passes them from
`TENSOR_PARALLEL_SIZE` / `MAX_MODEL_LEN`.

`TENSOR_PARALLEL_SIZE` must agree with the GPUs the container actually has
(`gpus_per_container` on CaaS, `--gpus` under `docker run`). Larger than the
visible device count fails at engine startup, which the sizing probe catches
before any matrix time is spent.

The engine shape is serialized into every result JSON under `config`, and into
`MANIFEST.txt`. A result file therefore says which engine produced it; do not
compare across differing shapes.

Two constraints remain, both benign because the kit already routes around them:

| Constraint | Where | Why it is left alone |
|---|---|---|
| `DEFAULT_VLLM_BIN` is an absolute Mac path | `harness.py` | `run_gpu_eval.sh` passes `--vllm-bin "$(command -v vllm)"`. |
| `OnlineLoadGen.run()` uses httpx's default 100-connection pool | `eval/loadgen.py` | Only bites if online concurrency exceeds 100 — at `ONLINE_RPS=20` that needs p99 above ~5 s. Watch for it in the `naive` arm, where a flooded engine is the point; if online p99 flattens suspiciously near a fixed value, this is the first suspect. |

Also worth stating plainly: the technique-B numbers are only trustworthy if the
capability probe came up clean and the compat canary is green — which is now
the first line of `/status` and the first line of `MANIFEST.txt`.

---

## Appendix — compose and dev boxes (SSH)

None of this is needed for the CaaS flow. It is for a box you already have a
shell on.

### Serving topology

`docker-compose.gpu.yml` is the *product*: a long-lived engine plus the gateway
— the Batch API on `:8080` in front of vLLM on `:8000`. It is a different thing
from the evaluation, and the two must not share a GPU.

```bash
rsync -az --exclude .venv --exclude .git ./ "$SSH_TARGET:~/tidal/"
ssh "$SSH_TARGET" 'cd ~/tidal && \
  MODEL=Qwen/Qwen2.5-7B-Instruct \
  SCHEDULER_MODE=tidal \
  TIDAL_API_KEY=<a-real-secret> \
  docker compose -f deploy/docker-compose.gpu.yml up -d --build'

ssh "$SSH_TARGET" 'curl -s localhost:8000/health; curl -s localhost:8080/healthz'
ssh "$SSH_TARGET" 'docker logs <engine-container> 2>&1 | grep -m1 "capabilities="'
```

Compose does not set `CAAS`, so `engine-entrypoint.sh` defaults to `MODE=serve`
there and nothing about this changed. Every knob is an env var read by compose:
`MODEL`, `MAX_MODEL_LEN`, `TBT_SLO`, `SCHEDULER_MODE`, `TENSOR_PARALLEL_SIZE`,
`GPU_COUNT`, `VLLM_IMAGE`, `BUILD_MODE`, `ENGINE_PORT`, `GATEWAY_PORT`,
`TIDAL_API_KEY`, `TIDAL_MAX_INFLIGHT`.

To deploy the *serving* stack to CaaS instead of the eval, take the payload
template and set `MODE=serve`, `SCHEDULER_MODE=stock|tidal`, `TBT_SLO`,
`EXTRA_VLLM_ARGS` and a `VLLM_API_KEY` secret; `traffic_port` stays `8000`.

### Running the eval by hand

**Stop the serving stack first.** Then either run the supervisor exactly as CaaS
would (useful for reproducing a status bug):

```bash
docker run --rm -p 8000:8000 \
  --gpus '"device=0"' --ipc host --shm-size 32g \
  -v tidal_hf-cache:/root/.cache/huggingface \
  -v "$PWD/results:/results" \
  -e MODE=selfdrive -e TIDAL_API_KEY=local-secret \
  -e MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e ONLINE_RPS=20 -e MINUTES=15 \
  -e TENSOR_PARALLEL_SIZE=1 -e MAX_MODEL_LEN=8192 \
  <registry>/tidal-engine:fast

curl -s localhost:8000/status | jq .
```

or skip the supervisor and run the script directly, which is the same
evaluation without the HTTP surface:

```bash
docker run --rm \
  --gpus '"device=0"' --ipc host --shm-size 32g \
  -v tidal_hf-cache:/root/.cache/huggingface \
  -v "$PWD/results:/results" \
  -e MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e ONLINE_RPS=20 -e MINUTES=15 \
  --entrypoint tidal-run-gpu-eval \
  <registry>/tidal-engine:fast
```

`--gpus '"device=0"'` with `TENSOR_PARALLEL_SIZE=1` is the single-card run and
leaves the other seven GPUs of an 8-GPU node free; to shard, raise both together
(`--gpus all -e TENSOR_PARALLEL_SIZE=8`). Results land in `./results` either
way, and `scp` gets them off the box.

### Running the compat canary on its own

```bash
docker run --rm --gpus all \
  -v tidal_hf-cache:/root/.cache/huggingface \
  -w /opt/tidal --entrypoint python3 \
  <registry>/tidal-engine:fast \
  -m pytest tests/unit/test_tidal_scheduler.py -q
```

`python3 -m pytest` rather than bare `pytest` is required, not stylistic:
`tests/` has no `__init__.py`, so `from tests.unit.engine_fixtures import ...`
only resolves because `-m` puts the working directory on `sys.path`.

- Green → the interface points plan B relies on are still there.
- `skipped` → vLLM did not import in this image. The image is broken.
- Red → upstream drifted. Re-grep `src/tidal/engine/scheduler.py` against this
  vLLM before trusting any technique-B number, and consider a pinned build.

The fixtures read the HF *config* for `TIDAL_TEST_MODEL` (default
`Qwen/Qwen2.5-0.5B-Instruct`), so the container needs either that config cached
or network access. Set `TIDAL_TEST_MODEL` to a model you have already staged.
