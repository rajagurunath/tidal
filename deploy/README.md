# Tidal GPU deployment kit (io.net CaaS)

A quick, unofficial way to get Tidal onto a real GPU: build one engine image
with Tidal baked into vLLM, put it on an io.net CaaS box, and run the full A/B
matrix unattended.

> **Unofficial.** This is a test deployment kit, not a production fleet
> procedure. The io.net CaaS deploy endpoint is approval-gated for good reason —
> every step that costs money or touches a shared box is called out below and
> should be run by hand, with eyes on it.

## Files

| File | What it is |
|---|---|
| `Dockerfile.engine` | vLLM GPU image + Tidal. `BUILD_MODE=fast` (default) or `pinned`. |
| `Dockerfile.gateway` | Slim Python image running `tidal serve`. No CUDA. |
| `engine-entrypoint.sh` | Turns `SCHEDULER_MODE` / `MODEL` / `MAX_MODEL_LEN` / `TBT_SLO` into a `vllm serve` command line. |
| `docker-compose.gpu.yml` | Serving topology: engine (GPU reservation) + gateway, healthchecked. |
| `caas-payload.template.json` | io.net CaaS deploy payload, `${PLACEHOLDER}` form. |
| `run_gpu_eval.sh` | Unattended evaluation: probe → 5-condition matrix → diurnal → deadline pair → figures → tarball. |
| `Dockerfile.*.dockerignore` | BuildKit-scoped ignore rules, so the kit does not need a repo-root `.dockerignore`. |

## Two topologies, and why they are not the same thing

**Serving** (`docker-compose.gpu.yml`): a long-lived engine plus the gateway.
This is the product — the Batch API on `:8080` in front of vLLM on `:8000`.

**Evaluation** (`run_gpu_eval.sh`): the harness *owns its engine*. For each
condition it launches a fresh `vllm serve` (stock, or with `--scheduler-cls`),
waits for `/health`, drives the load, and kills the process group. A second
engine on the same GPU makes every latency number meaningless, so **stop the
serving stack before running evals**.

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
worse than a crash if nobody looks. Look:

```bash
docker logs <engine-container> 2>&1 | grep -m1 'capabilities='
```

Any `fallback:` or missing capability in that line means a mechanism the paper
measures is not running in this image. Two ways to pin:

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

**On a Mac (arm64), you must cross-build or build on the node.** The vLLM base
images are `linux/amd64` only.

```bash
# From the repo root. Cross-build; VERIFY_IMPORTS=0 because importing torch
# under QEMU emulation is pathologically slow.
docker build --platform linux/amd64 \
  -f deploy/Dockerfile.engine \
  --build-arg VERIFY_IMPORTS=0 \
  -t <registry>/tidal-engine:fast .

docker build --platform linux/amd64 \
  -f deploy/Dockerfile.gateway \
  -t <registry>/tidal-gateway:latest .

docker push <registry>/tidal-engine:fast
docker push <registry>/tidal-gateway:latest
```

**Faster and more reliable: build on the node itself.** `rsync` the repo up and
`docker build` there natively — no emulation, and `VERIFY_IMPORTS=1` then
actually earns its keep by failing the build if this image's vLLM no longer
satisfies `import tidal.engine.scheduler`.

CaaS needs a registry it can pull from. A public tag is simplest; otherwise put
the credentials in the payload's `registry_config`.

---

## Step 2 — Hire or choose a node, then read its identity

Full procedure: the **`caas-model-deploy`** skill, Step 2. In short, over SSH
(`ubuntu@<public_ip>`):

```bash
ssh "$SSH_TARGET" "cat /etc/ionet_device_id"                                    # device_id
ssh "$SSH_TARGET" "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv"
ssh "$SSH_TARGET" "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'"
ssh "$SSH_TARGET" "free -h; grep -iE 'HugePages_(Total|Free|Rsvd)' /proc/meminfo"
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
start into ~5 min. The skill's Step 4.5 has the exact `hf download` container.

---

## Step 3 — Look up `node_pool_id` and `hardware_id`

These come from **Prod RDS (Preset `database_id=4`)** — do not guess them. The
`caas-model-deploy` skill's Step 4 has the two queries verbatim; the shape is:

- `node_pool_devices` joined to `node_pools`, filtered by `device_id`. The
  column is **`pool_id`**, not `node_pool_id` — the wrong name silently returns
  zero rows.
- `devices` filtered by `device_id`, for `hardware_id` (a *numeric string*:
  keep it quoted) and `hardware_quantity`.

---

## Step 4 — Fill the payload

`caas-payload.template.json` is `envsubst`-shaped:

```bash
export RESOURCE_PRIVATE_NAME=tidal-vllm-h200-1   # <model-slug>-<backend>-<gpu-tier>-<idx>
export DEVICE_ID=...            # /etc/ionet_device_id      (Step 2)
export HARDWARE_ID=...          # devices.hardware_id       (Step 3, quoted string)
export NODE_POOL_ID=...         # node_pool_devices.pool_id (Step 3)
export CREATED_BY=<your-handle>
export ENGINE_IMAGE=<registry>/tidal-engine:fast
export REGISTRY_USERNAME= REGISTRY_SECRET=       # empty for a public image
export SCHEDULER_MODE=tidal     # or `stock` for technique A
export MODEL=Qwen/Qwen2.5-7B-Instruct
export MAX_MODEL_LEN=8192 TBT_SLO=200 TENSOR_PARALLEL_SIZE=1
export EXTRA_VLLM_ARGS="--enable-prefix-caching"
export HF_TOKEN=...             # gated models only; never commit this
export VLLM_API_KEY=...         # the serving key

envsubst < deploy/caas-payload.template.json > /tmp/tidal-payload.json
jq . /tmp/tidal-payload.json    # must parse
```

Field by field:

| Field | Where it comes from |
|---|---|
| `resource_private_name` | You pick it. Fleet convention `<model-slug>-<backend>-<gpu-tier>-<idx>`; take the next free index by checking `resources.resource_private_name` in Prod RDS. |
| `device_id` | `/etc/ionet_device_id` on the box (Step 2). |
| `duration_hours` | How long to hold the reservation. `24` here; the fleet's long-lived deployments use `8600`. |
| `gpus_per_container` | **Must equal `TENSOR_PARALLEL_SIZE`.** Left as a literal `1` because JSON numbers cannot hold a placeholder — edit it by hand if you change TP. |
| `hardware_id` | Prod RDS `devices.hardware_id` (Step 3). Quoted string, e.g. `"203"`. |
| `node_pool_id` | Prod RDS `node_pool_devices.pool_id` (Step 3). |
| `container_config.env_variables` | Consumed by `engine-entrypoint.sh`. `SCHEDULER_MODE` is the technique A/B switch. |
| `container_config.secret_env_variables` | HF token and serving API key. Kept out of `env_variables` so they are not echoed back by the status endpoint. |
| `container_config.entrypoint` | `/usr/local/bin/tidal-engine-entrypoint`. Everything else is env-driven, which is why `args` is empty — a flat JSON array is a miserable place to keep knobs. |
| `container_config.traffic_port` | `8000`, matching `PORT`. Not published to a host port — see below. |
| `registry_config.image_url` | Your pushed engine tag. |

The gateway is **not** in this payload. Run it as a second CaaS deployment
(same shape, `Dockerfile.gateway` image, `traffic_port` 8080,
`TIDAL_VLLM_BASE_URL` pointing at the engine's `public_url`), or just run it
next to the engine over SSH — it needs no GPU.

---

## Step 5 — Deploy (approval-gated)

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

`traffic_port` is not published to a host port — CaaS containers live in a
Nomad network namespace. Get the real address from the containers endpoint:

```bash
curl -s "https://api.io.solutions/enterprise/v1/io-cloud/caas/deployment/<CLUSTER_ID>/containers" \
  -H "x-api-key: $IOCLOUD_API_KEY" | jq -r '.data.workers[].public_url'
```

Before that URL is live, smoke-test from inside:
`docker exec <container> curl -s http://localhost:8000/health`.

**Teardown** is the skill's Step 3 procedure —
`DELETE /enterprise/v1/io-cloud/caas/deployment/<CLUSTER_ID>`, one explicit
approval per cluster, then `io-infra-fleet-sync` to close the books.

---

## Alternative — skip CaaS, SSH and compose

For a throwaway test on a box you already have, this is faster and has fewer
moving parts:

```bash
rsync -az --exclude .venv --exclude .git ./ "$SSH_TARGET:~/tidal/"
ssh "$SSH_TARGET" 'cd ~/tidal && \
  MODEL=Qwen/Qwen2.5-7B-Instruct \
  SCHEDULER_MODE=tidal \
  TIDAL_API_KEY=<a-real-secret> \
  HF_TOKEN=<token-if-gated> \
  docker compose -f deploy/docker-compose.gpu.yml up -d --build'

ssh "$SSH_TARGET" 'docker compose -f ~/tidal/deploy/docker-compose.gpu.yml ps'
ssh "$SSH_TARGET" 'curl -s localhost:8000/health; curl -s localhost:8080/healthz'
```

Every knob is an env var read by compose: `MODEL`, `MAX_MODEL_LEN`, `TBT_SLO`,
`SCHEDULER_MODE`, `TENSOR_PARALLEL_SIZE`, `GPU_COUNT`, `VLLM_IMAGE`,
`BUILD_MODE`, `ENGINE_PORT`, `GATEWAY_PORT`, `TIDAL_API_KEY`, `TIDAL_MAX_INFLIGHT`.

---

## The 10-minute compat canary — run this before any eval

The single highest-value check on a new image. It exercises `TidalScheduler`
against *this image's* vLLM using real `Scheduler` machinery and real KV block
accounting, with no model weights loaded, so it takes minutes rather than an
evaluation's worth of GPU hours:

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

- Green → the interface points plan B relies on are still there. Proceed.
- `skipped` → vLLM did not import in this image. The image is broken; do not
  proceed.
- Red → upstream drifted. Re-grep `src/tidal/engine/scheduler.py` against this
  vLLM before trusting any technique-B number, and consider `BUILD_MODE=pinned`.

The fixtures read the HF *config* for `TIDAL_TEST_MODEL` (default
`Qwen/Qwen2.5-0.5B-Instruct`), so the container needs either that config cached
or network access. Set `TIDAL_TEST_MODEL` to a model you have already staged.

---

## Step 6 — Run the evaluation

Stop the serving engine first. Then, from the repo root on the node:

```bash
docker compose -f deploy/docker-compose.gpu.yml stop engine gateway

docker run --rm \
  --gpus '"device=0"' --ipc host --shm-size 32g \
  -v tidal_hf-cache:/root/.cache/huggingface \
  -v "$PWD/results:/opt/tidal/results" \
  -e MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e ONLINE_RPS=20 -e MINUTES=15 \
  -e HF_TOKEN="$HF_TOKEN" \
  --entrypoint tidal-run-gpu-eval \
  <registry>/tidal-engine:fast
```

`--gpus '"device=0"'` is deliberate: **the harness never passes
`--tensor-parallel-size`**, so every eval condition is a single-GPU run whatever
the box has. Pinning to one card makes that explicit and leaves the rest of an
8-GPU node free.

Expect roughly 2–3 hours at the defaults. Run it under `tmux`/`nohup`.

What it does, in order:

1. **Preflight** — `vllm` on PATH, `tidal.eval.harness` importable, `nvidia-smi`.
2. **Prewarm** — `hf download $MODEL`, so the first condition's health timeout
   is not racing a multi-GB download.
3. **Sizing probe** — a short `offline_only` run. This is both the engine-health
   gate (a bad image or missing driver fails here, before hours of matrix runs)
   and the measurement of the offline throughput ceiling in items/s.
4. **Matrix** — `online_only`, `offline_only`, `naive`, `technique_a`,
   `technique_b` at `ONLINE_RPS=20`, same seed, pool sized to
   `POOL_SAFETY × ceiling × window` so it **never drains** — a drained pool
   under-reports batch throughput, and the harness prints a `NOTE:` when it
   happens. Raise `POOL_SAFETY` if you see it.
5. **Diurnal** — `technique_a` and `technique_b` under sinusoidal load, for the
   tide-filling correlation.
6. **Deadline stress** — one control-vs-laxity pair. Both arms are
   `technique_a` with byte-identical flags; they differ only in the dispatcher's
   environment (`TIDAL_ESCALATION_HORIZON_S=1` + an unreachable floor for the
   control, defaults for the laxity arm), with a completion window sized to
   `DEADLINE_TIGHTNESS × ceiling`. The paper could not construct a
   discriminating regime on CPU — **`DEADLINE_TIGHTNESS` is the dial to sweep**,
   and one pair at one setting is a starting point, not an answer.
7. **Figures** — rendered from the matrix directory only, because
   `plots.load_results` keys by the payload's `condition` and the diurnal and
   deadline runs would otherwise collide with the matrix arms.
8. **Manifest + tarball** — `results/tidal-gpu-<host>-<utc>.tar.gz`.

Robustness: `set -euo pipefail`; each case is failure-isolated so one bad
condition does not abort the run; each case is capped by `CASE_TIMEOUT_S`;
stale listeners on the eval port are reaped between cases. **It is resumable** —
a case whose result JSON already parses is skipped, and a bare re-invocation
reuses the newest `gpu-<host>-*` directory. Delete a JSON to force a re-run, or
set `RESUME=0` for a clean one.

Knobs: `ONLINE_RPS` `MINUTES` `ONLINE_MAX_TOKENS` `BATCH_MAX_TOKENS`
`MAX_INFLIGHT` `POOL_SAFETY` `PROBE_ITEMS` `DIURNAL_MINUTES` `DEADLINE_MINUTES`
`DEADLINE_TIGHTNESS` `CASE_TIMEOUT_S` `SEED` `EVAL_PORT` `RESUME` `PREWARM`.

## Step 7 — Fetch results

```bash
scp "$SSH_TARGET:~/tidal/results/tidal-gpu-*.tar.gz" ./
tar -xzf tidal-gpu-*.tar.gz
```

Read `MANIFEST.txt` first — it carries the probe's measured ceiling, every
derived pool size, and the ran/skipped/failed case lists.

---

## Known harness constraints on GPU

`src/tidal/eval/harness.py` was written for a Mac CPU box. Several choices are
hardcoded and are **not** what you want at GPU scale. This kit deliberately does
not patch the harness — but you should know what you are measuring, and each of
these is a one-line change:

| Constraint | Where | Effect on GPU | Fix |
|---|---|---|---|
| `--enforce-eager` always | `VllmServer.command()` | No CUDA graphs. Decode throughput well below what the card can do. | Make it conditional on a flavor flag. |
| `TORCHDYNAMO_DISABLE=1` always | `VllmServer.environment()` | No `torch.compile`. Compounds the above. | Only set it on CPU builds. |
| `max_model_len` defaults to 4096, no CLI flag on `run` | `VllmServer` / `harness.run` | Long-context behaviour is untested no matter what `MAX_MODEL_LEN` the image sets. | Add `--max-model-len` to `run` and thread it into `RunConfig`. |
| No `--tensor-parallel-size` | `VllmServer.command()` | Every eval is single-GPU. An 8×H200 node is evaluated as 1×H200. | Add a `tp` field to `RunConfig`. |
| `HF_HUB_OFFLINE` defaults to `1` | `VllmServer.environment()` | A cold node cannot fetch weights. `run_gpu_eval.sh` exports `0` to work around it. | Default it to unset. |
| `httpx.AsyncClient()` default pool | `submit_batch_direct()` | Batch concurrency is capped at ~100 regardless of pool size, in `offline_only`, `naive`, and `technique_b`. | Pass explicit `httpx.Limits`. |
| `DEFAULT_VLLM_BIN` is an absolute Mac path | `harness.py` | Unusable in a container. `run_gpu_eval.sh` passes `--vllm-bin` explicitly. | Default to `shutil.which("vllm")`. |

Also worth stating plainly: the technique-B numbers are only trustworthy if the
capability probe came up clean (Step 0) and the compat canary is green.
