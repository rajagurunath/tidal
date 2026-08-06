# Tidal 🌊

**Batch + online co-serving for vLLM** — an OpenAI-compatible Batch API (`/v1/batches`, 24h SLA, half price) that runs on the *same* GPU as your online traffic, filling every iteration's spare token budget with batch work.

> Your GPU's decode iterations use ~1% of their token budget. Tidal sells the other 99%.

## Two techniques, one repo

| | Technique A — Gateway | Technique B — TidalScheduler |
|---|---|---|
| Where policy lives | Sidecar process (external control) | Inside the engine (`--scheduler-cls`) |
| vLLM requirement | Stock, `--scheduling-policy priority` | Latest main, plugin subclass |
| Batch admission | AIMD watermarks on `/metrics` (1 Hz) | Per-iteration token-budget packing with a self-calibrating interference model |
| 24h SLA | Laxity-driven priority escalation (LLF) | Same, plus KV guardband + cheap-victim preemption |

Both give you: durable OpenAI-wire `/v1/files` + `/v1/batches`, crash-safe SQLite/Postgres storage, laxity-based deadline guarantees, and metering at a 50% batch discount. The [paper](docs/paper/) compares them head-to-head.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
vllm serve Qwen/Qwen2.5-0.5B-Instruct --scheduling-policy priority   # terminal 1
tidal serve                                                          # terminal 2
```

Then use any OpenAI SDK against the gateway:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="tidal-dev-key")
f = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                          completion_window="24h")
```

Technique B: add `--scheduler-cls tidal.engine.scheduler.TidalScheduler` to `vllm serve`.

## Status

Alpha, under active development. Design docs in [`docs/plans/`](docs/plans/); research paper draft in [`docs/paper/`](docs/paper/).

Known limitations:

- **One dispatcher per store.** There is no ownership lease, so nothing stops a second `tidal serve` from running a tick loop against the same database. The store makes that safe for *data*: item claims are a single atomic `UPDATE ... RETURNING`, terminal item transitions are idempotent (a duplicate or late result never drifts `request_counts`), and attaching result files is a first-writer-wins claim. It is not safe for *work*: both processes would submit items, meter their own successes, and run independent AIMD targets against one GPU. Run exactly one dispatcher until a lease exists.
- **The default API key is a placeholder.** `tidal serve` warns loudly if it binds to a non-loopback address while `TIDAL_API_KEY` is still `tidal-dev-key`. Set a real secret before exposing the gateway.

## GPU deployment

The paper's numbers come from a deliberately modest CPU testbed. [`deploy/`](deploy/)
is the kit for taking Tidal to a real GPU on io.net CaaS: a vLLM image with
Tidal baked in (`SCHEDULER_MODE=stock|tidal` picks the technique), a slim
gateway image, a GPU compose file, a CaaS deploy payload template, and
`run_gpu_eval.sh` — an unattended, resumable run of the full A/B matrix plus
diurnal and deadline-stress cases that tars its own results.

Start with the runbook: **[`deploy/README.md`](deploy/README.md)**. Read the
vLLM version-pin caveat and run the 10-minute compat canary before trusting any
technique-B measurement.

## License

Apache-2.0.
