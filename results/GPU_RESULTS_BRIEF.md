# GPU + fleet + freeze/thaw results brief (2026-08-09 overnight)

Authoritative numbers for paper integration and tri-model review. Every number
traces to a JSON under `evidence/` or `results/`; nothing here is estimated.

## Setup
io.net dev CaaS marketplace, RTX A6000 48GB nodes (3 distinct physical nodes
across hires), Qwen2.5-7B-Instruct, vllm/vllm-openai:v0.26.0 image + tidal at
pinned SHA, τ=2048 default engine config, MAX_MODEL_LEN=8192, online 20 req/s
Poisson (seed 7), 6-min measured windows, warmup 45s. Total spend: $6.79 in archived price-quote receipts across 6 hires + one 1h
extension (~$0.94, HTTP 200 observed in-session but response not archived to
evidence — noted) ≈ $7.73. Teardown: 3/6 hires have DELETE HTTP 200 receipts;
2 were expiry-reclaimed or already-deleted (driver DELETE returned 400
resource-not-found); the final hire (1333) was deleted manually (HTTP 200
in-session, receipt note in evidence/TEARDOWN_NOTES.md); live check after
close: zero running deployments. $1.98 of spend (hires 1-2) produced no data
(marketplace node failures) — honest cost-per-datapoint includes it.

## Instrument credibility (CORRECTED per independent audit 2026-08-09)
The probe's whole-burst average (1438.3/1426.7/1467.6 otok/s) is RAMP-DOMINATED
(18.5s makespan, ~half pipeline fill at concurrency 512) and its 600-item pool
skews to shorter completions than the matrix pools — it is NOT a ceiling (the
diurnal run 'exceeded' it, the tell). The corrected ceiling = probe steady-state
tail (last 10s): 2049.6 / 2031.5 / 2057.0 otok/s, a ±0.62% cross-node band
(tighter instrument). Throughput normalization is per-node; LATENCY ratios are
cross-node (no paired within-node baseline exists) — labeled as such.

## Main GPU table (evidence/gpu-*/; gpu_summary.md)
| condition | online p50 | online p99 | vs baseline | batch otok/s | % of node ceiling |
|---|---|---|---|---|---|
| online_only (baseline) | 0.753s | 1.570s | 1.00× | — | — |
| technique_a | 0.886s | 1.927s | 1.18× / 1.23× (cross-node) | 1408 | **69.3%** of steady-state ceiling |
| technique_a (diurnal, 20-min) | 0.85s | 1.90s | ~1.13×/1.21× (cross-node) | 1607 | **78.1%** of steady-state ceiling |

Diurnal detail (n=14498 online, 0 errors): mean offered load is 12.1 req/s
(14498/1200s) vs the flat arm's 20 — the rows are different regimes, not a
dose-response. CORRECTION (2026-08-09, v2 figure audit): the earlier
"longer diurnal completions (54.8 vs 44.0)" explanation conflated the flat
matrix pool with the probe pool; recomputed from completion records the
means are nearly identical (diurnal 56.0 vs flat 54.8), so offered load is
the honest explanation. On total tokens/s the diurnal run is at 100.2% of
the whole-burst probe. corr(online/min, batch/min) = **−0.85**
once the first-minute pipeline-fill bin is excluded (the raw −0.38 is a
startup artifact: minute 0 completes 771 items vs 1667–1894 thereafter) —
i.e., the GPU tide AGREES with the CPU tide (−0.95). Load-split with the
artifact removed: 1718 vs 1836 items/min high/low (−6.4%).

## Technique B on GPU — honest status
- Compat canary PASSED on release v0.26.0 after the version-tolerant
  `_preempt_request` fix (the only real drift: one kwarg). 14 engine tests pass
  against BOTH release and dev vLLM.
- Matrix measurement NOT obtained: the harness's direct-submission path fails
  at GPU scale twice-over — (1) the original unbounded/hanging drain (fixed
  with bounded salvage, 12 new tests), then (2) a second defect where the
  case stalls immediately after warmup (log: last line "warmup: 30 requests in
  23.4s"). offline_only and naive GPU arms lost to the same path. CPU naive
  (22.6× p99) stands as the policy-free control. Roadmap item, not paper claim.

## Fleet (A-F) local experiment (results/fleet_a.json, fleet_pinned.json)
2 CPU replicas (Qwen2.5-0.5B, peak 1 rps, 16-token online completions — NOT the GPU setup above; Mac M4 Pro, thread-count isolation ONLY — macOS cannot pin
cores; shared memory bandwidth is a named confound), phase-shifted diurnal
(period 600s, phases 0/π), 20-min window, seed 7, 4000-item pool, identical
in both arms; placement=fleet vs placement=pinned (all batch → replica 0).
- Harvest: fleet 2494 in-window (32.2 tok/s) vs pinned 1884 (24.4) = **+32%**.
- Mechanism: per-replica corr(replica's own analytic online rate, its batch
  completions/2min) = **−0.83 (replica 1), −0.56 (replica 2)**; batch volume
  split ~even (1239/1255) but timed into each replica's trough.
- Cost: blended online p50 3.41s (fleet) vs 1.44s (pinned), p99 12.1 vs 3.7 —
  attributed to the shared-substrate confound (spreading batch loads BOTH
  engines' shared memory bus; pinning sacrifices one replica and spares the
  other). GPU fleets with real isolation are where the latency side resolves;
  the mechanism claim (placement follows slack) is what CPU proves.
- Implementation: fleet.py placement (sticky prefix affinity + least-loaded +
  headroom filter), per-replica AIMD + replica-local circuits, 95 new tests
  (424 total), production-wired (TIDAL_VLLM_REPLICAS).

## Freeze/thaw (A-P) spike (tidal-fleet-lab/ap-spike)
vLLM CPU build, ExampleConnector (the renamed SharedStorageConnector; only
device-agnostic in-tree connector), kv_role=kv_both, prefix caching disabled:
prefill in process 1 → KV persisted to disk (23.6 MB for the 1973-token
prompt: only whole 128-token blocks persist → 1920 cached tokens × exactly
12,288 B/token) → process killed → fresh process thaws → **bit-identical
greedy decode**, TTFT 7.277s → 0.229s = **31.8×** (median of 5). Short prompts (208 tok): only 1.9× (just 128 of 208 tokens cached + fixed per-file read costs) → crossover ≈1k tokens is an INTERPOLATION between the two measured points, not measured. A duplicate arm (in-engine prefix cache) measured 48.2×; the 31.8× cross-process number is quoted as primary because it alone proves durability.
Caveats: debug-grade connector; all-or-nothing exact-prompt-hash keying (no
partial prefix reuse — LMCache/Mooncake territory); total FLOPs unchanged —
only their placement in time moves; at 70B-class KV bytes/token ~2 orders
worse, making storage the binding economic constraint; GPU KV-connector mode
forces piecewise cudagraphs.

## Taxonomy for the paper section
A0 (single aggregated engine, measured on GPU) → A-F (fleet placement,
mechanism proven locally) → A-D (disaggregated-aware injection, design +
Dynamo metric mapping) → A-P (phase-split freeze/thaw, spike-proven locally).
vs Technique B: B = max visibility over min scope (one engine's iteration);
the A-ladder = coarser signals over widening scope, ending in a decision B
cannot express (parking work in time). Dynamo mapping: kv_usage→KVBM block
metrics; waiting→router queue depth; observed rate→per-worker throughput;
disagg nuance: batch is prefill-heavy → target prefill-worker troughs.
One-liner: "disaggregation separates prefill from decode in space; a
deadline-contracted batch tier separates them in time."

## Known defects list (repo issues / paper limitations)
1. Direct-submission arms fail at GPU scale (two-stage defect; salvage fix
   verified in unit tests but a second stall-after-warmup mode remains).
2. macOS fleet isolation is thread-count only.
3. Diurnal-B and B-matrix GPU numbers absent (blocked by defect 1).
4. Grafana per-deployment dashboards don't exist on dev (VPN'd Prometheus
   only); metrics evidence = harness's own 1 Hz scrapes + archived CaaS API
   responses + engine logs (all in evidence/).
