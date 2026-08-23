# The KV lease: deadline-scheduled KV-cache management for a contracted batch tier

**Research plan — 2026-08-23. Status: Phase 0 (no GPU required) ready to start.**

## Thesis

Every KV-cache system today moves *cache to meet the work*: requests arrive
when they arrive, and the cache layer scrambles to have the right blocks
warm. A deadline-contracted batch tier can do the inverse: **move the work
in time to meet the cache.** A batch item with hours of laxity runs when the
dispatcher decides — so the re-reference time of its KV is not *predicted*
(the reason online caches settle for LRU), it is *chosen*. Eviction becomes
Belady-implementable; prefetch and dispatch become one co-scheduled
decision; and the VRAM/DRAM/disk tiering follows the tide — online KV owns
VRAM unconditionally (the guardband, restated), batch KV is promoted from
disk as its trough approaches and demoted at the crest.

The second half of the claim comes from owning both sides of the stack.
Provider batch APIs erase structure the client had: LazyCode knows wave
k+1 re-reads the same repo context plus wave k's outputs, knows the whole
DAG, knows the deadline — and the API forces it to submit strangers. If the
client passes a KV hint (content-addressed: LazyCode's memo keys and wave
ids are already the right cache keys), Tidal can hold a **KV lease**: the
job pays prefill on shared context C once, the KV freezes to the store
between waves, and thaws for the next wave. The A-P rung already spiked the
mechanism (freeze/thaw, 31.8× faster TTFT, bit-identical output); this plan
generalizes that rung from "park work in time" to "park *state* in time
alongside the work."

One sentence for both papers' shared thesis: **batch APIs are stateless
only because the provider–customer boundary forces them to be; a co-owned
stack can price statefulness.** Tidal v1 sold a deadline. This sells
deadline + memory.

## What we already have (assets)

- Tidal gateway + laxity dispatcher over unmodified vLLM (paper v2.3, DOI
  10.5281/zenodo.22067989); the A-P freeze/thaw spike: 31.8× TTFT,
  bit-identical, KV ≈ 12 KiB/token on the spike model.
- LazyCode: content-addressed waves, memo keys, plan DAGs — a real agentic
  batch client that emits exactly the structure hints this plan needs
  (paper DOI 10.5281/zenodo.21905469, §Measured economics).
- A receipted three-arm study (OpenRouter, 2026-08-21) that measured the
  gaps this plan closes: cache stacking unbookable on public lanes;
  compiled-call economics R < T/5 with the restore term ε unmeasured.
- Back-of-envelope ε (restore cost / prefill cost): Qwen2.5-7B ≈ 56 KB of
  KV per token → an 8k-token prefix ≈ 450 MB ≈ 0.1 s off NVMe vs 1–2 s of
  prefill on an A6000-class card → ε ≈ 0.05–0.1, consistent with the 31.8×
  spike. If ε holds, LazyCode's input condition improves from R < T/5 to
  R·ε < T/5 — effectively removing depth-collapse pressure for leased jobs.

## Candidate contributions (what the paper would claim)

- **C1 — Decided-reuse tiering.** KV eviction/offload driven by *declared*
  next-use times (the dispatcher sets them), with dispatch and prefetch
  co-scheduled. Belady's algorithm made implementable by a scheduler that
  controls the future instead of predicting it.
- **C2 — Prefix-aware laxity dispatch.** Maximize prefix-hit rate subject
  to per-item deadlines and the online SLA guardband: group siblings by
  prefix hash into one trough; *delay* an item within its laxity until a
  sibling has warmed its prefix. (BatchLLM's grouping + deadlines +
  co-serving interference = a new scheduling formulation.)
- **C3 — The KV lease across agentic waves.** Session-KV for a stateless
  lane, keyed by the client's content addresses; measured R·ε < T/5
  closing LazyCode's biggest open economic term.
- **C4 — The stateful-batch contract.** An `kv_hint` API extension
  (job_id, prefix_hash, next_use_after) and a priced "affinity batch"
  product: cache-rate restores grounded in measured ε, not a marketing
  discount. No multi-tenant public provider can offer this; that is the
  point.

## Phase 0 — related-work sweep + design (no GPU, ~3 days)

Sweep and position against, at minimum: LMCache / CacheGen / CacheBlend;
Mooncake; CachedAttention/AttentionStore (ATC'24); BatchLLM; SGLang
RadixAttention; Preble; MemServe; **Parrot (OSDI'24 — the closest
ancestor: app-level DAG passed to the engine)**; Autellix; KVFlow-style
agent-workflow KV work; NVIDIA Dynamo KVBM. This space moves monthly —
sweep again before submission.

**Kill / pivot criteria (write the verdict down before building):**
- If laxity/deadline-driven KV tiering with co-decided dispatch+prefetch
  already exists → drop C1/C2, pivot the paper to C3/C4 (contract +
  economics), which survive on receipts.
- If agent-DAG KV hints in a *batch/deferred* setting already exist →
  sharpen C3 to the co-serving guardband + pricing angle only.
- If both exist → the contribution is the measured joint economics and the
  product framing; downgrade to a strong workshop paper / Tidal v3 section.

Also in Phase 0: specify the `kv_hint` wire extension to /v1/batches;
write a trace-driven simulator of dispatch policies (reuse the diurnal
traces from Tidal v2.2) to get upper-bound curves for C1/C2 before
spending a GPU-dollar. Deliverables: positioning note, API spec, simulator
plots.

## Phase A — mechanism, ε measured [NEEDS GPU: 1 node, ~1 week]

vLLM + LMCache (or the A-P spike harness extended) on one GPU with local
NVMe. Replay a real LazyCode 3-wave job. Measure: ε as a function of
context length (2k–64k) and inter-wave gap (minutes–hours); restore
bandwidth off DRAM vs NVMe; bit-identity of outputs after thaw; VRAM
pressure interaction with a synthetic online load. Deliverable: the ε
table — the single number C3/C4 price from.

## Phase B — prefix-aware laxity dispatcher [NEEDS GPU: same node, 1–2 weeks]

Implement C2 in the gateway (pure sidecar scheduling — **no vLLM
modifications**, preserving Tidal's identity). Under the diurnal online
trace + batch backlog, compare four policies: laxity-only (v2.2 baseline),
prefix-grouped, prefix-grouped + delayed-for-warmth, full C1+C2 with
disk tier. Metrics: prefix-hit rate, tokens recomputed, harvest %, online
p50/p99 vs guardband, effective $/token. Deliverable: the policy-ladder
table (the paper's headline figure).

## Phase C — joint economics with LazyCode [same node, ~3 days]

The "self-hosted lane" run both papers name as their next step, now with a
mechanism attached: a LazyCode job end-to-end through Tidal-with-lease,
per-item token ledgers, measured R·ε against the receipted public-lane
numbers from the 2026-08 study. Deliverable: the joint LazyCode×Tidal
economics section.

## Phase D — open ideas / bigger metal [NEEDS GPU: 2 nodes → 8×H200]

Explicitly a roadmap, not a promise:
- **KV pool across replicas** (A-F rung × KV): leases that survive replica
  reassignment; 2 nodes.
- **Disaggregated lease** (A-D/A-P combined): prefill worker writes the
  lease, decode worker thaws it hours later; 2+ nodes or one 8×GPU box.
- **Cold-tier compression** (CacheGen-style) for week-scale leases.
- **Partial reuse of wave outputs** (CacheBlend-style non-prefix fusion of
  wave-k results into wave-k+1 prompts) — high risk, high novelty.
- **Multi-tenant fairness + privacy**: content-addressed dedup across
  users of the same repo; KV is prompt-derived, so store policy = prompt
  retention policy.
- **Interaction with speculative decoding** on the batch lane.

## Resource ask (the io.net section)

| Phase | Hardware | Duration | Est. market cost | Blocked today? |
|---|---|---|---|---|
| 0 | none (laptop) | 3 days | $0 | no — starts now |
| A–C | 1× GPU ≥48 GB (A6000/L40S/H100), ≥1 TB local NVMe | 2–3 weeks | ~$150–400 rented | yes — io.net prod nodes currently saturated |
| D | 2 nodes, then one 8×H200 window | 1–2 weeks | larger; scheduled window fine | yes |

What the grantor gets: a published paper (both prior papers carry DOIs and
public repos) crediting io.net compute; an open-source gateway extension;
a blog/LinkedIn narrative that is *directly* io.net's pitch — a batch tier
that monetizes idle GPU capacity, now with a memory product on top.
Fallback if the ask stalls: Phases A–C are small enough to self-fund; only
Phase D genuinely needs granted metal.

## Success criteria

The paper stands if: ε ≤ 0.15 measured (C3 economics meaningful); the full
policy beats laxity-only by ≥30% on tokens-recomputed at equal online p99
(C2 real); one LazyCode job runs the lease end-to-end bit-identical (C3
mechanism); and the related-work sweep leaves at least C1-or-C2 plus C4
standing (novelty). If only C4 survives, it becomes a Tidal v3 section
rather than a standalone paper — written down now so we cannot move the
goalposts later.
