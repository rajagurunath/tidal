# Tidal: Deadline-Guaranteed Batch Serving Under Online SLOs in a Single LLM Engine

**Gurunath Lunkupali Venugopal**

## Abstract

Commercial LLM providers sell two products from the same weights: online inference, priced for latency, and a batch API priced at half the online rate with a 24-hour completion window. Self-hosted deployments have no equivalent of the second product, even though their GPUs spend most of the day far below capacity — a decode-dominated iteration uses a few percent of the engine's per-iteration token budget. Research systems that co-locate offline work behind online traffic (HyGen, ConServe) recover much of this idle capacity, but they treat offline work as best-effort: nothing in them can promise a customer *when* a batch will finish, which is precisely the property that makes the batch-API product sellable. We present Tidal, a system that adds a deadline-*guaranteed* batch tier to vLLM. Tidal contributes: (1) a durable, OpenAI-wire-compatible `/v1/batches` front end over an unmodified engine; (2) a laxity-driven escalation policy — classical Least-Laxity-First applied per batch job — that keeps on-track batches invisible to online traffic and escalates at-risk batches early, continuously, and independently; (3) a work-conserving in-engine scheduler, deployed as a plugin rather than a fork, that fills every iteration's token budget using an interference model calibrated online from live step timings, with no offline profiling; and (4) a head-to-head comparison of external (gateway-only) versus in-engine placement of the same policy, a design axis prior systems do not evaluate. On [EVAL-HW], Tidal sustains [RESULT: batch throughput as % of offline ceiling] while holding online p99 TTFT/TBT within [RESULT]% of a no-batch baseline, and meets [RESULT]% of batch deadlines under load where best-effort co-location misses them. The implementation, including a client that plans coding work onto batch APIs, is open source.

## 1. Introduction

The economics of LLM serving are dominated by GPU hours, and GPU hours are dominated by waste. An engine serving interactive chat traffic runs its scheduler every few tens of milliseconds; in each iteration it may decode one token for each of a few dozen live requests while its compute budget — the number of tokens the hardware could process in that same iteration at acceptable latency — sits in the thousands. Decode is memory-bandwidth-bound: the weights stream from HBM whether the iteration carries 20 tokens or 2,000. The marginal cost of adding prefill work to a decode-heavy iteration is therefore small until a compute knee, a fact exploited by chunked-prefill schedulers [Sarathi-Serve] and by every co-location system since.

Providers monetize this slack directly. OpenAI and Anthropic both sell batch APIs: submit a file of requests, receive results within 24 hours, pay half the online price. The discount is not charity — batch tokens are produced almost free in the shadow of online traffic. A self-hosted deployment running vLLM has no way to do the same. vLLM ships an offline batch runner that requires dedicating the machine, and an online server with no batch job surface at all; a proposal to add SLA tiers to its scheduler was closed without adoption [vLLM RFC #30256].

The research literature has the opposite gap. HyGen [hygen] and ConServe [conserve] co-locate offline requests behind online traffic with strict priority and preemption, recovering 78–88% of dedicated-offline throughput while respecting online SLOs. But their offline tier is pure best-effort: under sustained online load, offline work simply starves, and neither system can bound by how much. Deadline-aware LLM schedulers exist — Niyama [niyama] co-schedules QoS classes with per-request deadlines — but treat deadlines as soft targets to minimize violations of, at second-to-minute scale, with graceful degradation under overload. A 24-hour *guarantee*, the semantics of the commercial product, appears in none of them.

This matters because the guarantee is the product. A customer submits an overnight batch precisely because they can plan around "done by morning." Our own motivating client, lazycode [lazycode] — a coding agent that plans work with a realtime model and executes it on provider batch APIs at the discount — is viable only because the 24-hour window is dependable. Making self-hosted batch serving *sellable* therefore requires promoting the deadline from an optimization target to a contract.

Tidal does this with deliberately classical machinery. Each batch job carries a deadline; the scheduler tracks each job's *laxity* — time remaining minus work remaining at the observed completion rate — and maps urgency continuously onto the engine's priority space. A job that is on track exerts zero pressure on online traffic. A job that falls behind escalates early, hours before a wall-clock rule would notice, and independently of other jobs, avoiding synchronized floods. At maximum urgency a job sits just below online priority with a token-bucket floor guaranteeing forward progress; a configuration flag (`sla_strict`) allows true parity at the deadline, making the latency cost of strictness a measurable policy dial rather than a fixed opinion. This is Least-Laxity-First [llf], sixty years old and none the worse for it; its application per batch-job against an online tier inside an LLM engine is, to our knowledge, new.

We implement the policy at two placements and compare them. *Technique A* places all policy in a sidecar gateway: batch items are injected as low-priority requests into an unmodified vLLM, throttled by additive-increase/multiplicative-decrease watermarks over the engine's public metrics. It works against any recent vLLM release. *Technique B* moves admission into the engine via vLLM's pluggable scheduler interface: a subclass wraps the stock scheduler, fills each iteration's residual token budget with batch work, bounds interference with a latency model in the form of ConServe's context-aware polynomial [conserve] but fitted continuously from the engine's own step timings (no offline profiling), reserves KV-cache headroom for online bursts, and preempts the cheapest victim first. Neither placement forks vLLM.

Contributions:

1. **Hard-deadline semantics for a co-located batch tier.** Laxity-driven continuous escalation plus an admission floor turns a 24-hour window into a contract, not a hope. Prior co-location systems offer no offline deadline at all; prior deadline-aware schedulers offer no guarantee.
2. **A self-calibrating interference model.** Technique B fits its step-latency model from live (tokens, context, time) observations with an R²-gated fallback, removing the offline profiling sweep that HyGen and ConServe both require, and adapting to regime changes (model swap, thermal state) automatically.
3. **Deployability as a first-class constraint.** The entire system is a plugin (`--scheduler-cls`) plus a sidecar over upstream vLLM, with a durable OpenAI-wire batch API, crash recovery, and usage metering at the 50% batch discount — the parts a paper usually omits and an operator cannot.
4. **External vs in-engine policy, measured.** The same deadline policy at two placements, evaluated on identical traces — quantifying what in-engine admission actually buys over a metrics-driven sidecar.

## 2. Background and Motivation

### 2.1 One budget, two products

A vLLM V1 iteration schedules up to `max_num_batched_tokens` (τ) tokens across running and waiting requests; τ is chosen so an iteration's latency fits the time-between-tokens (TBT) target. A running decode consumes one token per iteration; a prefill consumes up to a chunk. The engine's KV cache — a separate budget — bounds how many requests may be *resident*. The token budget is the perishable good: unused tokens in an iteration are gone. [FIGURE: measured token-budget occupancy of a diurnal online trace; the area above the line is the batch product.]

### 2.2 Why best-effort is not a product

Consider a 100k-item batch submitted at hour 0 against a deployment that then runs at sustained high online load. Best-effort co-location delivers whatever slack happens to exist; if that is insufficient, the customer learns at hour 24 that the job is half done. A wall-clock aging rule — "priority rises every hour, parity at hour 24" — fails in the other direction: it escalates small on-track jobs pointlessly (interfering with online traffic for nothing) and escalates large at-risk jobs too late (parity at the deadline means full service begins when time is already gone). What determines feasibility is the relation between remaining work and remaining capacity — which is exactly laxity. [FIGURE: the three policies on the same at-risk batch: best-effort misses; wall-clock aging misses late; laxity meets.]

### 2.3 The placement question

Engine schedulers act per iteration (~10–100 ms) with perfect visibility of the token budget and KV state; an external controller acts per second on aggregate metrics. Intuition says in-engine must win. But the engine's own priority preemption already handles token-granularity interference between controller ticks, and an external controller survives engine upgrades untouched. How much SLO headroom and batch throughput does in-engine placement actually buy? No prior co-location work isolates this variable; we measure it.

## 3. Design

### 3.1 System overview

[FIGURE: architecture — gateway (files/batches API, SQLite/Postgres store, dispatcher) beside vLLM; online traffic direct to vLLM; technique B swaps the stock scheduler for TidalScheduler via --scheduler-cls.]

Batch jobs enter through an OpenAI-wire-compatible `/v1/files` + `/v1/batches` API; any OpenAI SDK works unchanged with a different `base_url`. Requests are parsed, validated line-by-line, and stored durably (SQLite in WAL mode by default; the store is a repository interface, so Postgres is a DSN change). Items progress `pending → inflight → succeeded/failed`, with crash recovery re-queuing resultless inflight items on restart — preemption-tolerant because vLLM's prefix cache makes re-execution of a partially-processed prompt cheap. Online traffic never touches the gateway: the gateway can add batch work, but is physically incapable of delaying an online request.

### 3.2 Laxity-driven escalation (both techniques)

For each active batch b with deadline D_b, remaining item count R_b, and observed completion rate r_b (sliding window, ε-floored):

    laxity(b) = (D_b − now) − R_b / r_b
    urgency(b) = clamp(1 − laxity(b) / H, 0, 1)          H = escalation horizon (6h ≪ W)
    priority(b) = round(P_max − urgency(b) · (P_max − P_min))    P_max=100, P_min=1

vLLM's priority scheduler orders waiting requests by `(priority, arrival)` and, under KV pressure, preempts the lowest-priority running request — so priority 100 batch work is glass: transparent to online traffic, shattered first under pressure. An on-track batch (laxity ≥ H) stays at 100 for its whole life; the horizon H, not the 24h window W, scales urgency — scaling by W would smuggle wall-clock aging back in, escalating healthy batches merely because time has passed (our own first implementation made exactly this mistake; an adversarial cross-model review caught it, and the regression test now pins the hour-12 healthy-batch case). A batch whose rate collapses escalates immediately — not at a threshold hour — and each batch climbs its own trajectory, so escalation never synchronizes into a flood. Above urgency 0.9 a token bucket guarantees a minimum admission rate even at watermark-closed load; at urgency 1.0, priority reaches P_min = 1, one notch below online. With `sla_strict`, it reaches 0: batch items then tie with online requests and win on arrival time. We keep strictness a dial and measure its price in §6.[RESULT-XREF]

Escalation is applied at submission time (vLLM fixes a request's priority at admission). The backlog — where laxity matters — is by construction the un-submitted portion, so submission-time escalation captures the mechanism; in-engine re-aging of already-queued items is future work and would need request-metadata plumbing upstream.

### 3.3 Technique A: external control

A 1 Hz loop scrapes three public metrics (`num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`), smooths them (EWMA, α=0.5), and adjusts a target in-flight batch count by AIMD: halve when online queueing is detected (conservatively: aggregate waiting exceeding our own in-flight count, since the waiting metric is not priority-partitioned) or KV usage crosses a high watermark (0.85); increment when below a low watermark (0.70). Items are claimed from the store in deadline order, then grouped by shared prompt prefix within a batch — recovering most of HyGen's prefix-sharing benefit [hygen] through vLLM's existing prefix cache, with arrival order as the only lever — and submitted at their batch's current laxity priority. Engine unavailability opens a circuit: target zero, in-flight re-queued, resume on scrape success.

The AIMD shape is deliberately TCP-like: the gateway probes for capacity and backs off multiplicatively on congestion signals, keeping batch pressure stable without engine cooperation. Its blind spot is granularity — between ticks, a burst can meet a full complement of batch work; the engine's own preemption covers that window, at recompute cost.

### 3.4 Technique B: in-engine work conservation

`TidalScheduler` subclasses the V1 scheduler behind vLLM's `--scheduler-cls` plug point and wraps — never reimplements — the stock `schedule()`:

**Pre-gate.** Classify waiting requests (priority ≥ 10 ⇒ batch). If no online requests are resident, batch is unconstrained: the stock scheduler fills τ with it (offline mode). Otherwise compute a batch token cap X = max x such that T̂(P_on + x, C) ≤ TBT_slo · (1 − guard), and hide batch waiting requests beyond X from the stock pass (restored exception-safely afterward). Enforcement is at chunk granularity: we bound how many batch requests the stock scheduler may see, sized by their next-chunk estimates, rather than editing its inner token loop — coarser than per-token, faithful in aggregate, and robust to upstream drift (the wrapped surface survived a 1,682-commit jump during this work).

**Interference model.** T̂(P, C) = k₁P + k₂P(P+C) + k₄(P+C) + k₅ — ConServe's context-aware form, which prices the attention cost that a pure token count misses (a 2K-token chunk against 40K of resident context costs ~2.4× the same chunk fresh [conserve]). Unlike ConServe (20-minute offline sweep) and HyGen (re-profiling per model/hardware/SLO change), coefficients are fitted continuously from the engine's own (P, C, step-time) stream: ring buffer, periodic least squares, coefficients clamped non-negative, an R² gate deciding readiness. Before convergence — or if fit quality degrades under a regime change — the cap falls back to a conservative static fraction of τ. The cap solve is the positive root of the induced quadratic, computed in rationalized form so it degrades gracefully to the linear case as k₂ → 0.

**KV guardband.** Batch admission stops when KV usage crosses 1 − H, where H is an EWMA of observed online KV demand clamped to [5%, 30%] — Borg-style reclamation with a safety margin [borg]: enough headroom that a p99 online burst admits without evicting anyone. Eviction of batch work begins proactively at 1 − H/2, choosing the victim with the least unrecoverable work (computed tokens not covered by prefix cache) — the cheapest thing to throw away in a recompute-only engine. Stock priority preemption remains the backstop.

**Telemetry.** The scheduler exports per-step batch/online token counts, unfilled slack (an alarmed condition, not a default — work conservation is the goal), model MAPE, and preemption lost-tokens.

### 3.5 What we deliberately did not build

ConServe's layer-wise safepoint preemption and incremental KV checkpointing [conserve] address unbounded iteration times; with chunked prefill bounding every iteration to τ, our worst-case online admission delay is one bounded iteration (tens of ms), so we take that simplicity. Both remain natural extensions for reproduction on larger models where τ-bounded iterations are still long.

## 4. Implementation

Tidal is ~[LOC] lines of Python: gateway (FastAPI + SQLAlchemy; the API layer is wire-verified against the official OpenAI SDK in unit tests), dispatcher, and the scheduler plugin, with [TESTCOUNT] unit tests and integration tests that drive a live engine. The store's state machines mirror OpenAI's batch semantics exactly (including the un-summed request_counts of cancelled/expired batches). Metering prices batch usage at 0.5× a configurable online price table. The client side is lazycode's provider adapter — the OpenAI-wire twin of its Anthropic batch adapter, including content-keyed idempotent submission and crash-recovery reconciliation — so an agentic client plans work onto a self-hosted batch tier exactly as it does onto a commercial one [lazycode]. Everything runs on an unmodified upstream vLLM (v0.26 series); development and the small-scale evaluation ran on a CPU-only build (Apple M4 Pro), which we note as evidence of the plugin surface's portability.

## 5. Evaluation

Setup: [EVAL-HW: Mac M4 Pro CPU, Qwen2.5-0.5B-Instruct, τ=..., and (if available) H100/H200 runs]. Online load: Poisson and diurnal (sinusoidal) arrival traces of short chat requests, plus a bursty trace. Batch load: [N]-item jobs of [workload]. Conditions: online-only (latency ceiling), offline-only (throughput ceiling), naive co-location (priority field only, no Tidal policy), Technique A, Technique B. Metrics: online TTFT/TBT p50/p99; batch throughput as fraction of offline ceiling; deadline attainment under load; preemption lost-tokens; T̂ MAPE over time.

Questions:
- **Q1 (co-location tax):** what do techniques A and B do to online p99 TTFT/TBT versus online-only? [RESULTS TABLE]
- **Q2 (work conservation):** what fraction of the offline ceiling does each technique sustain at low/mid/high online load? [RESULTS TABLE; HyGen reports 84%, ConServe 78–88% on GPU-class hardware — direct numeric comparison is architecture-relative, shape comparison is fair]
- **Q3 (the guarantee):** under sustained online load sized so best-effort misses a tight deadline, does laxity escalation meet it, and at what online latency price? Including the `sla_strict` dial. [RESULTS FIGURE: the money plot]
- **Q4 (placement):** A vs B on all of the above — what does in-engine buy? [RESULTS + DISCUSSION]
- **Q5 (self-calibration):** T̂ convergence time from cold start, MAPE steady-state, and behavior across a mid-run regime change. [RESULTS FIGURE]

[RESULTS SECTIONS TO BE FILLED FROM tidal/eval RUNS — no numbers appear in this draft that were not measured.]

## 6. Related Work

Sarathi-Serve introduced chunked prefill and stall-free batching, the token-budget iteration structure vLLM V1 inherits [sarathi]. HyGen [hygen] and ConServe [conserve] co-locate offline behind online with strict priority — HyGen with a profiled latency budget and prefix-sharing offline order, ConServe with a context-aware latency model, layer-wise preemption, and incremental KV checkpointing; both are engine forks, and neither gives offline work any deadline. OOCO [ooco] partitions by Roofline analysis; Echo [echo] targets co-location across serving instances. On the deadline side, QLM [qlm] reorders a queue by SLO-violation risk via an ILP estimator (and beats EDF as a baseline); Ascendra [ascendra] and SCORPIO [scorpio] prioritize online requests by deadline risk; Niyama [niyama] is closest to us, co-scheduling interactive and non-interactive classes with per-request deadlines under an EDF/SRPF-interpolated priority — but its deadlines are seconds-to-minutes soft targets with eager relegation under overload, the philosophical opposite of a guarantee. FREESH [freesh] implements true least-laxity scheduling in an LLM engine for energy-aware frequency scaling over a single request class — evidence the math runs in this setting, aimed at a different problem. Within vLLM itself, priority scheduling was motivated by batch/interactive co-location [vllm-rfc-6077], an SLA-tier proposal was declined [vllm-rfc-30256], and an online batch-API endpoint remains an open PR [vllm-pr-44445] — collectively, demand without an assembled answer. Tidal's individual mechanisms are all known; its contribution is the combination — hard 24-hour batch-API semantics, laxity escalation against an online tier, no-profiling calibration, and non-fork deployability — plus the placement comparison none of the above performs.

## 7. Limitations

Single engine, single model; multi-replica routing is future work. Submission-time escalation cannot re-age items already in the engine queue (bounded impact: the backlog dominates laxity, and queue residence is short by construction). The chunk-granularity cap is coarser than per-token enforcement. The step-time signal on GPU builds is complicated by async scheduling; our timing attribution is exact on the CPU build and approximate otherwise. `scheduler_cls` is explicitly not a stable vLLM interface; our wrapper touches a three-method surface and we pin a version, but upstream can break it. The 50% price is inherited from the commercial anchor, not derived from a cost model — deriving the profit-maximizing discount from measured marginal cost is an open question we'd like to answer with production traces.

## 8. Conclusion

The batch API is the rare product where the entire margin is scheduling. Tidal shows that the guarantee that makes it sellable — 24 hours, hard — can be added to an unmodified open-source engine with classical real-time machinery and careful placement of policy, and that the resulting system fills the GPU's idle token budget rather than merely tolerating co-location. We hope it becomes both a usable piece of infrastructure and a baseline for the deadline-aware serving work the literature has so far left soft.

## References

[sarathi] A. Agrawal et al. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. OSDI 2024. arXiv:2403.02310.
[hygen] T. Sun, P. Wang, F. Lai. HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location. arXiv:2501.14808.
[conserve] Y. Qiao et al. ConServe: Fine-Grained GPU Harvesting for LLM Online and Offline Co-Serving. arXiv:2410.01228.
[niyama] Niyama: Breaking the Silos of LLM Inference Serving. arXiv:2503.22562.
[freesh] FREESH: Fair, Real-time, Energy-Efficient Scheduling for LLM Serving. arXiv:2511.00807.
[qlm] QLM: Queue Management for SLO-Oriented Large Language Model Serving. SoCC 2024. arXiv:2407.00047.
[ascendra] Ascendra: Dynamic Request Prioritization for Efficient LLM Serving. arXiv:2504.20828.
[scorpio] SCORPIO: Serving the Right Requests at the Right Time. arXiv:2505.23022.
[ooco] Online-Offline Co-location for LLM Serving. arXiv:2511.21862.
[echo] Echo: Efficient Co-scheduling of Hybrid Online-Offline Tasks. arXiv:2504.03651.
[llf] A. K. Mok. Fundamental Design Problems of Distributed Systems for the Hard-Real-Time Environment. MIT, 1983.
[borg] A. Verma et al. Large-scale cluster management at Google with Borg. EuroSys 2015.
[vllm] W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. arXiv:2309.06180.
[vllm-rfc-6077] vLLM RFC: Priority Scheduling. github.com/vllm-project/vllm/issues/6077.
[vllm-rfc-30256] vLLM RFC: SLA-Tiered Scheduling (closed, not planned). github.com/vllm-project/vllm/issues/30256.
[vllm-pr-44445] vLLM PR: OpenAI-compatible online Batch and Files API (open). github.com/vllm-project/vllm/pull/44445.
[infercept] R. Abhyankar et al. InferCept: Efficient Intercept Support for Augmented LLM Inference. arXiv:2402.01869.
[lazycode] lazycode: plan with a realtime LLM, execute on batch APIs. github.com/rajagurunath/lazycode.
