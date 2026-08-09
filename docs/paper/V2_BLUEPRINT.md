# Paper v2 blueprint (Fable's professor verdict, reconciling three reviews)

Inputs: (1) exemplar-structure study of ConServe + BlendServe; (2) Opus structural
memo (treated as input, not verdict); (3) Codex professor memo (pending —
integrate before executing); (4) my own judgment. v1 frozen at tag `paper-v1`.

## Thesis (settled)

**"A batch tier you can deploy today on an unmodified serving engine — and its
measured price."** The deadline contract is the *product framing* (why this is a
tier and not scavenging); placement is the *second finding*; the A-ladder is
*discussion*. The GPU result is the paper. All three reviews converge here;
adopted.

- Headline: 69.3% of the node's steady-state dedicated-offline ceiling at
  1.18×/1.23× online p50/p99 (cross-node), over stock vllm/vllm-openai:v0.26.0.
  Inside the 78–88% band the fork-based literature reports, without the fork.
- Honest label carried in the contribution itself: the deadline contract is
  mechanism-validated, outcome-undemonstrated (Q3 stands as a negative result).

## Title (settled)

**Tidal: A Deadline-Contracted Batch Tier for Unmodified LLM Serving Engines**

## Contributions (C-list; final wording after Codex part 4)

- C1 The measured price of the no-fork placement at GPU scale (Table GPU, CDF fig).
- C2 A deadline contract with both faces (admission refusal + horizon-scaled LLF
  escalation + progress floor), labeled mechanism-validated/outcome-undemonstrated.
- C3 Placement of co-serving policy as a measured design axis — a
  throughput/latency frontier, not a ranking; inversion scoped CPU-only in the
  bullet itself; the one GPU point shows the frontier moves with hardware.
- C4 A ceiling-measurement discipline: steady-state-tail denominator, ±0.62%
  cross-node band, and the invariant *harvest > 100% of a claimed ceiling proves
  the denominator is not a ceiling*.
- Self-calibrating T̂ model: DEMOTED to §3 design + one eval subsection —
  it is ConServe's model form refit online; the delta (no offline profiling
  sweep, R² gate) is one sentence of design, not a numbered contribution.
  (Exemplar study disagreed; my call, pending Codex's part-4 opinion.)
- Deployability artifacts (idempotency, crash recovery): engineering hygiene,
  not a contribution. One §4 paragraph. BUT convert non-fork into a measurement:
  survived a 1,682-commit upstream jump; release-canary drift = one kwarg.

## Structure (~13.5pp body target, single col 11pt)

| § | Content | pp |
|---|---|---|
| Abstract | ≤190 words, headline number by word ~120, no audit narrative, no chronology fossils | 0.3 |
| 1 Intro | 3 problem bullets (each carries a number) → insight → 4 contribution bullets (1 sentence each) → "Results." para. Promote the space/time one-liner here. | 1.75 |
| 2 Motivation | 2.1 one budget/two products (+ NEW Fig 1a: online-only token occupancy vs τ); 2.2 best-effort is not a product (+ Fig 1b: naive arm 27.3×/22.6× as the strawman) ending in R1–R3 requirements; 2.3 placement question (6 lines); GEMV→GEMM paragraph lives here (see below) | 1.5 |
| 3 Design | 3.1 overview (NEW merged Fig: one datapath, both placements, stages ①–⑤); 3.2 contract arithmetic + redesigned priority fig (3 curves: best-effort / wall-clock aging / laxity-with-horizon, hour-12 point annotated); 3.3 technique A (AIMD, claiming, circuit — no lifecycle page); 3.4 technique B (+ 6-line Algorithm 1, cost line per mechanism); 3.5 robustness (consolidated fallbacks) | 3.0 |
| 4 Implementation | 0.5pp: LOC/tests, store, metering, lazycode = 1 sentence + citation. Cut fig-loop. | 0.5 |
| 5 Evaluation | 5 numbered questions up front. 5.1 setup (both testbeds, ONE table) + "Defining the ceiling" (the audit, once, as a rule) ; 5.2 baselines + NEW systems-comparison table T1 (HyGen/ConServe/BlendServe/Niyama/Tidal × deadline?/admission?/online tier?/fork?/placements/harvest); 5.3 GPU headline (Table, merged CDF+harvest fig, diurnal fig w/ shaded fill bin; state diurnal mean offered load 12.1 req/s vs flat 20 — NOT directly comparable, say so); 5.4 placement (merged CPU table with regime column, frontier scatter Fig F6); 5.5 the contract honestly (Q3 negative + shrunk deadline table); 5.6 self-calibration (4 lines + timeline panel) | 5.0 |
| 6 Discussion: widening the control plane | 3 paragraphs: fleet (+32%, r=−0.83/−0.56, confound named), disagg (Dynamo mapping, 1 sentence), freeze/thaw (31.8×, FLOPs unchanged, storage economics). Pointer to Appendix C. | 0.5 |
| 7 Related work | 4 bolded topics; drop-in diff paragraphs vs ConServe and BlendServe (drafted in exemplar study; edit for voice); ask what denominator prior harvest numbers used | 0.75 |
| 8 Limitations | one list of six, one line each | 0.5 |
| 9 Conclusion | 4 sentences; NO "necessary next step" fossil | 0.15 |
| Appendices | A artifact/provenance+spend; B operational semantics+lazycode; C A-ladder (tables 6–8, fig-aladder, fleet_antiphase); D full deadline table + CPU CDF | ~3 |

## The GEMV→GEMM paragraph (user-requested; my technical verdict, Codex to confirm)

TRUE with three required precisions:
1. It applies to the **weight-stream GEMMs** (QKV/O projections, MLP): with B
   concurrent decodes, one HBM weight read serves B tokens — arithmetic
   intensity rises ~linearly in B until the roofline knee. Thin-batch decode is
   GEMV-*like*; co-served batch decodes thicken B toward τ.
2. **Attention does NOT amortize**: each request reads its own KV; KV bytes
   scale with B and context. At large B the regime can flip to KV-bandwidth-bound.
   So "GEMV→GEMM" is the projection/MLP story, said explicitly.
3. Modern engines already batch — the honest phrasing is "raises the arithmetic
   intensity of decode's weight GEMMs by thickening the effective batch," not a
   literal GEMV→GEMM conversion.
Evidence discipline: we have NO controlled batch-size sweep; the measured low
marginal latency (+1408 batch otok/s at 1.18× online p50) is *consistent with*
weight-stream amortization, not a demonstration. Phrase exactly so. (A B-sweep
roofline microbench is a cheap future GPU item; note in roadmap, not paper claim.)

## Corrections all reviews agree must land

- Kill ALL stale sentences (conclusion "necessary next step"; setup "8×H200 …
  must be earned"; "planned GPU evaluation"; singular "testbed"; "subsequent",
  "post-correction" chronology fossils). Grep list: H200, next step, subsequent,
  post-correction, planned, primary goal of the GPU.
- Diurnal comparability: 78.1% carries its mean offered load (12.1 req/s vs
  flat 20) wherever quoted, including abstract.
- Restore the over-withdrawn amplitude finding: GPU tide in phase with CPU tide
  (−0.85 vs −0.95) but 3.4× shallower (−6.4% vs −22% peak/trough split),
  consistent with more headroom at 78% of ceiling than CPU had at 45%.
- Reconcile test counts (single current number, stated once).
- Economics argument appears ONCE (intro). Inversion told ONCE at length (§5.4).
  Ceiling story ONCE (§5.1). Repetition inventory in the two memos is the cut list.
- Floats: `[t]` discipline; no page with <25 lines of body text.

## Figure plan (7 figures + 4 tables in body)

Keep: priority curve (redesigned, 3-curve), technique-B iteration fig, CPU CDF,
GPU CDF (merged w/ harvest panel), diurnal tide (merged CPU+GPU, shaded fill bin),
GPU table, deadline table (3 rows).
NEW: F1 motivation 2-panel; F3 merged overview; F6 placement-frontier scatter
(x = online p99 ratio log, y = % ceiling, points naive/A-CPU/B-CPU/A-GPU,
HyGen–ConServe 78–88% band shaded, B-GPU open marker "not obtained"); T1 systems
comparison table.
Cut from body: fig-loop, technique-A lifecycle page, gpu_throughput single-bar,
batch_throughput bars, CPU diurnal table, fig-aladder + fleet_antiphase (→ App C),
tables taxonomy/fleet/dynamo (→ App C).

## Process

1. Integrate Codex professor memo (esp. parts 4 and 6) into this blueprint.
2. Fable rewrites main.tex from blank buffer against this blueprint (v1 stays
   at tag; same filename).
3. Figure work: matplotlib scripts for F1/F5/F6/F7 from existing JSONs
   (delegated, cost-efficient model); TikZ redesigns F2/F3 (Fable).
4. Compile (tectonic), visual page-by-page check, grep the fossil list.
5. Codex gate on the compiled v2 diff vs blueprint; fix; push.
