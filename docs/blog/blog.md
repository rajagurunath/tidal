# Sell your idle tokens at half price: a Batch API for your own vLLM

Your GPU is bored. Not idle in the way `nvidia-smi` shows — utilization might read 60% — but bored in a more specific way: every scheduler iteration of your vLLM server can process a few thousand tokens, and when it's decoding chat responses for twenty users, it processes about twenty. The weights get streamed from HBM either way. The other ~99% of each iteration's token budget evaporates, every ten milliseconds, all day.

OpenAI and Anthropic figured out how to sell that evaporation. It's called the Batch API: upload a file of requests, get results within 24 hours, pay half price. The discount works because batch tokens are produced in the shadow of online traffic, on capacity that would otherwise be wasted. If you self-host, you can't buy this product and you can't sell it either — vLLM has no `/v1/batches`, and the offline batch runner it does ship wants the whole machine to itself.

Tidal fixes that. It's an open-source system that gives your vLLM deployment the same two products the commercial providers have: online inference at full price and full priority, and a durable, OpenAI-wire-compatible batch tier at half price with a 24-hour completion contract — admission that refuses infeasible jobs, and escalation that spends online headroom only when a deadline demands it — on the same GPU, at the same time.

## The part that isn't obvious: the guarantee

Co-locating batch behind online traffic is a known trick. Research systems (HyGen, ConServe) do it with strict priority: online requests always schedule first, batch fills the leftovers, and under memory pressure batch gets preempted. vLLM's own priority scheduling gives you the mechanism out of the box.

What nobody gives you is the *deadline*. In every existing co-location system, batch work is best-effort: if online traffic stays high, your overnight batch is simply not done in the morning, and nothing in the system ever noticed it drifting. That's fine for a research throughput graph. It's fatal for a product — the entire reason a customer uses a batch API is that "done by morning" is dependable enough to plan around.

Tidal's answer is a sixty-year-old idea from real-time scheduling: **laxity**. For every batch job, once a second:

```
laxity  = time_remaining − work_remaining / observed_rate
urgency = clamp(1 − laxity / 6h, 0, 1)      # ramps only in the last 6h of slack
priority = 100 → 1 as urgency goes 0 → 1    # online is 0; batch never displaces it
```

A batch that's on track has hours of laxity, urgency zero, priority 100 — it never asks for more than the leftover capacity. (Co-location itself isn't free: our measurements show ~1.8× online p99 even with everything behaving. What escalation guarantees is that batch never takes *more* than the deadline requires.) A batch that's falling behind — because your online traffic spiked and the engine preempted its work — sees its laxity shrink and starts climbing the priority ladder *early*, hours before a wall-clock rule would react, and independently of every other batch. At maximum urgency it sits one notch below online priority with a token-bucket floor guaranteeing forward progress. Deadline math, not hope. And the same arithmetic runs at submission time: a batch whose projected completion already exceeds its window is rejected up front with the numbers in the error — a contract has an admission face, not just a scheduling face.

Two details we got wrong first, so you don't have to:

- **Don't scale urgency by the 24h window.** Our first implementation did, and it quietly turned laxity into wall-clock aging — a perfectly healthy batch at hour 12 escalated to priority ~50 for no reason. Scale by a short escalation *horizon* (6h): urgency stays exactly zero until slack actually gets scarce. An adversarial cross-model code review caught this; the regression test now pins it.
- **Don't panic on ignorance.** A freshly submitted batch has no observed completion rate. Divide by epsilon and it looks infinitely late — instant maximum urgency. Tidal escalates on *evidence* (observed rate) or genuine deadline proximity, never on a missing denominator.

## Two ways to place the policy

Tidal ships both, because we wanted to measure the difference honestly.

**Technique A — the gateway.** A sidecar process owns `/v1/files` + `/v1/batches` (any OpenAI SDK works — point `base_url` at it), stores everything durably in SQLite (Postgres is a DSN swap), and injects batch items into a completely stock vLLM as low-priority requests. It watches the engine's public metrics once a second and adjusts how many batch items are in flight with a TCP-style AIMD loop: back off multiplicatively when online requests queue or KV cache runs hot, probe upward otherwise. Online traffic never touches the gateway — it physically cannot add latency to a request it never sees.

**Technique B — the in-engine scheduler.** `--scheduler-cls tidal.engine.scheduler.TidalScheduler`. A subclass that wraps vLLM's V1 scheduler (wraps — not forks, not reimplements) and fills every iteration's token budget: online first, then batch up to a cap computed from an interference model `T̂(P, C)` that predicts step time from tokens-this-step and resident context. The model calibrates itself from the engine's own step timings — no offline profiling sweep, unlike the research systems — and a KV guardband keeps enough cache headroom that online bursts never wait for memory. When pressure does hit, it evicts the batch request with the least unrecoverable work (prefix-cached tokens survive preemption in vLLM, so "cheapest victim" is measurable).

A, in one sentence: policy outside, works with any recent vLLM, reacts at 1 Hz. B: policy inside, needs the plugin hook, reacts every iteration and harvests slack A can't see. The paper has the head-to-head numbers.

## What it looks like

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --scheduling-policy priority   # your existing server
tidal serve                                                          # the batch tier
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://your-box:8080/v1", api_key="...")
f = client.files.create(file=open("overnight.jsonl", "rb"), purpose="batch")
b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                          completion_window="24h")
# ...come back tomorrow
print(client.files.content(client.batches.retrieve(b.id).output_file_id).text)
```

Crash-safe (items re-queue on restart; vLLM's prefix cache makes re-runs cheap), cancel/expiry semantics identical to OpenAI's (including the weird ones, like partial results surviving cancellation), and a metering ledger that prices batch tokens at 0.5× your online price table — so `tidal report` tells you what your idle capacity earned.

And because the wire format is exactly OpenAI's, existing batch-API clients work unchanged. Our test case is [lazycode](https://github.com/rajagurunath/lazycode), a coding agent that plans work with a realtime model and executes it overnight on batch APIs at the discount — its Tidal adapter is the same ~500 lines as its Anthropic one, and the orchestration layer can't tell the difference.

## What we measured

On our CPU testbed (yes, a MacBook — mechanism validation, not GPU numbers): naive co-location with vLLM's priority field alone destroyed online latency (22.6× p99) while the gateway delivered the same batch work at 1.76×. The A-vs-B comparison produced the night's most interesting result — a placement *inversion*: the in-engine scheduler harvested 71% of the offline throughput ceiling versus the gateway's 45%, but at 8–10× online p99 versus the gateway's ~2×, because on this hardware the binding interference is resident-decode contention, which sits below the reach of admission-time control. External-vs-in-engine isn't a ranking; it's a throughput/latency frontier, and nobody in the fork-based literature has measured it because each system only evaluates its own placement. Full distributions, negative results included, in the paper.

Then we rented actual GPUs — RTX A6000s off the io.net marketplace, Qwen2.5-7B, $7.73 total — and the CPU tax mostly evaporated. The gateway harvested **98.7% of that node's own dedicated-offline ceiling** (1408 output tok/s against an independently probed 1426.7) while online latency went from 0.753s → 0.886s p50 and 1.570s → 1.927s p99: **1.18× / 1.23×**. On the CPU box the same technique cost 2.54× p50. The short version: on under-utilized GPUs, co-serving is close to free harvest, and the tide effect grows as utilization rises — the MacBook was the high-utilization regime all along. The diurnal run makes the same point backwards: correlation between online and batch was only −0.38 (versus −0.95 on CPU), because the GPU is so over-provisioned at 20 req/s that batch runs near the ceiling in *both* tides. It still out-harvested the flat run (1607 vs 1408 tok/s) — deeper troughs supply more aggregate slack than constant mid-load. Technique B on GPU: the compat canary passed on release v0.26.0, but we never got the matrix numbers — our own harness has a direct-submission defect that stalls right after warmup, and it ate the offline-only, naive, and technique-B GPU arms. Roadmap item, not a result.

Two things we built after that, both smaller experiments with bigger implications. **Fleets**: give the dispatcher two replicas running phase-shifted diurnal load and let placement follow slack, and it harvests **+32%** more batch work than pinning everything to one replica (2494 vs 1884 items), with per-replica anti-correlation of −0.83 and −0.56 — it really is timing work into each replica's trough. The honest caveat is that our two "replicas" share a memory bus (macOS can't pin cores, only thread counts), so the fleet arm's blended online latency is worse, not better — 3.41s p50 vs 1.44s. GPU fleets with real isolation are where that resolves; what CPU proves is the mechanism. **Freeze/thaw**: prefill a 1973-token prompt in one process, persist the KV to disk, kill the process, thaw in a fresh one — bit-identical greedy decode, TTFT 7.277s → 0.229s, **31.8×**. Below ~1k tokens it isn't worth it (short prompts only get 1.9×, fixed read costs). And it moves FLOPs in time rather than saving any: at 12 KB/token for a 0.5B model — two orders worse for 70B-class — storage is the binding constraint, and the connector we used is debug-grade with all-or-nothing prompt-hash keying. But it's the thing a deadline buys you that nothing else in the stack can: disaggregation separates prefill from decode in space; a contracted batch tier separates them in time.

## Try it

Code, paper draft, and the animated architecture explainer: **github.com/rajagurunath/tidal**. It runs on anything vLLM runs on — including, as we can attest after this week, a CPU-only MacBook.
