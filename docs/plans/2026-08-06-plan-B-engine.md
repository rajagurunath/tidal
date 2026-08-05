# Plan B — TidalScheduler (in-engine work-conserving co-scheduler) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `scheduler_cls` plugin for latest vLLM V1 (checkout `vllm-experiments/vllm` @ e6d67fdd) that fills every iteration's token budget — online first, batch in the slack — bounded by a self-calibrating interference model and a KV guardband.

**Architecture:** Wrapper-subclass strategy: `TidalScheduler(Scheduler)` does NOT reimplement the 600-line `schedule()`. It gates *around* it — pre-hook decides which batch requests are eligible this step (side-queue the rest), post-hook records telemetry and calibration samples. Spec: `vllm-experiments/docs/superpowers/specs/2026-08-05-tidal-scheduler-design.md`.

**Key mechanism decisions (deviations from spec, documented for the paper):**
1. **Token cap at chunk granularity.** The per-step batch token cap X cannot be enforced inside `super().schedule()`'s inner loop without copying it. Instead we bound *how many batch requests are visible* to the stock scheduler each step: Σ expected batch tokens ≈ n_batch_prefills×chunk + n_batch_decodes ≤ X. Coarser than per-token, faithful in aggregate, and robust to upstream drift.
2. **Proactive cheap-victim eviction in the pre-hook** (our victim choice via `self._preempt_request`), stock inline preemption stays as backstop (its victim rule is inline in `schedule()`, not overridable).
3. **In-engine laxity aging is OUT of v1** — the engine only sees a priority int; deadlines live in the gateway, which applies laxity at submission time. Documented limitation + future work.
4. **Step time** = monotonic delta between consecutive `schedule()` calls, attributed to the previous step's (P, C). CPU-safe; GPU-async caveat documented.

**Tech Stack:** Python, numpy (fitting), vLLM from the host venv (`vllm-experiments/.venv`, editable @ latest main). Engine tests run in THAT venv, not tidal's own.

## Global Constraints

- Never modify files under `vllm-experiments/vllm/` (upstream checkout). All code in `tidal/src/tidal/engine/`.
- Batch classification: `request.priority >= 10` (gateway bands 100..1; online 0). Priority 1 items (max escalation) count as batch for capping but sort ahead of all batch in the stock heap — intended.
- Config from `TidalConfig` fields: `tbt_slo_ms, interference_guard, cold_start_batch_frac, fit_window_steps, fit_r2_gate, kv_guardband_min/max`.
- vLLM interface points (verified @ e6d67fdd): `schedule()` (scheduler.py:440), `SchedulingPolicy.PRIORITY` preemption (:591), `_preempt_request` (:1275), `update_from_output` (:1671), `SchedulerConfig.scheduler_cls` (config/scheduler.py:117). Re-grep before coding; upstream moves fast.
- `--scheduler-cls tidal.engine.scheduler.TidalScheduler` must work with `--scheduling-policy priority` on the Mac CPU build.

---

### Task B1: Self-calibrating latency model

**Files:**
- Create: `src/tidal/engine/latency_model.py`
- Test: `tests/unit/test_latency_model.py` (runs in tidal venv; pure numpy)

**Interfaces (produces):**
```python
class StepLatencyModel:
    def __init__(self, window: int, r2_gate: float): ...
    def observe(self, p: int, c: int, step_ms: float) -> None      # ring buffer
    def ready(self) -> bool                                        # fit converged (R² ≥ gate)
    def predict_ms(self, p: int, c: int) -> float                  # k1·p + k2·p(p+c) + k4·(p+c) + k5
    def max_batch_tokens(self, p_on: int, c_total: int, budget_ms: float) -> int
        # largest x ≥ 0 with predict_ms(p_on + x, c_total) ≤ budget_ms (closed-form quadratic root)
    def mape(self) -> float                                        # rolling accuracy
```
Coefficients clamped ≥ 0; refit every 256 observations (cheap lstsq on 4 features).

- [ ] **Step 1: failing tests**

```python
def test_recovers_synthetic_coefficients():
    m = StepLatencyModel(window=4000, r2_gate=0.85)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        p, c = int(rng.integers(1, 2048)), int(rng.integers(0, 8192))
        t = 0.01*p + 1e-6*p*(p+c) + 2e-4*(p+c) + 5 + rng.normal(0, 0.1)
        m.observe(p, c, t)
    assert m.ready() and abs(m.predict_ms(512, 1024) - true_t(512, 1024)) / true_t(512, 1024) < 0.05

def test_not_ready_before_min_samples_or_bad_r2(): ...
def test_max_batch_tokens_inverse_of_predict():   # predict(p_on+x*) ≤ budget < predict(p_on+x*+64)
def test_max_batch_tokens_zero_when_online_alone_exceeds_budget(): ...
def test_mape_reflects_noise(): ...
```

- [ ] Steps 2–5: fail → implement → pass → `git commit -m "feat(engine): self-calibrating step-latency model"`

### Task B2: KV guardband tracker

**Files:**
- Create: `src/tidal/engine/guardband.py`
- Test: `tests/unit/test_guardband.py`

**Interfaces (produces):**
```python
class KvGuardband:
    def __init__(self, h_min: float, h_max: float, alpha: float = 0.3): ...
    def observe_online_demand(self, blocks_allocated: int, pool_blocks: int) -> None
    def h(self) -> float                       # clamped EWMA of demand bursts
    def admit_ok(self, kv_usage: float) -> bool        # kv_usage ≤ 1 − h
    def evict_needed(self, kv_usage: float) -> bool    # kv_usage > 1 − h/2
```

- [ ] Steps: failing tests (clamping, hysteresis gap `admit_ok` vs `evict_needed`, EWMA responds to bursts) → implement → pass → commit `feat(engine): KV guardband tracker`

### Task B3: TidalScheduler wrapper subclass

**Files:**
- Create: `src/tidal/engine/scheduler.py`
- Test: `tests/unit/test_tidal_scheduler.py` — **runs in the vLLM venv**: `vllm-experiments/.venv/bin/python -m pytest`. Build the scheduler via the upstream test helpers (read `vllm-experiments/vllm/tests/v1/core/` for `create_scheduler`-style fixtures and copy the minimal construction pattern into our conftest).

**Interfaces:**
- Consumes: B1 `StepLatencyModel`, B2 `KvGuardband`.
- Produces: `class TidalScheduler(vllm.v1.core.sched.scheduler.Scheduler)` with:
  - `schedule()` = pre-gate → `super().schedule()` → post-telemetry.
  - Pre-gate: classify waiting batch requests (priority ≥ 10); compute `X = max_batch_tokens(...)` if model ready else `cold_start_batch_frac · τ` (or `slack` when zero online requests resident → offline mode); hide excess batch waiting requests in `self._tidal_held` (removed from `self.waiting`, restored after super() returns — MUST restore even on exception); if `guardband.evict_needed(kv_usage)`: `_preempt_request(cheapest_victim)` where victim = running batch request minimizing `num_computed_tokens − len(prefix cached blocks)·block_size`.
  - Post: `self.stats = TidalStats(batch_tokens_scheduled, online_tokens, slack_unfilled, step_predictions...)`; feed `StepLatencyModel.observe` with previous step's (P, C) and measured inter-schedule time; log a compact line every 100 steps.
  - `classify(req) -> bool` staticmethod (is_batch) for testability.

- [ ] **Step 1: failing tests**

```python
def test_online_only_schedules_identically_to_stock():
    # same request set (all priority 0) → SchedulerOutput token map equals stock Scheduler's
def test_batch_held_when_model_caps():
    # model stub predicts over-budget → batch waiting requests are NOT in scheduled output,
    # and are back in self.waiting afterward (no loss)
def test_offline_mode_fills_budget_with_batch():
    # zero online → batch scheduled up to τ (chunked), slack_unfilled == 0 given enough items
def test_guardband_blocks_batch_admission_at_high_kv(): ...
def test_proactive_eviction_picks_cheapest_batch_victim(): ...
def test_exception_in_super_restores_held_requests(): ...
```

- [ ] Steps 2–5: fail → implement → pass in vLLM venv → `git commit -m "feat(engine): TidalScheduler wrapper subclass"`

### Task B4: Engine integration smoke (Mac CPU)

**Files:**
- Create: `tests/integration/test_e2e_engine.py` (`@pytest.mark.integration`)
- Modify: `tests/integration/conftest.py` — fixture param to launch vLLM with `--scheduler-cls tidal.engine.scheduler.TidalScheduler` (PYTHONPATH includes tidal/src; env `TIDAL_TBT_SLO_MS` etc. read by scheduler via `TidalConfig.from_env()`).

- [ ] Steps: test = A8's e2e against technique-B engine + assert engine log shows offline-mode fill and capping transitions; make pass; commit `test: e2e engine integration`

### Task B5: A/B eval harness

**Files:**
- Create: `src/tidal/eval/loadgen.py` (Poisson + diurnal online generator, direct-to-vLLM, latency recorder), `src/tidal/eval/harness.py` (runs conditions {online_only, technique_a, technique_b, offline_ceiling}, emits `results/<cond>.json`), `src/tidal/eval/plots.py` (comparison charts; follow the dataviz skill)
- Test: `tests/unit/test_loadgen.py` (deterministic seed → expected arrival stats), `tests/unit/test_harness_metrics.py` (percentile math)

- [ ] Steps: TDD metrics math → implement → run locally (small: 3-min conditions, 30-item batches) → commit `feat(eval): A/B harness and plots`

---
**Self-review done against scheduler spec §§5–7: budget ledger→B3 pre-gate (chunk-granularity deviation documented), T̂→B1, guardband/victim→B2/B3, gateway delta→A5 config flag, eval→B5. No placeholders. Signatures consistent.**
