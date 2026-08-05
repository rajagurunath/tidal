# Plan A — Tidal Gateway (external-control co-serving) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OpenAI-wire-compatible `/v1/files` + `/v1/batches` gateway that durably stores batch jobs and injects them into a stock vLLM (priority scheduling) governed by AIMD watermarks + laxity-driven SLA escalation.

**Architecture:** FastAPI app + SQLAlchemy repository (SQLite WAL, Postgres-ready) + async dispatcher loop polling vLLM `/metrics`. Online traffic never passes through the gateway. Spec: `vllm-experiments/docs/superpowers/specs/2026-08-05-tidal-batch-online-coserving-design.md`.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2 (sync ORM behind `asyncio.to_thread`), httpx, pytest + pytest-asyncio (asyncio_mode=auto), uv.

## Global Constraints

- All store access through `tidal.store.interfaces.Repository` protocol — components never import the ORM.
- State machines exactly as documented in `interfaces.py`; illegal transitions raise `IllegalTransition`.
- OpenAI wire compatibility: responses must be readable by the official `openai` Python SDK pointed at the gateway (`base_url`).
- Priorities: online=0 (implicit, set by online clients or absent), batch submissions carry `priority` from laxity mapping, capped at `batch_priority_min=1` unless `sla_strict`.
- Config only via `tidal.config.TidalConfig` (env-overridable). No other global state.
- Every commit passes `ruff check` and the unit test suite. Line length 100.
- Do not modify `src/tidal/store/interfaces.py` or `src/tidal/config.py` (contract files; propose changes instead).

---

### Task A1: Store — SQLAlchemy repository

**Files:**
- Create: `src/tidal/store/repo.py` (models + `SqlRepository`)
- Test: `tests/unit/test_store.py`

**Interfaces:**
- Produces: `SqlRepository` implementing every method of `Repository` (see `interfaces.py` for exact signatures — they are the source of truth).
- SQLite DSNs get `PRAGMA journal_mode=WAL` on connect; blob bytes live under `blob_dir/<file_id>.jsonl`.

- [ ] **Step 1: failing tests** — cover at minimum:

```python
def test_create_batch_with_items_roundtrip(repo):
    f = repo.create_file("batch", "in.jsonl", b'{"custom_id":"a"}\n')
    b = repo.create_batch(f.id, "/v1/chat/completions", 24, {"k": "v"},
                          items=[("a", 1, 0, {"model": "m", "messages": []})])
    assert b.status is BatchStatus.VALIDATING and b.counts_total == 1
    assert repo.get_batch(b.id).metadata == {"k": "v"}

def test_claim_orders_by_edf_then_prefix_then_line(repo):
    # two batches: b_late expires later but created first; b_soon expires sooner
    ... create b_late (window 24h) with items prefix_groups [1,0]
    ... create b_soon (window 1h) with one item
    claimed = repo.claim_pending_items(limit=3)
    assert [i.batch_id for i in claimed][0] == b_soon.id          # EDF first
    assert [i.prefix_group for i in claimed[1:]] == [0, 1]        # then prefix
    assert all(i.state is ItemState.INFLIGHT for i in claimed)    # atomic claim

def test_retryable_failure_requeues_until_max_attempts(repo): ...
    # record_item_failure(retryable=True) x2 → PENDING each time, attempts increments
    # 3rd (== max_item_attempts) → FAILED

def test_requeue_inflight_recovers_crash(repo): ...
def test_expire_batch_marks_remaining_and_batch(repo): ...
def test_cancel_flow_and_illegal_transition_raises(repo):
    ...
    with pytest.raises(IllegalTransition):
        repo.set_batch_status(completed_batch.id, BatchStatus.IN_PROGRESS)

def test_progress_counts_live(repo): ...
```

- [ ] **Step 2:** run → all FAIL (no `repo.py`)
- [ ] **Step 3:** implement `repo.py`: ORM models (`files`, `batches`, `batch_items`, `usage_ledger`, `price_table`), transition guard table, atomic claim via `UPDATE ... WHERE state='pending'` inside one transaction.
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** `git commit -m "feat(store): SQLAlchemy repository with OpenAI batch state machines"`

### Task A2: JSONL validation + prefix grouping

**Files:**
- Create: `src/tidal/api/jsonl.py`
- Test: `tests/unit/test_jsonl.py`

**Interfaces:**
- Produces: `parse_batch_input(content: bytes, endpoint: str) -> list[ParsedLine]` where `ParsedLine = (custom_id: str, line_no: int, prefix_group: int, body: dict)`; raises `BatchInputError(line_no, reason)` on: bad JSON, missing/duplicate `custom_id`, `method != "POST"`, `url != endpoint`, `body.stream == True`, `body.n > 1`, missing `model`/`messages` (chat) or `prompt` (completions).
- `prefix_group`: stable bucket = index of the line's 64-token-ish prefix (first 256 chars of the serialized first message) in first-seen order — lines sharing a prefix share a group.

- [ ] Steps: failing tests (valid file, each rejection reason with line number, prefix grouping of shared-prefix lines, 10k-line file under 1s) → implement → pass → `git commit -m "feat(api): batch input JSONL validation and prefix grouping"`

### Task A3: Batch API routes

**Files:**
- Create: `src/tidal/api/schemas.py` (pydantic wire models), `src/tidal/api/app.py` (`create_app(cfg, repo) -> FastAPI`)
- Test: `tests/unit/test_api.py` (httpx `ASGITransport`; no network)

**Interfaces:**
- Consumes: `Repository`, `parse_batch_input`.
- Produces: routes `POST /v1/files` (multipart), `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, `POST /v1/batches`, `GET /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel`. Bearer auth on all routes (401 otherwise). Wire shapes per OpenAI: batch object fields `id, object="batch", endpoint, errors, input_file_id, completion_window, status, output_file_id, error_file_id, created_at (unix int), in_progress_at, expires_at, request_counts{total,completed,failed}, metadata`.
- Produces: `app.state.repo`, `app.state.cfg` for the dispatcher wiring in A5.

- [ ] Steps: failing tests — upload→create→retrieve happy path via **openai SDK against ASGI transport** (`OpenAI(base_url=..., http_client=...)`), invalid JSONL → batch `failed` with per-line error file, cancel semantics, auth 401, list pagination — → implement → pass → `git commit -m "feat(api): OpenAI-wire /v1/files and /v1/batches"`

### Task A4: Laxity escalation + AIMD controller (pure logic)

**Files:**
- Create: `src/tidal/dispatcher/laxity.py`, `src/tidal/dispatcher/aimd.py`
- Test: `tests/unit/test_laxity.py`, `tests/unit/test_aimd.py`

**Interfaces:**
- Produces (laxity): `laxity_seconds(expires_at, now, remaining_items, rate_items_per_s) -> float`; `urgency(laxity_s, window_s) -> float` (clamped 0..1); `priority_for(urgency, cfg) -> int` (linear 100→1, `sla_strict` allows 0 at urgency 1.0); `ObservedRate` (windowed completions/sec with ε floor).
- Produces (aimd): `AimdController(cfg)` with `.update(kv_usage: float, waiting: int, inflight: int) -> int` returning new target; halves on `waiting - inflight > tolerance` or `kv > kv_high`; +1 when `kv < kv_low`; respects `[floor, max_inflight]`; `.set_floor(n)`.

- [ ] Steps: failing tests —

```python
def test_on_track_batch_stays_at_100():   # laxity >> 0 → urgency 0 → priority 100
def test_at_risk_batch_escalates_early(): # huge backlog, low rate → urgency ~1 at hour 2
def test_priority_capped_at_1_unless_strict():
def test_urgency_monotone_in_time_and_backlog():
def test_aimd_halves_on_online_queueing_and_grows_on_low_kv():
def test_aimd_never_below_floor():
```

→ implement → pass → `git commit -m "feat(dispatcher): laxity escalation and AIMD controller"`

### Task A5: Dispatcher loop + vLLM client + metrics poller

**Files:**
- Create: `src/tidal/dispatcher/vllm_client.py` (`VllmClient.chat(body, priority) -> (result_json, prompt_toks, completion_toks)`, `.scrape() -> EngineMetrics(running, waiting, kv_usage)`, both httpx-async; `EngineDown` on connect failure)
- Create: `src/tidal/dispatcher/loop.py` (`Dispatcher(cfg, repo, client)` with `.tick()` (one iteration, testable) and `.run()` (forever); startup calls `repo.requeue_inflight()`)
- Test: `tests/unit/test_dispatcher.py` with `FakeRepo`/`FakeClient`

**Interfaces:**
- Consumes: A1 `Repository`, A4 controllers.
- Produces: tick algorithm = scrape→EWMA→AIMD target→laxity floor→claim `target − len(inflight)` items (EDF×prefix order from repo)→submit each with laxity priority→on completion record success/failure + metering hook (A6)→expiry sweep (`expire_batch` past `expires_at`)→finalize batches whose items are all terminal (A7 assembler).

- [ ] Steps: failing tests — tick fills to target with correct priorities; halves under fake load spike; escalated batch floor keeps ≥1 inflight during sustained "high load" fakes; `EngineDown` → circuit-open, inflight requeued; expiry sweep finalizes with partial output — → implement → pass → `git commit -m "feat(dispatcher): load-aware dispatch loop"`

### Task A6: Metering

**Files:**
- Create: `src/tidal/metering/ledger.py` (`price(model, ptoks, ctoks, cfg, table) -> float` at `batch_discount`; `Meter.on_success(item)` → `repo.record_usage`), `src/tidal/metering/report.py` (`usage_report(repo, since) -> str` table)
- Test: `tests/unit/test_metering.py`

- [ ] Steps: failing tests (0.5× math incl. rounding, report aggregation) → implement → pass → commit `feat(metering)`

### Task A7: Result assembly + CLI

**Files:**
- Create: `src/tidal/api/assembler.py` (`finalize_if_done(repo, batch_id) -> BatchRecord | None` — when no PENDING/INFLIGHT remain: build output JSONL `{"id","custom_id","response":{"status_code":200,"request_id","body"},"error":null}` + error file lines `{"custom_id","error":{...}}`, `repo.finalize_batch`)
- Create: `src/tidal/cli.py` (typer: `tidal serve` = uvicorn app + dispatcher task; `tidal report`)
- Test: `tests/unit/test_assembler.py` (output parses; counts match; expired batch gets partial output + error entries for expired items)

- [ ] Steps: TDD as above → commit `feat(api): result assembly and CLI`

### Task A8: Integration test (live vLLM, Mac CPU)

**Files:**
- Create: `tests/integration/test_e2e_gateway.py` (`@pytest.mark.integration`), `tests/integration/conftest.py` (fixture: launch `vllm serve Qwen/Qwen2.5-0.5B-Instruct --scheduling-policy priority --enforce-eager` with `TORCHDYNAMO_DISABLE=1`, cwd=/tmp; skip if `TIDAL_IT=0`)

- [ ] Steps: test — submit 20-item batch via openai SDK; Poisson online load (5 rps of tiny chats direct to vLLM); assert batch completes, output valid, online p99 latency < 2× no-batch baseline (CPU tolerance) — → make pass → commit `test: e2e gateway integration`

---
**Self-review done against spec §§5–10: all sections map to tasks A1–A8; no placeholders; signatures consistent with interfaces.py.**
