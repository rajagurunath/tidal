"""Tests for the OpenAI-wire /v1/files and /v1/batches routes (Task A3).

The acceptance bar is the official ``openai`` SDK: it is pointed at the ASGI
app through ``httpx.ASGITransport`` (no network) and must be able to drive the
whole batch lifecycle.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest
from fakes import FakeRepository
from openai import AsyncOpenAI

from tidal.api.app import create_app
from tidal.config import TidalConfig
from tidal.store.interfaces import BatchStatus, ItemState

CHAT = "/v1/chat/completions"
BASE = "http://tidal.test"


def jsonl(*lines: dict) -> bytes:
    return ("\n".join(json.dumps(line) for line in lines) + "\n").encode()


def chat_request(custom_id: str, content: str = "hello") -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": CHAT,
        "body": {"model": "m", "messages": [{"role": "user", "content": content}]},
    }


GOOD_INPUT = jsonl(chat_request("a"), chat_request("b", "hello there"))
BAD_INPUT = b'%s\n{"broken\n' % json.dumps(chat_request("a")).encode()


@pytest.fixture
def cfg() -> TidalConfig:
    return TidalConfig(api_key="test-key")


@pytest.fixture
def repo() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def app(cfg, repo):
    return create_app(cfg, repo)


@pytest.fixture
async def http(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE) as client:
        yield client


@pytest.fixture
def auth(cfg):
    return {"Authorization": f"Bearer {cfg.api_key}"}


@pytest.fixture
def sdk(http, cfg) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=cfg.api_key, base_url=f"{BASE}/v1", http_client=http, max_retries=0)


# -- wiring ----------------------------------------------------------------


def test_app_exposes_repo_and_cfg_for_the_dispatcher(app, cfg, repo):
    assert app.state.repo is repo
    assert app.state.cfg is cfg


async def test_healthz_needs_no_auth(http):
    resp = await http.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_missing_bearer_token_is_401_openai_error(http):
    resp = await http.get("/v1/batches")
    assert resp.status_code == 401
    assert resp.json()["error"]["message"]
    assert resp.json()["error"]["type"]


async def test_wrong_bearer_token_raises_authentication_error(http, cfg):
    bad = AsyncOpenAI(api_key="nope", base_url=f"{BASE}/v1", http_client=http, max_retries=0)
    with pytest.raises(openai.AuthenticationError):
        await bad.batches.list()


async def test_every_batch_route_requires_auth(http):
    for method, path in [
        ("POST", "/v1/files"),
        ("GET", "/v1/files/file-x"),
        ("GET", "/v1/files/file-x/content"),
        ("POST", "/v1/batches"),
        ("GET", "/v1/batches"),
        ("GET", "/v1/batches/batch-x"),
        ("POST", "/v1/batches/batch-x/cancel"),
    ]:
        resp = await http.request(method, path)
        assert resp.status_code == 401, f"{method} {path}"


# -- SDK lifecycle ---------------------------------------------------------


async def test_sdk_drives_upload_create_retrieve_cancel(sdk, repo):
    uploaded = await sdk.files.create(file=("in.jsonl", GOOD_INPUT), purpose="batch")
    assert uploaded.object == "file"
    assert uploaded.id.startswith("file-")
    assert uploaded.filename == "in.jsonl"
    assert uploaded.bytes == len(GOOD_INPUT)
    assert uploaded.purpose == "batch"
    assert uploaded.status == "processed"
    assert isinstance(uploaded.created_at, int)

    batch = await sdk.batches.create(
        input_file_id=uploaded.id,
        endpoint=CHAT,
        completion_window="24h",
        metadata={"job": "nightly"},
    )
    assert batch.object == "batch"
    assert batch.id.startswith("batch-")
    assert batch.status == "validating"
    assert batch.endpoint == CHAT
    assert batch.input_file_id == uploaded.id
    assert batch.completion_window == "24h"
    assert batch.metadata == {"job": "nightly"}
    assert batch.request_counts.total == 2
    assert batch.request_counts.completed == 0
    assert batch.request_counts.failed == 0
    assert isinstance(batch.created_at, int)
    assert batch.expires_at == batch.created_at + 24 * 3600
    assert batch.output_file_id is None
    assert batch.error_file_id is None

    # items landed with prefix groups and line numbers from the parser
    items = repo.all_items(batch.id)
    assert [i.custom_id for i in items] == ["a", "b"]
    assert [i.line_no for i in items] == [1, 2]
    assert [i.prefix_group for i in items] == [0, 1]

    fetched = await sdk.batches.retrieve(batch.id)
    assert fetched.id == batch.id
    assert fetched.status == "validating"
    assert fetched.in_progress_at is None  # nothing has been dispatched yet

    # Every item is still PENDING, so the cancel has nothing to wait for: it
    # lands in CANCELLED on the spot rather than parking in CANCELLING.
    cancelled = await sdk.batches.cancel(batch.id)
    assert cancelled.status == "cancelled"
    assert cancelled.cancelled_at is not None
    assert all(i.state is ItemState.CANCELLED for i in repo.all_items(batch.id))


async def test_sdk_reads_output_file_of_a_finished_batch(sdk, repo):
    source = repo.create_file("batch", "in.jsonl", GOOD_INPUT)
    batch = repo.create_batch(source.id, CHAT, 24, {}, [("a", 1, 0, {"model": "m"})])
    output = (
        json.dumps(
            {
                "id": "resp-1",
                "custom_id": "a",
                "response": {"status_code": 200, "request_id": "req-1", "body": {"ok": True}},
                "error": None,
            }
        )
        + "\n"
    ).encode()
    out_file = repo.create_file("batch_output", f"{batch.id}_output.jsonl", output)
    repo.set_batch_status(batch.id, BatchStatus.IN_PROGRESS)
    repo.finalize_batch(batch.id, BatchStatus.COMPLETED, out_file.id, None)

    fetched = await sdk.batches.retrieve(batch.id)
    assert fetched.status == "completed"
    assert fetched.output_file_id == out_file.id
    assert fetched.completed_at is not None
    assert fetched.in_progress_at is not None  # stamped when it started

    meta = await sdk.files.retrieve(out_file.id)
    assert meta.purpose == "batch_output"
    assert meta.bytes == len(output)

    content = await sdk.files.content(out_file.id)
    assert content.text == output.decode()
    assert json.loads(content.text)["custom_id"] == "a"


async def test_retrieve_reports_live_request_counts(sdk, repo):
    uploaded = await sdk.files.create(file=("in.jsonl", GOOD_INPUT), purpose="batch")
    batch = await sdk.batches.create(
        input_file_id=uploaded.id, endpoint=CHAT, completion_window="24h"
    )
    first, second = repo.all_items(batch.id)
    repo.record_item_success(first.id, {"ok": True}, 5, 7, "req-1")
    repo.record_item_failure(second.id, "boom", retryable=False)

    fetched = await sdk.batches.retrieve(batch.id)
    assert fetched.request_counts.total == 2
    assert fetched.request_counts.completed == 1
    assert fetched.request_counts.failed == 1


async def test_list_batches_paginates(sdk):
    uploaded = await sdk.files.create(file=("in.jsonl", GOOD_INPUT), purpose="batch")
    created = []
    for i in range(3):
        created.append(
            await sdk.batches.create(
                input_file_id=uploaded.id,
                endpoint=CHAT,
                completion_window="24h",
                metadata={"n": str(i)},
            )
        )

    page = await sdk.batches.list(limit=2)
    assert len(page.data) == 2
    assert page.has_more is True
    assert [b.object for b in page.data] == ["batch", "batch"]

    rest = await sdk.batches.list(limit=2, after=page.data[-1].id)
    assert len(rest.data) == 1
    assert rest.has_more is False
    assert {b.id for b in page.data} | {rest.data[0].id} == {b.id for b in created}


async def test_list_response_is_an_openai_list_object(http, auth):
    resp = await http.get("/v1/batches", headers=auth)
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == []
    assert body["has_more"] is False


# -- validation failures ---------------------------------------------------


async def test_invalid_jsonl_creates_a_failed_batch_with_an_error_file(sdk, repo):
    uploaded = await sdk.files.create(file=("bad.jsonl", BAD_INPUT), purpose="batch")

    batch = await sdk.batches.create(
        input_file_id=uploaded.id, endpoint=CHAT, completion_window="24h"
    )
    assert batch.status == "failed"
    assert batch.error_file_id is not None
    assert batch.request_counts.total == 0
    assert batch.errors is not None
    assert batch.errors.object == "list"
    assert batch.errors.data[0].line == 2
    assert batch.errors.data[0].code
    assert "json" in batch.errors.data[0].message.lower()

    # the error file is downloadable and line-oriented
    content = await sdk.files.content(batch.error_file_id)
    rows = [json.loads(line) for line in content.text.splitlines() if line.strip()]
    assert rows[0]["error"]["line"] == 2

    # ...and its purpose maps onto an OpenAI file purpose
    meta = await sdk.files.retrieve(batch.error_file_id)
    assert meta.purpose == "batch_output"

    # the failed batch is still retrievable with the same shape
    fetched = await sdk.batches.retrieve(batch.id)
    assert fetched.status == "failed"
    assert fetched.failed_at is not None
    assert repo.get_batch(batch.id).status is BatchStatus.FAILED


async def test_completion_window_only_accepts_24h(http, auth):
    upload = await http.post(
        "/v1/files",
        headers=auth,
        files={"file": ("in.jsonl", GOOD_INPUT)},
        data={"purpose": "batch"},
    )
    file_id = upload.json()["id"]
    resp = await http.post(
        "/v1/batches",
        headers=auth,
        json={"input_file_id": file_id, "endpoint": CHAT, "completion_window": "1h"},
    )
    assert resp.status_code == 400
    assert "completion_window" in resp.json()["error"]["message"]


async def test_unsupported_endpoint_is_rejected(sdk):
    uploaded = await sdk.files.create(file=("in.jsonl", GOOD_INPUT), purpose="batch")
    with pytest.raises(openai.BadRequestError):
        await sdk.batches.create(
            input_file_id=uploaded.id, endpoint="/v1/embeddings", completion_window="24h"
        )


async def test_unknown_input_file_is_404(sdk):
    with pytest.raises(openai.NotFoundError):
        await sdk.batches.create(
            input_file_id="file-missing", endpoint=CHAT, completion_window="24h"
        )


async def test_input_file_must_have_batch_purpose(http, auth):
    resp = await http.post(
        "/v1/files",
        headers=auth,
        files={"file": ("in.jsonl", GOOD_INPUT)},
        data={"purpose": "assistants"},
    )
    assert resp.status_code == 400
    assert "purpose" in resp.json()["error"]["message"]


async def test_unknown_batch_and_file_are_404(sdk):
    with pytest.raises(openai.NotFoundError):
        await sdk.batches.retrieve("batch-missing")
    with pytest.raises(openai.NotFoundError):
        await sdk.files.retrieve("file-missing")
    with pytest.raises(openai.NotFoundError):
        await sdk.files.content("file-missing")
    with pytest.raises(openai.NotFoundError):
        await sdk.batches.cancel("batch-missing")


async def test_cancel_with_no_open_items_finalizes_immediately(sdk, repo):
    """A cancel that has nothing in flight must not wedge the batch.

    ``cancel_batch`` only moves the batch to CANCELLING — the dispatcher's
    finalize hook normally closes it once the last in-flight item settles. With
    no open items there is no such event, so the route finalizes inline;
    otherwise the batch would sit in CANCELLING until something else happened
    to touch it.
    """
    source = repo.create_file("batch", "in.jsonl", GOOD_INPUT)
    batch = repo.create_batch(source.id, CHAT, 24, {}, [("a", 1, 0, {"model": "m"})])
    (item,) = repo.claim_pending_items(1)
    repo.record_item_success(item.id, {"choices": [{"message": {"content": "hi"}}]}, 3, 4, "req-1")

    cancelled = await sdk.batches.cancel(batch.id)

    assert cancelled.status == "cancelled"
    assert cancelled.cancelled_at is not None
    assert repo.get_batch(batch.id).status is BatchStatus.CANCELLED
    # The work that did finish before the cancel is still delivered.
    assert cancelled.output_file_id is not None
    lines = repo.read_file_content(cancelled.output_file_id).decode().splitlines()
    assert json.loads(lines[0])["custom_id"] == "a"


async def test_cancelling_a_terminal_batch_is_409(sdk, repo, http, auth):
    source = repo.create_file("batch", "in.jsonl", GOOD_INPUT)
    batch = repo.create_batch(source.id, CHAT, 24, {}, [("a", 1, 0, {"model": "m"})])
    repo.set_batch_status(batch.id, BatchStatus.IN_PROGRESS)
    repo.finalize_batch(batch.id, BatchStatus.COMPLETED, None, None)

    resp = await http.post(f"/v1/batches/{batch.id}/cancel", headers=auth)
    assert resp.status_code == 409
    assert resp.json()["error"]["message"]
    with pytest.raises(openai.ConflictError):
        await sdk.batches.cancel(batch.id)
