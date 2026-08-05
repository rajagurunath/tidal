"""Unit tests for result assembly and the CLI (Task A7).

The assembler is exercised against the real repository: the output/error
files it writes are read back through ``repo.read_file_content`` and parsed,
so the wire shape is checked end to end.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from tidal.api.assembler import finalize_if_done
from tidal.store.interfaces import BatchStatus, ItemState
from tidal.store.repo import SqlRepository

T0 = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)
LATER = T0 + timedelta(hours=25)


@pytest.fixture
def repo(tmp_path):
    """A fresh file-backed SQLite repository per test."""
    return SqlRepository(f"sqlite:///{tmp_path / 'tidal.db'}", str(tmp_path / "blobs"))


def _batch(repo, custom_ids=("a", "b"), *, now: datetime = T0):
    f = repo.create_file("batch", "in.jsonl", b"{}\n", now=now)
    return repo.create_batch(
        f.id,
        "/v1/chat/completions",
        24,
        {},
        [
            (custom_id, line_no, 0, {"model": "m", "messages": []})
            for line_no, custom_id in enumerate(custom_ids, start=1)
        ],
        now=now,
    )


def _claim(repo, n: int, now: datetime = T0):
    return repo.claim_pending_items(n, now=now)


def _body(custom_id: str) -> dict:
    return {"id": f"chatcmpl-{custom_id}", "choices": [{"message": {"content": custom_id}}]}


def _lines(repo, file_id: str) -> list[dict]:
    raw = repo.read_file_content(file_id).decode()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


# -- happy path ----------------------------------------------------------


def test_finalize_writes_output_file_and_completes(repo):
    batch = _batch(repo)
    repo.set_batch_status(batch.id, BatchStatus.IN_PROGRESS, now=T0)
    for item in _claim(repo, 2):
        repo.record_item_success(item.id, _body(item.custom_id), 10, 5, "req-1", now=T0)

    rec = finalize_if_done(repo, batch.id, now=T0)

    assert rec is not None
    assert rec.status is BatchStatus.COMPLETED
    assert rec.error_file_id is None
    lines = _lines(repo, rec.output_file_id)
    assert [line["custom_id"] for line in lines] == ["a", "b"]
    assert all(line["id"].startswith("batch_req_") for line in lines)
    assert len({line["id"] for line in lines}) == 2
    assert lines[0]["error"] is None
    assert lines[0]["response"]["status_code"] == 200
    assert lines[0]["response"]["request_id"] == "req-1"
    assert lines[0]["response"]["body"] == _body("a")
    assert repo.get_file(rec.output_file_id).purpose == "batch_output"


def test_finalize_completes_a_batch_that_never_left_validating(repo):
    batch = _batch(repo, ("a",))
    (item,) = _claim(repo, 1)
    repo.record_item_success(item.id, _body("a"), 1, 1, None, now=T0)

    rec = finalize_if_done(repo, batch.id, now=T0)

    assert rec.status is BatchStatus.COMPLETED
    # No vLLM request id recorded -> the item id stands in.
    assert _lines(repo, rec.output_file_id)[0]["response"]["request_id"] == item.id


# -- skipping ------------------------------------------------------------


def test_skips_while_items_are_outstanding(repo):
    batch = _batch(repo)
    (item,) = _claim(repo, 1)
    repo.record_item_success(item.id, _body("a"), 1, 1, None, now=T0)

    assert finalize_if_done(repo, batch.id, now=T0) is None
    assert repo.get_batch(batch.id).status is BatchStatus.VALIDATING


def test_is_idempotent(repo):
    batch = _batch(repo, ("a",))
    (item,) = _claim(repo, 1)
    repo.record_item_success(item.id, _body("a"), 1, 1, None, now=T0)

    first = finalize_if_done(repo, batch.id, now=T0)
    assert finalize_if_done(repo, batch.id, now=T0) is None
    assert repo.get_batch(batch.id).output_file_id == first.output_file_id


def test_unknown_batch_is_skipped(repo):
    assert finalize_if_done(repo, "batch-nope", now=T0) is None


# -- failures ------------------------------------------------------------


def test_failed_items_land_in_the_error_file(repo):
    batch = _batch(repo)
    repo.set_batch_status(batch.id, BatchStatus.IN_PROGRESS, now=T0)
    good, bad = _claim(repo, 2)
    repo.record_item_success(good.id, _body("a"), 1, 1, "req-1", now=T0)
    repo.record_item_failure(bad.id, "engine returned 500", retryable=False, now=T0)

    rec = finalize_if_done(repo, batch.id, now=T0)

    assert rec.status is BatchStatus.COMPLETED
    assert [line["custom_id"] for line in _lines(repo, rec.output_file_id)] == ["a"]
    (err,) = _lines(repo, rec.error_file_id)
    assert err["custom_id"] == "b"
    assert err["response"] is None
    assert err["error"]["message"] == "engine returned 500"
    assert err["id"].startswith("batch_req_")
    assert repo.get_file(rec.error_file_id).purpose == "batch_error"


def test_all_failed_batch_has_no_output_file(repo):
    batch = _batch(repo, ("a",))
    (item,) = _claim(repo, 1)
    repo.record_item_failure(item.id, "boom", retryable=False, now=T0)

    rec = finalize_if_done(repo, batch.id, now=T0)

    assert rec.status is BatchStatus.COMPLETED
    assert rec.output_file_id is None
    assert len(_lines(repo, rec.error_file_id)) == 1


# -- expiry and cancellation ---------------------------------------------


def test_expired_batch_keeps_partial_output_and_flags_expired_items(repo):
    batch = _batch(repo, ("a", "b", "c"))
    repo.set_batch_status(batch.id, BatchStatus.IN_PROGRESS, now=T0)
    done, *_ = _claim(repo, 1)
    repo.record_item_success(done.id, _body("a"), 1, 1, "req-1", now=T0)
    repo.expire_batch(batch.id, now=LATER)

    rec = finalize_if_done(repo, batch.id, now=LATER)

    assert rec.status is BatchStatus.EXPIRED
    assert [line["custom_id"] for line in _lines(repo, rec.output_file_id)] == ["a"]
    errors = _lines(repo, rec.error_file_id)
    assert [line["custom_id"] for line in errors] == ["b", "c"]
    assert {line["error"]["code"] for line in errors} == {"batch_expired"}
    # completed + failed < total: expired items count as neither.
    assert repo.batch_progress(batch.id) == (3, 1, 0)


def test_cancelling_batch_finalizes_as_cancelled(repo):
    batch = _batch(repo, ("a", "b"))
    repo.set_batch_status(batch.id, BatchStatus.IN_PROGRESS, now=T0)
    (done,) = _claim(repo, 1)
    repo.record_item_success(done.id, _body("a"), 1, 1, "req-1", now=T0)
    repo.cancel_batch(batch.id, now=T0)

    rec = finalize_if_done(repo, batch.id, now=T0)

    assert rec.status is BatchStatus.CANCELLED
    assert [line["custom_id"] for line in _lines(repo, rec.output_file_id)] == ["a"]
    # Cancelled items are not errors — they are simply absent from both files.
    assert rec.error_file_id is None


# -- repo.list_items (added for A7) --------------------------------------


def test_list_items_filters_by_state_and_orders_by_line(repo):
    batch = _batch(repo, ("a", "b", "c"))
    other = _batch(repo, ("x",), now=T0 + timedelta(hours=1))
    items = repo.list_items(batch.id)
    assert [i.custom_id for i in items] == ["a", "b", "c"]
    assert [i.custom_id for i in repo.list_items(other.id)] == ["x"]

    (claimed,) = _claim(repo, 1)
    repo.record_item_success(claimed.id, _body("a"), 1, 1, None, now=T0)
    succeeded = repo.list_items(batch.id, [ItemState.SUCCEEDED])
    assert [i.custom_id for i in succeeded] == ["a"]
    assert succeeded[0].result_json == _body("a")
    assert len(repo.list_items(batch.id, [ItemState.PENDING, ItemState.SUCCEEDED])) == 3
    assert repo.list_items("batch-nope") == []


# -- CLI -----------------------------------------------------------------


def test_cli_report_runs_against_a_real_repository(tmp_path):
    from tidal.cli import app

    dsn = f"sqlite:///{tmp_path / 'tidal.db'}"
    blobs = str(tmp_path / "blobs")
    repo = SqlRepository(dsn, blobs)
    repo.set_price("m", 1.0, 2.0)
    batch = _batch(repo, ("a",))
    (item,) = _claim(repo, 1)
    repo.record_item_success(item.id, _body("a"), 1_000_000, 0, None, now=T0)
    repo.record_usage(item.id, "m", 1_000_000, 0, 0.5)
    assert batch.counts_total == 1

    env = {"TIDAL_DSN": dsn, "TIDAL_BLOB_DIR": blobs}
    result = CliRunner().invoke(app, ["report", "--since-hours", "24"], env=env)

    assert result.exit_code == 0, result.output
    assert "m" in result.output
    assert "0.500000" in result.output  # billed at the batch discount
    assert "1.000000" in result.output  # would-be online price


def test_cli_exposes_serve_and_report():
    pytest.importorskip("tidal.dispatcher.loop")
    from tidal.cli import app

    names = {cmd.name or cmd.callback.__name__ for cmd in app.registered_commands}
    assert {"serve", "report"} <= names


def _run_serve(tmp_path, monkeypatch, **env_overrides):
    """Invoke ``tidal serve`` with uvicorn stubbed out, returning (result, spy)."""
    pytest.importorskip("tidal.dispatcher.loop")
    import uvicorn

    from tidal import cli as cli_module

    built: dict = {}
    real = cli_module.make_repository

    def spy(dsn: str, blob_dir: str, max_attempts: int = 3):
        built["max_attempts"] = max_attempts
        built["repo"] = real(dsn, blob_dir, max_attempts)
        return built["repo"]

    monkeypatch.setattr(cli_module, "make_repository", spy)
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)

    env = {
        "TIDAL_DSN": f"sqlite:///{tmp_path / 'tidal.db'}",
        "TIDAL_BLOB_DIR": str(tmp_path / "blobs"),
        **env_overrides,
    }
    return CliRunner().invoke(cli_module.app, ["serve"], env=env), built


def test_cli_serve_passes_max_item_attempts_to_the_repository(tmp_path, monkeypatch):
    """TIDAL_MAX_ITEM_ATTEMPTS was configurable but never reached the store."""
    result, built = _run_serve(tmp_path, monkeypatch, TIDAL_MAX_ITEM_ATTEMPTS="1")

    assert result.exit_code == 0, result.output
    assert built["max_attempts"] == 1
    assert built["repo"].max_attempts == 1


def test_cli_serve_warns_when_the_default_key_is_bound_publicly(tmp_path, monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="tidal.cli"):
        result, _built = _run_serve(tmp_path, monkeypatch, TIDAL_HOST="0.0.0.0")
    assert result.exit_code == 0, result.output
    assert "tidal-dev-key" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="tidal.cli"):
        result, _built = _run_serve(
            tmp_path, monkeypatch, TIDAL_HOST="0.0.0.0", TIDAL_API_KEY="a-real-secret"
        )
    assert result.exit_code == 0, result.output
    assert caplog.text == ""

    caplog.clear()
    with caplog.at_level("WARNING", logger="tidal.cli"):
        result, _built = _run_serve(tmp_path, monkeypatch)  # loopback default
    assert result.exit_code == 0, result.output
    assert caplog.text == ""
