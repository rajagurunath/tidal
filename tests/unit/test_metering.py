"""Unit tests for metering (Task A6).

Everything runs against the real SQLAlchemy repository on a throwaway SQLite
file, so the ledger's FK constraints and the price table are exercised for
real. Time is always injected via ``now=``; no sleeps.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from tidal.config import TidalConfig
from tidal.metering.ledger import Meter, price
from tidal.metering.report import usage_report
from tidal.store.interfaces import ItemRecord, ItemState
from tidal.store.repo import SqlRepository

T0 = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)

#: (input, output) USD per million tokens.
TABLE = {"m": (1.0, 2.0)}


@pytest.fixture
def repo(tmp_path):
    """A fresh file-backed SQLite repository per test."""
    return SqlRepository(f"sqlite:///{tmp_path / 'tidal.db'}", str(tmp_path / "blobs"))


@pytest.fixture
def cfg():
    return TidalConfig(served_model="m", batch_discount=0.5)


def _succeeded_item(
    repo,
    *,
    body: dict | None = None,
    prompt_tokens: int = 1000,
    completion_tokens: int = 500,
    custom_id: str = "a",
    now: datetime = T0,
) -> ItemRecord:
    """Create a one-item batch, claim it and mark it SUCCEEDED."""
    f = repo.create_file("batch", "in.jsonl", b"{}\n", now=now)
    repo.create_batch(
        f.id,
        "/v1/chat/completions",
        24,
        {},
        [(custom_id, 1, 0, {"model": "m", "messages": []} if body is None else body)],
        now=now,
    )
    (item,) = repo.claim_pending_items(1, now=now)
    return repo.record_item_success(
        item.id, {"choices": []}, prompt_tokens, completion_tokens, "req-1", now=now
    )


# -- price ---------------------------------------------------------------


def test_price_halves_the_online_list_price():
    assert price("m", 1_000_000, 500_000, TABLE, 0.5) == 1.0
    assert price("m", 1_000_000, 500_000, TABLE, 1.0) == 2.0


def test_price_is_linear_in_tokens():
    assert price("m", 1234, 0, TABLE, 0.5) == 0.000617
    assert price("m", 0, 1234, TABLE, 0.5) == 0.001234


def test_price_rounds_to_micro_dollars():
    # 999_999 * 0.1 / 1e6 * 0.5 = 0.04999995 -> 0.05 at micro-dollar precision.
    assert price("m", 999_999, 0, {"m": (0.1, 0.0)}, 0.5) == 0.05
    # Sub-micro-dollar amounts round away rather than accumulating float noise.
    assert price("m", 1, 0, {"m": (0.3, 0.0)}, 0.5) == 0.0


def test_price_of_unknown_model_is_zero_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="tidal.metering.ledger"):
        assert price("nope", 1_000_000, 1_000_000, TABLE, 0.5) == 0.0
    assert "nope" in caplog.text


# -- Meter ---------------------------------------------------------------


def test_meter_records_discounted_usage(repo, cfg):
    repo.set_price("m", 1.0, 2.0)
    item = _succeeded_item(repo)

    Meter(repo, cfg).on_success(item)

    (row,) = repo.usage_summary(None)
    assert row == {
        "model": "m",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        # (1000 * 1.0 + 500 * 2.0) / 1e6 * 0.5
        "cost_usd": 0.001,
        "items": 1,
    }


def test_meter_reads_the_model_from_the_request_body(repo, cfg):
    repo.set_price("other-model", 10.0, 10.0)
    item = _succeeded_item(repo, body={"model": "other-model", "messages": []})

    Meter(repo, cfg).on_success(item)

    (row,) = repo.usage_summary(None)
    assert row["model"] == "other-model"
    assert row["cost_usd"] == 0.0075  # 1500 tokens * 10 / 1e6 * 0.5


def test_meter_falls_back_to_the_served_model(repo, cfg):
    repo.set_price("m", 1.0, 2.0)
    item = _succeeded_item(repo, body={"messages": []})

    Meter(repo, cfg).on_success(item)

    (row,) = repo.usage_summary(None)
    assert row["model"] == "m" and row["cost_usd"] == 0.001


def test_meter_picks_up_prices_added_after_the_first_call(repo, cfg):
    meter = Meter(repo, cfg)
    first = _succeeded_item(repo, custom_id="a")
    meter.on_success(first)  # no price table yet -> $0

    repo.set_price("m", 1.0, 2.0)
    second = _succeeded_item(repo, custom_id="b")
    meter.on_success(second)

    (row,) = repo.usage_summary(None)
    assert row["items"] == 2
    assert row["cost_usd"] == 0.001  # only the second item was priced


def test_meter_never_breaks_the_dispatcher(repo, cfg, caplog):
    ghost = ItemRecord(
        id="item-does-not-exist",
        batch_id="batch-nope",
        custom_id="a",
        line_no=1,
        prefix_group=0,
        request_json={"model": "m"},
        state=ItemState.SUCCEEDED,
        usage_prompt_tokens=10,
        usage_completion_tokens=10,
    )
    with caplog.at_level(logging.ERROR, logger="tidal.metering.ledger"):
        Meter(repo, cfg).on_success(ghost)  # must not raise
    assert repo.usage_summary(None) == []
    assert "item-does-not-exist" in caplog.text


# -- report --------------------------------------------------------------


def test_usage_report_aggregates_and_shows_savings(repo, cfg):
    repo.set_price("m", 1.0, 2.0)
    meter = Meter(repo, cfg)
    for custom_id in ("a", "b"):
        meter.on_success(_succeeded_item(repo, custom_id=custom_id))

    out = usage_report(repo, None)

    assert "m" in out
    assert "2" in out  # items
    assert "0.002000" in out  # batch cost, 2 x $0.001
    assert "0.004000" in out  # would-be online cost
    assert "TOTAL" in out


def test_usage_report_columns_are_aligned(repo, cfg):
    repo.set_price("m", 1.0, 2.0)
    repo.set_price("z-model", 3.0, 4.0)
    meter = Meter(repo, cfg)
    meter.on_success(_succeeded_item(repo, custom_id="a"))
    meter.on_success(
        _succeeded_item(repo, custom_id="b", body={"model": "z-model", "messages": []})
    )

    lines = usage_report(repo, None).splitlines()

    assert len({len(line) for line in lines}) == 1, lines
    # Rows are ordered by model, with the totals last.
    assert [line.split()[0] for line in lines[2:]][:3] == ["m", "z-model", "TOTAL"]


def test_usage_report_honours_since(repo, cfg):
    repo.set_price("m", 1.0, 2.0)
    item = _succeeded_item(repo)
    Meter(repo, cfg).on_success(item)
    repo.record_usage(item.id, "m", 9_000_000, 0, 4.5, now=T0 - timedelta(days=2))

    recent = usage_report(repo, datetime.now(UTC) - timedelta(hours=1))
    everything = usage_report(repo, None)

    assert "4.501000" in everything  # both ledger rows aggregated
    assert "4.501000" not in recent
    assert "0.001000" in recent  # only the row inside the window


def test_usage_report_is_readable_when_empty(repo):
    assert "no usage" in usage_report(repo, None).lower()
