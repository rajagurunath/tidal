"""Human-readable usage report over the ledger (Task A6).

``tidal report`` prints this. Each row shows what the batch traffic actually
cost, what the same tokens would have cost at the model's online list price,
and the difference — the number the whole project exists to make large.
"""

from __future__ import annotations

from datetime import datetime

from tidal.metering.ledger import price
from tidal.store.interfaces import Repository

__all__ = ["usage_report"]

_HEADERS = (
    "model",
    "items",
    "prompt_tok",
    "completion_tok",
    "billed_usd",
    "online_usd",
    "saved_usd",
)
#: Left-align the model name, right-align every numeric column.
_ALIGN = ("<", ">", ">", ">", ">", ">", ">")
_GUTTER = "  "


def usage_report(repo: Repository, since: datetime | None = None) -> str:
    """Render the usage ledger as an aligned fixed-width table.

    Args:
        repo: the store to read ``usage_summary`` and ``price_table`` from.
        since: only count ledger rows at or after this timestamp; ``None``
            reports everything.

    Returns:
        A multi-line string: header, rule, one row per model, then a TOTAL
        row. Every line is the same width, so it stays readable in a terminal
        and in a log.
    """
    rows = repo.usage_summary(since)
    if not rows:
        window = f" since {since.isoformat()}" if since is not None else ""
        return f"No usage recorded{window}."

    prices = repo.price_table()
    body: list[tuple[str, ...]] = []
    totals = {"items": 0, "prompt": 0, "completion": 0, "billed": 0.0, "online": 0.0}

    for row in rows:
        model = str(row["model"])
        prompt_tokens = int(row["prompt_tokens"])
        completion_tokens = int(row["completion_tokens"])
        billed = float(row["cost_usd"])
        # What the same tokens would have cost at the undiscounted list price.
        online = price(model, prompt_tokens, completion_tokens, prices, 1.0)
        body.append(
            _cells(model, int(row["items"]), prompt_tokens, completion_tokens, billed, online)
        )
        totals["items"] += int(row["items"])
        totals["prompt"] += prompt_tokens
        totals["completion"] += completion_tokens
        totals["billed"] += billed
        totals["online"] += online

    body.append(
        _cells(
            "TOTAL",
            int(totals["items"]),
            int(totals["prompt"]),
            int(totals["completion"]),
            float(totals["billed"]),
            float(totals["online"]),
        )
    )

    widths = [max(len(cell) for cell in column) for column in zip(_HEADERS, *body, strict=True)]
    lines = [_row(_HEADERS, widths), ""]
    lines.extend(_row(cells, widths) for cells in body)
    lines[1] = "-" * len(lines[0])
    return "\n".join(lines)


def _cells(
    model: str,
    items: int,
    prompt_tokens: int,
    completion_tokens: int,
    billed: float,
    online: float,
) -> tuple[str, ...]:
    return (
        model,
        f"{items:,}",
        f"{prompt_tokens:,}",
        f"{completion_tokens:,}",
        f"{billed:.6f}",
        f"{online:.6f}",
        f"{online - billed:.6f}",
    )


def _row(cells: tuple[str, ...], widths: list[int]) -> str:
    return _GUTTER.join(
        f"{cell:{align}{width}}" for cell, align, width in zip(cells, _ALIGN, widths, strict=True)
    )
