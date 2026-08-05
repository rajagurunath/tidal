"""Usage metering: batch-discounted pricing and the ledger writer (Task A6).

Tidal's whole pitch is that batch work rides on idle capacity, so it bills at
``TidalConfig.batch_discount`` (0.5 by default) of the model's online list
price. The list prices live in the store's price table — the gateway never
hard-codes a price for a model, and a model nobody has priced simply costs
nothing rather than taking the dispatcher down.
"""

from __future__ import annotations

import logging

from tidal.config import TidalConfig
from tidal.store.interfaces import ItemRecord, Repository

__all__ = ["Meter", "price"]

log = logging.getLogger(__name__)

#: Costs are stored to the micro-dollar; anything smaller is float noise.
PRICE_DECIMALS = 6

_TOKENS_PER_PRICE_UNIT = 1_000_000.0


def price(
    model: str,
    prompt_toks: int,
    completion_toks: int,
    table: dict[str, tuple[float, float]],
    discount: float,
) -> float:
    """Return the USD cost of one request, discounted.

    Args:
        model: model name as it appears in the request body.
        prompt_toks: prompt (input) tokens consumed.
        completion_toks: completion (output) tokens generated.
        table: ``{model: (input_usd_per_mtok, output_usd_per_mtok)}`` — the
            *online* list prices, i.e. ``repo.price_table()``.
        discount: multiplier applied to the list price; pass
            ``cfg.batch_discount`` for batch traffic and ``1.0`` to compute the
            would-be online cost.

    An unknown model is charged $0.00 and logged: metering must never be the
    reason a batch fails.
    """
    entry = table.get(model)
    if entry is None:
        log.warning(
            "no list price for model %r; charging $0.00 — seed the price table "
            "(repo.set_price) to meter this model",
            model,
        )
        return 0.0
    input_per_mtok, output_per_mtok = entry
    raw = (prompt_toks * input_per_mtok + completion_toks * output_per_mtok) / (
        _TOKENS_PER_PRICE_UNIT
    )
    return round(raw * discount, PRICE_DECIMALS)


class Meter:
    """Writes one priced ledger row per successfully completed item.

    The dispatcher calls :meth:`on_success` from its completion path, so this
    class is deliberately defensive: any store failure is logged and
    swallowed rather than propagated into the dispatch loop.
    """

    def __init__(self, repo: Repository, cfg: TidalConfig) -> None:
        self._repo = repo
        self._cfg = cfg
        self._table: dict[str, tuple[float, float]] | None = None

    def price_table(self, *, refresh: bool = False) -> dict[str, tuple[float, float]]:
        """Return the cached price table, re-reading it when asked.

        The table changes rarely (an operator seeding prices), so it is cached
        and only refreshed when a model is missing from it.
        """
        if self._table is None or refresh:
            self._table = dict(self._repo.price_table())
        return self._table

    def on_success(self, item: ItemRecord) -> None:
        """Record ``item``'s token usage and its discounted cost."""
        model = self._model_of(item)
        prompt_tokens = int(item.usage_prompt_tokens or 0)
        completion_tokens = int(item.usage_completion_tokens or 0)

        table = self.price_table()
        if model not in table:
            # A price may have been seeded since this Meter last looked.
            table = self.price_table(refresh=True)
        cost = price(model, prompt_tokens, completion_tokens, table, self._cfg.batch_discount)

        try:
            self._repo.record_usage(item.id, model, prompt_tokens, completion_tokens, cost)
        except Exception:
            log.exception("failed to record usage for item %s (model %s)", item.id, model)

    def _model_of(self, item: ItemRecord) -> str:
        body = item.request_json or {}
        return str(body.get("model") or self._cfg.served_model)
