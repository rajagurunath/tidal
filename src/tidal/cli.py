"""``tidal`` command line.

Two commands:

* ``tidal serve``  — the gateway: FastAPI (uvicorn) with the dispatcher loop
  running as a background task inside the app's lifespan, so one process owns
  both the API and the batch injection.
* ``tidal report`` — print the usage ledger.

Configuration comes from :class:`~tidal.config.TidalConfig` (environment
variables prefixed ``TIDAL_``); the flags below are thin overrides for the
handful of fields worth typing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

import typer

from tidal.api.app import create_app
from tidal.api.assembler import finalize_if_done
from tidal.config import TidalConfig
from tidal.metering.ledger import Meter
from tidal.metering.report import usage_report
from tidal.store.interfaces import Repository, make_repository

log = logging.getLogger("tidal.cli")

#: Shipped placeholder bearer token — fine on loopback, never on a public bind.
DEFAULT_API_KEY = "tidal-dev-key"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

app = typer.Typer(
    add_completion=False,
    help="Tidal — an OpenAI-compatible batch gateway that co-serves with online vLLM traffic.",
)

#: Placeholder list price (USD per million tokens) for the served model, used
#: only when the operator has not seeded a price. Batch work is billed at
#: ``cfg.batch_discount`` of it.
DEFAULT_INPUT_USD_PER_MTOK = 1.0
DEFAULT_OUTPUT_USD_PER_MTOK = 2.0

DsnOption = Annotated[str | None, typer.Option("--dsn", help="Override TIDAL_DSN.")]
BlobDirOption = Annotated[str | None, typer.Option("--blob-dir", help="Override TIDAL_BLOB_DIR.")]


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host", help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port.")] = None,
    dsn: DsnOption = None,
    blob_dir: BlobDirOption = None,
) -> None:
    """Run the batch gateway and its dispatcher loop."""
    import uvicorn

    # Imported here (not at module scope) so ``tidal report`` works even when
    # the dispatcher's optional dependencies are unavailable.
    from tidal.dispatcher.loop import Dispatcher
    from tidal.dispatcher.vllm_client import VllmClient

    cfg = _config(dsn=dsn, blob_dir=blob_dir, host=host, port=port)
    _warn_on_default_key(cfg)
    repo = make_repository(cfg.dsn, cfg.blob_dir, cfg.max_item_attempts)
    seed_prices(repo, cfg)

    meter = Meter(repo, cfg)

    def finalize(batch_id: str):
        """Dispatcher hook: assemble results once a batch has drained."""
        return finalize_if_done(repo, batch_id)

    dispatcher = Dispatcher(
        cfg, repo, VllmClient(cfg), on_success=meter.on_success, finalize=finalize
    )
    api = create_app(cfg, repo)
    attach_dispatcher(api, dispatcher)

    typer.echo(f"tidal gateway on http://{cfg.host}:{cfg.port} -> vLLM {cfg.vllm_base_url}")
    uvicorn.run(api, host=cfg.host, port=cfg.port)


@app.command()
def report(
    since_hours: Annotated[
        float | None,
        typer.Option("--since-hours", help="Only count usage from the last N hours."),
    ] = None,
    dsn: DsnOption = None,
    blob_dir: BlobDirOption = None,
) -> None:
    """Print token usage, batch cost, and savings against online list prices."""
    cfg = _config(dsn=dsn, blob_dir=blob_dir)
    repo = make_repository(cfg.dsn, cfg.blob_dir, cfg.max_item_attempts)
    since = datetime.now(UTC) - timedelta(hours=since_hours) if since_hours else None
    typer.echo(usage_report(repo, since))


def _warn_on_default_key(cfg: TidalConfig) -> None:
    """Shout if the shipped bearer token is about to be exposed off-box.

    ``cfg.api_key`` is the only thing between the internet and a queue that
    spends GPU time, and the default is printed in the README.
    """
    if cfg.api_key != DEFAULT_API_KEY or cfg.host in LOOPBACK_HOSTS:
        return
    message = (
        f"SECURITY: serving on {cfg.host}:{cfg.port} with the default API key "
        f"{DEFAULT_API_KEY!r}. Anyone who can reach this port can submit batch "
        "work and read every batch's results. Set TIDAL_API_KEY to a secret."
    )
    log.warning(message)
    typer.secho(f"WARNING  {message}", fg=typer.colors.RED, bold=True, err=True)


def seed_prices(repo: Repository, cfg: TidalConfig) -> None:
    """Give the served model a list price if it does not have one yet."""
    if cfg.served_model not in repo.price_table():
        repo.set_price(cfg.served_model, DEFAULT_INPUT_USD_PER_MTOK, DEFAULT_OUTPUT_USD_PER_MTOK)


def attach_dispatcher(api, dispatcher) -> None:
    """Run ``dispatcher.run()`` for the lifetime of the ASGI app.

    ``create_app`` is a contract surface owned by the API task, so the loop is
    bolted on here by replacing the router's lifespan context rather than by
    changing the app factory.
    """

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(dispatcher.run(), name="tidal-dispatcher")
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    api.router.lifespan_context = lifespan


def _config(
    *,
    dsn: str | None = None,
    blob_dir: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> TidalConfig:
    cfg = TidalConfig.from_env()
    for field, value in (("dsn", dsn), ("blob_dir", blob_dir), ("host", host), ("port", port)):
        if value is not None:
            setattr(cfg, field, value)
    return cfg


if __name__ == "__main__":  # pragma: no cover
    app()
