"""The Tidal gateway: OpenAI-wire ``/v1/files`` and ``/v1/batches``.

Online traffic never comes through here — this app only accepts batch work,
stores it durably, and exposes it for the dispatcher (which reads
``app.state.repo`` / ``app.state.cfg``).

Every repository call is synchronous by contract, so it is pushed onto a
thread with ``asyncio.to_thread`` to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from tidal.api.jsonl import BatchInputError, parse_batch_input
from tidal.api.schemas import (
    COMPLETION_WINDOW,
    SUPPORTED_ENDPOINTS,
    BatchCreateRequest,
    BatchError,
    BatchErrors,
    BatchListResponse,
    BatchObject,
    ErrorBody,
    ErrorResponse,
    FileObject,
)
from tidal.config import TidalConfig
from tidal.store.interfaces import BatchStatus, IllegalTransition, Repository

INPUT_PURPOSE = "batch"
ERROR_PURPOSE = "batch_error"


class ApiError(Exception):
    """An error rendered in OpenAI's ``{"error": {...}}`` envelope."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = ErrorBody(message=message, type=type, param=param, code=code)
        super().__init__(message)


def _error_response(status_code: int, body: ErrorBody) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=body).model_dump(),
    )


def create_app(cfg: TidalConfig, repo: Repository) -> FastAPI:
    app = FastAPI(title="Tidal Batch API", version="0.1.0")
    app.state.cfg = cfg
    app.state.repo = repo

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.body)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(_v1_router(cfg, repo))
    return app


def _v1_router(cfg: TidalConfig, repo: Repository) -> APIRouter:
    async def require_auth(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        token = ""
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token != cfg.api_key:
            raise ApiError(
                401,
                "Incorrect API key provided. Set a valid bearer token for the Tidal gateway.",
                type="invalid_request_error",
                code="invalid_api_key",
            )

    router = APIRouter(prefix="/v1", dependencies=[Depends(require_auth)])

    async def call(fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def load_file(file_id: str):
        rec = await call(repo.get_file, file_id)
        if rec is None:
            raise ApiError(404, f"No such file: {file_id}", param="file_id", code="not_found")
        return rec

    async def load_batch(batch_id: str):
        rec = await call(repo.get_batch, batch_id)
        if rec is None:
            raise ApiError(404, f"No such batch: {batch_id}", param="batch_id", code="not_found")
        return rec

    async def render_batch(rec) -> BatchObject:
        counts = await call(repo.batch_progress, rec.id)
        errors = None
        if rec.status is BatchStatus.FAILED and rec.error_file_id:
            errors = await _read_errors(repo, call, rec.error_file_id)
        return BatchObject.from_record(rec, counts=counts, errors=errors)

    # -- files -------------------------------------------------------------

    @router.post("/files", response_model=FileObject)
    async def upload_file(
        file: Annotated[UploadFile, File()],
        purpose: Annotated[str, Form()] = INPUT_PURPOSE,
    ) -> FileObject:
        if purpose != INPUT_PURPOSE:
            raise ApiError(
                400,
                f"Unsupported purpose {purpose!r}: the Tidal gateway only accepts 'batch'.",
                param="purpose",
            )
        content = await file.read()
        rec = await call(repo.create_file, purpose, file.filename or "input.jsonl", content)
        return FileObject.from_record(rec)

    @router.get("/files/{file_id}", response_model=FileObject)
    async def retrieve_file(file_id: str) -> FileObject:
        return FileObject.from_record(await load_file(file_id))

    @router.get("/files/{file_id}/content")
    async def download_file(file_id: str) -> Response:
        rec = await load_file(file_id)
        content = await call(repo.read_file_content, file_id)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{rec.filename}"'},
        )

    # -- batches -----------------------------------------------------------

    @router.post("/batches", response_model=BatchObject)
    async def create_batch(body: BatchCreateRequest) -> BatchObject:
        if body.completion_window != COMPLETION_WINDOW:
            raise ApiError(
                400,
                f"completion_window must be '{COMPLETION_WINDOW}', "
                f"got {body.completion_window!r}.",
                param="completion_window",
            )
        if body.endpoint not in SUPPORTED_ENDPOINTS:
            raise ApiError(
                400,
                f"Unsupported endpoint {body.endpoint!r}. "
                f"Supported: {', '.join(SUPPORTED_ENDPOINTS)}.",
                param="endpoint",
            )
        source = await load_file(body.input_file_id)
        if source.purpose != INPUT_PURPOSE:
            raise ApiError(
                400,
                f"File {source.id} has purpose {source.purpose!r}, expected 'batch'.",
                param="input_file_id",
            )

        content = await call(repo.read_file_content, source.id)
        metadata = body.metadata or {}
        try:
            parsed = parse_batch_input(content, body.endpoint)
        except BatchInputError as exc:
            rec = await _fail_validation(repo, call, body, metadata, exc, cfg)
            return await render_batch(rec)

        items = [tuple(line) for line in parsed]
        rec = await call(
            repo.create_batch,
            body.input_file_id,
            body.endpoint,
            cfg.completion_window_hours,
            metadata,
            items,
        )
        return await render_batch(rec)

    @router.get("/batches", response_model=BatchListResponse)
    async def list_batches(limit: int = 20, after: str | None = None) -> BatchListResponse:
        limit = max(1, min(limit, 100))
        records = await call(repo.list_batches, limit + 1, after)
        has_more = len(records) > limit
        records = records[:limit]
        data = [await render_batch(rec) for rec in records]
        return BatchListResponse(
            data=data,
            first_id=data[0].id if data else None,
            last_id=data[-1].id if data else None,
            has_more=has_more,
        )

    @router.get("/batches/{batch_id}", response_model=BatchObject)
    async def retrieve_batch(batch_id: str) -> BatchObject:
        return await render_batch(await load_batch(batch_id))

    @router.post("/batches/{batch_id}/cancel", response_model=BatchObject)
    async def cancel_batch(batch_id: str) -> BatchObject:
        rec = await load_batch(batch_id)
        try:
            rec = await call(repo.cancel_batch, batch_id)
        except IllegalTransition as exc:
            raise ApiError(
                409,
                f"Cannot cancel batch {batch_id} in status {rec.status.value}: {exc}",
                param="batch_id",
                code="invalid_state",
            ) from exc
        return await render_batch(rec)

    return router


async def _fail_validation(
    repo: Repository,
    call: Any,
    body: BatchCreateRequest,
    metadata: dict[str, str],
    exc: BatchInputError,
    cfg: TidalConfig,
):
    """OpenAI semantics: a bad input file yields a batch that exists and is
    immediately ``failed``, with the per-line reasons in an error file."""
    line = {
        "custom_id": None,
        "error": {
            "code": "invalid_request",
            "message": exc.reason,
            "param": None,
            "line": exc.line_no or None,
        },
    }
    error_file = await call(
        repo.create_file,
        ERROR_PURPOSE,
        f"{body.input_file_id}_errors.jsonl",
        (json.dumps(line) + "\n").encode(),
    )
    rec = await call(
        repo.create_batch,
        body.input_file_id,
        body.endpoint,
        cfg.completion_window_hours,
        metadata,
        [],
    )
    return await call(repo.finalize_batch, rec.id, BatchStatus.FAILED, None, error_file.id)


async def _read_errors(repo: Repository, call: Any, error_file_id: str) -> BatchErrors | None:
    try:
        raw = await call(repo.read_file_content, error_file_id)
    except Exception:
        return None
    errors: list[BatchError] = []
    for raw_line in raw.decode("utf-8", "replace").splitlines():
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        err = payload.get("error") or {}
        errors.append(
            BatchError(
                code=err.get("code") or "invalid_request",
                message=err.get("message") or "",
                param=err.get("param"),
                line=err.get("line"),
            )
        )
    return BatchErrors(data=errors) if errors else None
