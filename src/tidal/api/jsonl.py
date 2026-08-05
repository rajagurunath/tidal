"""Validation and prefix grouping for OpenAI batch-input JSONL.

Each line of a batch input file is an envelope::

    {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "...", "messages": [...]}}

``parse_batch_input`` validates the whole file up front (OpenAI fails the
batch, not the individual line, on a malformed input file) and assigns every
line a ``prefix_group``: lines whose leading prompt text matches share a
bucket, so the dispatcher can submit prefix-affine work back to back and let
vLLM's prefix cache do the rest.
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

CHAT_ENDPOINT = "/v1/chat/completions"
COMPLETIONS_ENDPOINT = "/v1/completions"
SUPPORTED_ENDPOINTS = (CHAT_ENDPOINT, COMPLETIONS_ENDPOINT)

#: How much of the first message is hashed into a prefix bucket. 256 chars is
#: roughly the 64-token shared-system-prompt scale we care about.
PREFIX_CHARS = 256


class BatchInputError(Exception):
    """A batch input file rejected at line ``line_no`` (0 = whole file)."""

    def __init__(self, line_no: int, reason: str) -> None:
        self.line_no = line_no
        self.reason = reason
        super().__init__(f"line {line_no}: {reason}" if line_no else reason)


class ParsedLine(NamedTuple):
    """One validated request. Tuple-compatible with ``Repository.create_batch``
    items, which take ``(custom_id, line_no, prefix_group, request_json)``."""

    custom_id: str
    line_no: int
    prefix_group: int
    body: dict


def parse_batch_input(content: bytes, endpoint: str) -> list[ParsedLine]:
    """Validate ``content`` against ``endpoint`` and assign prefix groups.

    Raises ``BatchInputError`` on the first offending line.
    """
    if endpoint not in SUPPORTED_ENDPOINTS:
        raise BatchInputError(0, f"unsupported endpoint {endpoint!r}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BatchInputError(0, "input file is not valid UTF-8") from exc

    parsed: list[ParsedLine] = []
    seen_ids: set[str] = set()
    groups: dict[str, int] = {}

    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        envelope = _load_json(raw, line_no)
        custom_id = _validate_envelope(envelope, line_no, endpoint, seen_ids)
        body = envelope["body"]
        _validate_body(body, line_no, endpoint)
        key = _prefix_key(body, endpoint)
        group = groups.setdefault(key, len(groups))
        parsed.append(ParsedLine(custom_id, line_no, group, body))

    if not parsed:
        raise BatchInputError(0, "input file contains no requests")
    return parsed


def _load_json(raw: str, line_no: int) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BatchInputError(line_no, f"invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise BatchInputError(line_no, "each line must be a JSON object")
    return obj


def _validate_envelope(
    envelope: dict, line_no: int, endpoint: str, seen_ids: set[str]
) -> str:
    custom_id = envelope.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        raise BatchInputError(line_no, "missing or empty custom_id")
    if custom_id in seen_ids:
        raise BatchInputError(line_no, f"duplicate custom_id {custom_id!r}")
    seen_ids.add(custom_id)

    method = envelope.get("method")
    if not isinstance(method, str) or method.upper() != "POST":
        raise BatchInputError(line_no, f"method must be 'POST', got {method!r}")

    url = envelope.get("url")
    if url != endpoint:
        raise BatchInputError(line_no, f"url must be {endpoint!r}, got {url!r}")

    if not isinstance(envelope.get("body"), dict):
        raise BatchInputError(line_no, "body must be a JSON object")
    return custom_id


def _validate_body(body: dict, line_no: int, endpoint: str) -> None:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise BatchInputError(line_no, "body.model is required")

    if endpoint == CHAT_ENDPOINT:
        if not isinstance(body.get("messages"), list):
            raise BatchInputError(line_no, "body.messages is required for chat requests")
    elif body.get("prompt") is None:
        raise BatchInputError(line_no, "body.prompt is required for completion requests")

    if body.get("stream"):
        raise BatchInputError(line_no, "stream is not supported in batch requests")

    n = body.get("n")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n > 1):
        raise BatchInputError(line_no, f"n must be 1 for batch requests, got {n!r}")


def _prefix_key(body: dict, endpoint: str) -> str:
    if endpoint == COMPLETIONS_ENDPOINT:
        return _serialize(body.get("prompt"))[:PREFIX_CHARS]
    messages = body.get("messages") or []
    first = messages[0] if messages else None
    content = first.get("content") if isinstance(first, dict) else first
    return _serialize(content)[:PREFIX_CHARS]


def _serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
