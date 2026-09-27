"""Tests for correlation-ID response and logging integration (C9.2)."""

import asyncio

from app.middleware.correlation_id import (
    CORRELATION_ID_HEADER,
    CorrelationIdMiddleware,
    get_correlation_id,
)


def run_request(incoming_id=None):
    messages = []
    seen = {}

    async def inner(scope, receive, send):
        seen["state_id"] = scope["state"]["correlation_id"]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    headers = []
    if incoming_id is not None:
        headers.append((b"x-correlation-id", incoming_id.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/detections/one",
        "headers": headers,
    }
    asyncio.run(CorrelationIdMiddleware(inner)(scope, receive, send))

    start = next(item for item in messages if item["type"] == "http.response.start")
    response_headers = {
        name.decode().lower(): value.decode() for name, value in start["headers"]
    }
    return seen["state_id"], response_headers[CORRELATION_ID_HEADER.lower()]


def test_valid_incoming_id_is_reused_everywhere(caplog):
    with caplog.at_level("INFO", logger="echo.api"):
        state_id, response_id = run_request("client-trace-123")

    assert state_id == response_id == "client-trace-123"
    assert "client-trace-123" in caplog.text
    assert any(
        getattr(record, "correlation_id", None) == "client-trace-123"
        for record in caplog.records
    )


def test_missing_id_is_generated():
    state_id, response_id = run_request()
    assert state_id == response_id
    assert len(response_id) == 32


def test_unsafe_incoming_id_is_replaced():
    state_id, response_id = run_request("contains spaces")
    assert state_id == response_id
    assert response_id != "contains spaces"


def test_request_context_is_cleared_after_response():
    run_request("client-trace-123")
    assert get_correlation_id() is None
