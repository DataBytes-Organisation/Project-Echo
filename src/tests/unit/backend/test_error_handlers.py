"""Tests for the global exception handlers.

Task C4.3.

The handlers are async functions, so each test builds a minimal Starlette
Request and runs the handler with asyncio.run. Doing it this way rather than
through fastapi.testclient keeps the test suite free of extra dependencies -
TestClient requires httpx, which is not currently in the backend
requirements.
"""

import asyncio
import json

import pytest
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import error_handlers
from app.exceptions import (
    DetectionError,
    DetectionNotFoundError,
    DetectionRuleError,
    DetectionStorageError,
)
from app.middleware.correlation_id import CORRELATION_ID_HEADER


def make_request(method="GET", path="/detections", correlation_id=None, headers=None):
    """Build the smallest Request object the handlers need."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (name.lower().encode(), value.encode())
            for name, value in (headers or {}).items()
        ],
        "scheme": "http",
        "server": ("testserver", 80),
        "root_path": "",
    }
    if correlation_id:
        scope["state"] = {"correlation_id": correlation_id}
    return Request(scope)


def body_of(response):
    """Return the JSON body of a JSONResponse as a dict."""
    return json.loads(response.body.decode())


# ---------------------------------------------------------------------------
# Unhandled exceptions
# ---------------------------------------------------------------------------


def test_unhandled_exception_returns_500():
    response = asyncio.run(
        error_handlers.unhandled_exception_handler(make_request(), ValueError("boom"))
    )
    assert response.status_code == 500


def test_unhandled_exception_does_not_leak_the_original_message():
    """The whole point of the handler: internal detail must not go to the client."""
    secret = "mongodb://root:root_password@ts-mongodb-cont/UserSample"
    response = asyncio.run(
        error_handlers.unhandled_exception_handler(
            make_request(), RuntimeError("connection failed for " + secret)
        )
    )
    raw = response.body.decode()
    assert secret not in raw
    assert "root_password" not in raw
    assert "RuntimeError" not in raw


def test_unhandled_exception_returns_the_generic_message():
    response = asyncio.run(
        error_handlers.unhandled_exception_handler(make_request(), ValueError("boom"))
    )
    assert body_of(response)["error"]["message"] == error_handlers.GENERIC_MESSAGE


def test_unhandled_exception_logs_the_full_traceback(caplog):
    """Detail must still reach the server log, otherwise it is undiagnosable."""
    with caplog.at_level("ERROR", logger="echo.api"):
        asyncio.run(
            error_handlers.unhandled_exception_handler(
                make_request(), ValueError("boom-in-the-log")
            )
        )
    assert "boom-in-the-log" in caplog.text
    assert "ValueError" in caplog.text


def test_request_id_appears_in_both_response_and_log(caplog):
    with caplog.at_level("ERROR", logger="echo.api"):
        response = asyncio.run(
            error_handlers.unhandled_exception_handler(make_request(), ValueError("boom"))
        )
    request_id = body_of(response)["error"]["request_id"]
    assert request_id
    assert request_id in caplog.text


def test_existing_correlation_id_is_reused_in_body_header_and_log(caplog):
    request_id = "caller-trace-42"
    with caplog.at_level("ERROR", logger="echo.api"):
        response = asyncio.run(
            error_handlers.unhandled_exception_handler(
                make_request(correlation_id=request_id), ValueError("boom")
            )
        )

    assert body_of(response)["error"]["request_id"] == request_id
    assert response.headers[CORRELATION_ID_HEADER] == request_id
    assert request_id in caplog.text
    assert any(
        getattr(record, "correlation_id", None) == request_id
        for record in caplog.records
    )


def test_request_ids_are_unique():
    ids = {
        body_of(
            asyncio.run(
                error_handlers.unhandled_exception_handler(make_request(), ValueError("x"))
            )
        )["error"]["request_id"]
        for _ in range(20)
    }
    assert len(ids) == 20


# ---------------------------------------------------------------------------
# Deliberate HTTPExceptions
# ---------------------------------------------------------------------------


def test_http_exception_preserves_status_code():
    response = asyncio.run(
        error_handlers.http_exception_handler(
            make_request(), StarletteHTTPException(status_code=404, detail="Species not found")
        )
    )
    assert response.status_code == 404


def test_http_exception_preserves_the_message():
    """A 404 raised on purpose is not a fault, so its message is passed through."""
    response = asyncio.run(
        error_handlers.http_exception_handler(
            make_request(), StarletteHTTPException(status_code=404, detail="Species not found")
        )
    )
    assert body_of(response)["error"]["message"] == "Species not found"


def test_http_exception_uses_the_standard_envelope():
    response = asyncio.run(
        error_handlers.http_exception_handler(
            make_request(), StarletteHTTPException(status_code=403, detail="Forbidden")
        )
    )
    error = body_of(response)["error"]
    assert set(error) == {"type", "message", "request_id"}
    assert error["type"] == "http_error"


def test_http_exception_preserves_protocol_headers():
    response = asyncio.run(
        error_handlers.http_exception_handler(
            make_request(),
            StarletteHTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            ),
        )
    )
    assert response.headers["www-authenticate"] == "Bearer"


# ---------------------------------------------------------------------------
# Detection domain failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception,status_code,error_type",
    [
        (DetectionRuleError("bad filters"), 400, "detection_rule_error"),
        (DetectionNotFoundError("missing"), 404, "detection_not_found"),
        (DetectionStorageError("unavailable"), 503, "detection_storage_error"),
    ],
)
def test_detection_errors_are_mapped_to_stable_responses(
    exception, status_code, error_type
):
    response = asyncio.run(
        error_handlers.detection_exception_handler(make_request(), exception)
    )
    assert response.status_code == status_code
    assert body_of(response)["error"]["type"] == error_type


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def _validation_error():
    return RequestValidationError(
        [
            {
                "loc": ("body", "confidence"),
                "msg": "value is not a valid float",
                "type": "type_error.float",
            }
        ]
    )


def test_validation_error_returns_422():
    response = asyncio.run(
        error_handlers.validation_exception_handler(make_request("POST"), _validation_error())
    )
    assert response.status_code == 422


def test_validation_error_keeps_field_level_detail():
    """The caller needs to know which field was wrong in order to fix it."""
    response = asyncio.run(
        error_handlers.validation_exception_handler(make_request("POST"), _validation_error())
    )
    details = body_of(response)["error"]["details"]
    assert "confidence" in json.dumps(details)


def test_validation_error_uses_the_standard_envelope():
    response = asyncio.run(
        error_handlers.validation_exception_handler(make_request("POST"), _validation_error())
    )
    error = body_of(response)["error"]
    assert error["type"] == "validation_error"
    assert "request_id" in error


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_exception_handlers_attaches_all_three():
    from fastapi import FastAPI

    app = FastAPI()
    error_handlers.register_exception_handlers(app)

    assert app.exception_handlers[Exception] is error_handlers.unhandled_exception_handler
    assert (
        app.exception_handlers[StarletteHTTPException]
        is error_handlers.http_exception_handler
    )
    assert (
        app.exception_handlers[RequestValidationError]
        is error_handlers.validation_exception_handler
    )
    assert app.exception_handlers[DetectionError] is error_handlers.detection_exception_handler


def test_register_exception_handlers_returns_the_app():
    from fastapi import FastAPI

    app = FastAPI()
    assert error_handlers.register_exception_handlers(app) is app
