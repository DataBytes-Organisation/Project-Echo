"""One public error contract for every Project Echo API endpoint."""

import json
import logging
from typing import Any, Optional

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.exceptions import DetectionError
from app.middleware.correlation_id import CORRELATION_ID_HEADER, correlation_id_for_request

logger = logging.getLogger(__name__)
STATUS_CODES = {400: "BAD_REQUEST", 401: "UNAUTHENTICATED", 403: "FORBIDDEN", 404: "RESOURCE_NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 409: "CONFLICT", 410: "RESOURCE_GONE", 413: "PAYLOAD_TOO_LARGE", 422: "VALIDATION_ERROR", 423: "LOCKED", 429: "RATE_LIMIT_EXCEEDED", 500: "INTERNAL_ERROR", 502: "UPSTREAM_ERROR", 503: "SERVICE_UNAVAILABLE", 504: "GATEWAY_TIMEOUT"}


def error_body(
    status_code: int,
    message: str,
    details: Optional[Any] = None,
    correlation_id: Optional[str] = None,
) -> dict:
    error = {"code": STATUS_CODES.get(status_code, "REQUEST_FAILED"), "message": message}
    error["details"] = details
    if correlation_id:
        error["correlation_id"] = correlation_id
    return {"status": "failed", "error": error}


def error_response(
    status_code: int,
    message: str,
    details: Optional[Any] = None,
    headers: Optional[dict] = None,
    correlation_id: Optional[str] = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    if correlation_id:
        response_headers[CORRELATION_ID_HEADER] = correlation_id
    return JSONResponse(
        status_code=status_code,
        content=error_body(status_code, message, details, correlation_id),
        headers=response_headers,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    details = exc.detail if not isinstance(exc.detail, str) else None
    message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return error_response(
        exc.status_code,
        message,
        details,
        dict(exc.headers) if exc.headers else None,
        correlation_id_for_request(request),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(
        422,
        "Request validation failed.",
        exc.errors(),
        correlation_id=correlation_id_for_request(request),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error", exc_info=exc)
    return error_response(
        500,
        "An unexpected server error occurred.",
        correlation_id=correlation_id_for_request(request),
    )


async def detection_exception_handler(request: Request, exc: DetectionError) -> JSONResponse:
    """Map validated detection failures into the shared API error contract."""
    request_id = correlation_id_for_request(request)
    logger.warning(
        "Detection request failed: %s", exc, extra={"correlation_id": request_id}
    )
    return error_response(
        exc.status_code,
        str(exc),
        details={"type": exc.error_type},
        correlation_id=request_id,
    )


class StandardizeErrorResponseMiddleware(BaseHTTPMiddleware):
    """Normalise legacy JSON error responses returned directly by routes."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if response.status_code < 400 or "application/json" not in response.headers.get("content-type", ""):
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            # Reading body_iterator consumes it. Restore the original bytes before
            # returning an error response that is not JSON-decodable.
            response.body_iterator = _single_chunk_iterator(body)
            return response
        if (
            isinstance(payload, dict)
            and payload.get("status") == "failed"
            and isinstance(payload.get("error"), dict)
            and {"code", "message"} <= set(payload["error"])
        ):
            response.body_iterator = _single_chunk_iterator(body)
            return response
        raw_message = payload.get("message", payload.get("detail", payload.get("error"))) if isinstance(payload, dict) else None
        message = raw_message if isinstance(raw_message, str) else "The request could not be completed."
        details = payload if not isinstance(raw_message, str) else None
        headers = {key: value for key, value in response.headers.items() if key.lower() not in {"content-length", "content-type"}}
        return error_response(
            response.status_code,
            message,
            details,
            headers,
            correlation_id_for_request(request),
        )


async def _single_chunk_iterator(body: bytes):
    yield body
