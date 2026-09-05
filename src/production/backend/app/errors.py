"""One public error contract for every Project Echo API endpoint."""

import json
import logging
from typing import Any, Optional

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)
STATUS_CODES = {400: "BAD_REQUEST", 401: "UNAUTHENTICATED", 403: "FORBIDDEN", 404: "RESOURCE_NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 409: "CONFLICT", 410: "RESOURCE_GONE", 413: "PAYLOAD_TOO_LARGE", 422: "VALIDATION_ERROR", 423: "LOCKED", 429: "RATE_LIMIT_EXCEEDED", 500: "INTERNAL_ERROR", 502: "UPSTREAM_ERROR", 503: "SERVICE_UNAVAILABLE"}


def error_body(status_code: int, message: str, details: Optional[Any] = None) -> dict:
    return {"error": {"code": STATUS_CODES.get(status_code, "REQUEST_FAILED"), "message": message, "details": details}}


def error_response(status_code: int, message: str, details: Optional[Any] = None, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(status_code, message, details), headers=headers)


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    details = exc.detail if not isinstance(exc.detail, str) else None
    message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return error_response(exc.status_code, message, details, dict(exc.headers) if exc.headers else None)


async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(422, "Request validation failed.", exc.errors())


async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error", exc_info=exc)
    return error_response(500, "An unexpected server error occurred.")


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
            return response
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict) and {"code", "message", "details"} <= set(payload["error"]):
            return response
        raw_message = payload.get("message", payload.get("detail", payload.get("error"))) if isinstance(payload, dict) else None
        message = raw_message if isinstance(raw_message, str) else "The request could not be completed."
        details = payload if not isinstance(raw_message, str) else None
        headers = {key: value for key, value in response.headers.items() if key.lower() not in {"content-length", "content-type"}}
        return error_response(response.status_code, message, details, headers)
