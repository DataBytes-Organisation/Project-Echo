"""Application-wide exception handling for the Project Echo backend.

Without these handlers, an unexpected error escapes FastAPI and the caller
receives a raw traceback containing file paths, line numbers and sometimes
variable contents. That is an information leak, and it is unhelpful to the
client, which cannot parse it.

Registering handlers centrally means every route is covered, including the
nine routers that currently contain no try/except of their own. It is also
the reason this is a better approach than adding try/except to each route:
there is one place to change the response shape, and no route can be
forgotten.

Every response produced here uses the same JSON shape::

    {"error": {"type": "...", "message": "...", "request_id": "..."}}

The request_id is also written to the server log, so a user reporting "I got
an error, the id was 3f2a..." can be matched to the exact traceback.

Related task: C4.3.
"""

import logging
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions import DetectionError
from app.middleware.correlation_id import (
    CORRELATION_ID_HEADER,
    configure_correlation_logging,
    correlation_id_for_request,
    new_correlation_id,
)

logger = logging.getLogger("echo.api")
configure_correlation_logging(logger)

# Returned instead of the real exception text for unhandled errors. The detail
# goes to the log, not to the caller.
GENERIC_MESSAGE = (
    "An internal error occurred. Quote the request_id when reporting this."
)


def new_request_id():
    """Backward-compatible alias for the correlation ID generator."""
    return new_correlation_id()


def error_response(
    status_code,
    error_type,
    message,
    request_id,
    extra=None,
    headers=None,
):
    """Build the standard error response body."""
    payload = {
        "error": {
            "type": error_type,
            "message": message,
            "request_id": request_id,
        }
    }
    if extra is not None:
        payload["error"]["details"] = extra
    response_headers = dict(headers or {})
    response_headers[CORRELATION_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(payload),
        headers=response_headers,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle deliberate HTTPException raises, e.g. raise HTTPException(404).

    These are expected outcomes rather than faults, so the message the route
    supplied is passed through unchanged. Only the envelope is standardised.
    """
    request_id = correlation_id_for_request(request)
    logger.info(
        "HTTP %s on %s %s (request_id=%s): %s",
        exc.status_code,
        request.method,
        request.url.path,
        request_id,
        exc.detail,
        extra={"correlation_id": request_id},
    )
    return error_response(
        status_code=exc.status_code,
        error_type="http_error",
        message=str(exc.detail),
        request_id=request_id,
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle malformed request bodies and query parameters.

    FastAPI's default 422 body is a bare list, which is inconsistent with
    every other error the API returns. The field-level detail is preserved
    under "details" because the caller needs it to fix their request.
    """
    request_id = correlation_id_for_request(request)
    logger.info(
        "Validation failure on %s %s (request_id=%s): %s",
        request.method,
        request.url.path,
        request_id,
        exc.errors(),
        extra={"correlation_id": request_id},
    )
    return error_response(
        status_code=422,
        error_type="validation_error",
        message="The request body or parameters failed validation.",
        request_id=request_id,
        extra=exc.errors(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch anything not handled above.

    The full traceback is written to the server log. The client receives only
    a generic message and the request_id, so that internal paths, driver
    internals and connection strings are never returned over the network.

    exc_info is passed the exception object explicitly rather than relying on
    logger.exception(), which reads the ambient exception state and therefore
    records nothing when the handler is invoked outside an except block.
    """
    request_id = correlation_id_for_request(request)
    logger.error(
        "Unhandled %s on %s %s (request_id=%s)",
        type(exc).__name__,
        request.method,
        request.url.path,
        request_id,
        exc_info=exc,
        extra={"correlation_id": request_id},
    )
    return error_response(
        status_code=500,
        error_type="internal_error",
        message=GENERIC_MESSAGE,
        request_id=request_id,
    )


async def detection_exception_handler(request: Request, exc: DetectionError):
    """Map expected detection failures to stable, client-safe responses."""
    request_id = correlation_id_for_request(request)
    log_level = logging.WARNING if exc.status_code >= 500 else logging.INFO
    logger.log(
        log_level,
        "Detection failure on %s %s (request_id=%s, type=%s): %s",
        request.method,
        request.url.path,
        request_id,
        exc.error_type,
        exc,
        exc_info=exc if exc.status_code >= 500 else None,
        extra={"correlation_id": request_id},
    )
    return error_response(
        status_code=exc.status_code,
        error_type=exc.error_type,
        message=str(exc),
        request_id=request_id,
    )


def register_exception_handlers(app):
    """Attach all handlers to ``app``.

    Call this once, on the FastAPI instance that is actually served.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    # Register the domain exception before the generic catch-all. FastAPI
    # resolves the most specific matching exception class at request time.
    app.add_exception_handler(DetectionError, detection_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    return app
