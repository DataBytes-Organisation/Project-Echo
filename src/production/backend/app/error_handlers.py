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
import uuid

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("echo.api")

# Returned instead of the real exception text for unhandled errors. The detail
# goes to the log, not to the caller.
GENERIC_MESSAGE = (
    "An internal error occurred. Quote the request_id when reporting this."
)


def new_request_id():
    """Return a short identifier used to tie a response back to a log entry."""
    return uuid.uuid4().hex[:12]


def error_response(status_code, error_type, message, request_id, extra=None):
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
    return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle deliberate HTTPException raises, e.g. raise HTTPException(404).

    These are expected outcomes rather than faults, so the message the route
    supplied is passed through unchanged. Only the envelope is standardised.
    """
    request_id = new_request_id()
    logger.info(
        "HTTP %s on %s %s (request_id=%s): %s",
        exc.status_code,
        request.method,
        request.url.path,
        request_id,
        exc.detail,
    )
    return error_response(
        status_code=exc.status_code,
        error_type="http_error",
        message=str(exc.detail),
        request_id=request_id,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle malformed request bodies and query parameters.

    FastAPI's default 422 body is a bare list, which is inconsistent with
    every other error the API returns. The field-level detail is preserved
    under "details" because the caller needs it to fix their request.
    """
    request_id = new_request_id()
    logger.info(
        "Validation failure on %s %s (request_id=%s): %s",
        request.method,
        request.url.path,
        request_id,
        exc.errors(),
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
    request_id = new_request_id()
    logger.error(
        "Unhandled %s on %s %s (request_id=%s)",
        type(exc).__name__,
        request.method,
        request.url.path,
        request_id,
        exc_info=exc,
    )
    return error_response(
        status_code=500,
        error_type="internal_error",
        message=GENERIC_MESSAGE,
        request_id=request_id,
    )


def register_exception_handlers(app):
    """Attach all handlers to ``app``.

    Call this once, on the FastAPI instance that is actually served. Note
    that app/main.py currently constructs FastAPI() twice; handlers attached
    to the discarded first instance would have no effect.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    return app
