"""Correlation IDs for API responses and structured log records.

Each HTTP request gets one identifier.  A valid caller-provided
``X-Correlation-ID`` is reused; otherwise the middleware generates one.  The
same value is exposed through request state, returned as a response header,
and attached to log records written while the request is being processed.
"""

import contextvars
import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders

CORRELATION_ID_HEADER = "X-Correlation-ID"
MAX_CORRELATION_ID_LENGTH = 128
_VALID_CORRELATION_ID = re.compile(r"^[A-Za-z0-9._:-]+$")
_correlation_id = contextvars.ContextVar("echo_correlation_id", default=None)

logger = logging.getLogger("echo.api")


def new_correlation_id():
    """Return an opaque identifier suitable for responses and logs."""
    return uuid.uuid4().hex


def is_valid_correlation_id(value):
    """Reject blank, oversized, or log/header-unsafe caller values."""
    return bool(
        value
        and len(value) <= MAX_CORRELATION_ID_LENGTH
        and _VALID_CORRELATION_ID.fullmatch(value)
    )


def get_correlation_id(default=None):
    """Return the ID bound to the current async request context."""
    return _correlation_id.get() or default


def correlation_id_for_request(request):
    """Read the ID installed on a Starlette request, with safe fallbacks."""
    request_id = getattr(request.state, "correlation_id", None)
    if request_id:
        return request_id

    incoming = request.headers.get(CORRELATION_ID_HEADER)
    if is_valid_correlation_id(incoming):
        return incoming

    return get_correlation_id() or new_correlation_id()


class CorrelationIdFilter(logging.Filter):
    """Attach ``correlation_id`` to log records for structured formatters."""

    def filter(self, record):
        if not getattr(record, "correlation_id", None):
            record.correlation_id = get_correlation_id("-")
        return True


def configure_correlation_logging(target_logger=None):
    """Install the correlation filter once on the application logger."""
    target_logger = target_logger or logger
    if not any(isinstance(item, CorrelationIdFilter) for item in target_logger.filters):
        target_logger.addFilter(CorrelationIdFilter())
    return target_logger


class CorrelationIdMiddleware:
    """Plain ASGI middleware so streaming responses remain unbuffered."""

    def __init__(self, app, header_name=CORRELATION_ID_HEADER):
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        incoming = headers.get(self.header_name)
        request_id = incoming if is_valid_correlation_id(incoming) else new_correlation_id()

        state = scope.setdefault("state", {})
        state["correlation_id"] = request_id
        token = _correlation_id.set(request_id)
        started_at = time.perf_counter()
        status_code = 500

        async def send_with_correlation_id(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers[self.header_name] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_correlation_id)
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Request completed method=%s path=%s status=%s duration_ms=%.2f correlation_id=%s",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status_code,
                duration_ms,
                request_id,
                extra={
                    "correlation_id": request_id,
                    "http_method": scope.get("method", "-"),
                    "http_path": scope.get("path", "-"),
                    "http_status": status_code,
                    "duration_ms": duration_ms,
                },
            )
            _correlation_id.reset(token)


def add_correlation_id(app):
    """Attach request correlation middleware and logging support."""
    configure_correlation_logging()
    app.add_middleware(CorrelationIdMiddleware)
    return app
