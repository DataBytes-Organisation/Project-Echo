"""Configurable timeout protection for safe Backend API requests."""

import asyncio
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.errors import error_response


logger = logging.getLogger(__name__)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Return a standard 504 response when a safe request takes too long.

    Write requests are deliberately excluded. Cancelling a database write after
    the client receives a timeout can leave the result uncertain, because the
    underlying synchronous operation may still finish in a worker thread.
    """

    SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

    def __init__(self, app, timeout_seconds: float) -> None:
        super().__init__(app)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.timeout_seconds = timeout_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.method not in self.SAFE_METHODS:
            return await call_next(request)

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Backend request timed out",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "timeout_seconds": self.timeout_seconds,
                },
            )
            return error_response(
                504,
                "Request timed out.",
            )
