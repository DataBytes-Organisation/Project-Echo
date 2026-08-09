"""Configurable, per-client sliding-window rate limiting for the FastAPI API.

The limiter stores counters in this API process. This is appropriate for a
single backend process; use a shared store such as Redis before scaling to
multiple API replicas.
"""

import asyncio
import math
import os
import time
from collections import defaultdict, deque
from typing import Deque, DefaultDict

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware


DEFAULT_EXEMPT_PATHS = {"/", "/docs", "/openapi.json", "/openapi-export", "/spec/summary"}


def _env_bool(name: str, default: bool) -> bool:
    """Read a conventional boolean environment variable safely."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    """Read a positive integer setting, falling back for invalid values."""
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Limit non-exempt requests per client IP using a sliding time window."""

    def __init__(self, app, requests_per_window: int, window_seconds: int, enabled: bool):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._requests: DefaultDict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._next_cleanup = 0.0

    @classmethod
    def from_environment(cls, app):
        """Build middleware from documented API environment variables."""
        return cls(
            app=app,
            requests_per_window=_env_positive_int("RATE_LIMIT_REQUESTS", 120),
            window_seconds=_env_positive_int("RATE_LIMIT_WINDOW_SECONDS", 60),
            enabled=_env_bool("RATE_LIMIT_ENABLED", True),
        )

    @staticmethod
    def _is_exempt(path: str) -> bool:
        return path in DEFAULT_EXEMPT_PATHS or path.startswith("/docs/") or path.startswith("/redoc")

    @staticmethod
    def _client_key(request: Request) -> str:
        """Return the client address; trust forwarded headers only when enabled."""
        if _env_bool("RATE_LIMIT_TRUST_PROXY", False):
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",", 1)[0].strip()
        return request.client.host if request.client else "unknown"

    def _purge_expired_clients(self, now: float) -> None:
        """Periodically remove empty histories so inactive clients do not accumulate."""
        if now < self._next_cleanup:
            return

        cutoff = now - self.window_seconds
        for client_key, history in list(self._requests.items()):
            while history and history[0] <= cutoff:
                history.popleft()
            if not history:
                del self._requests[client_key]
        self._next_cleanup = now + self.window_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled or request.method == "OPTIONS" or self._is_exempt(request.url.path):
            return await call_next(request)

        client_key = self._client_key(request)
        now = time.monotonic()
        async with self._lock:
            self._purge_expired_clients(now)
            history = self._requests[client_key]
            cutoff = now - self.window_seconds
            while history and history[0] <= cutoff:
                history.popleft()
            if len(history) >= self.requests_per_window:
                retry_after = max(1, math.ceil(self.window_seconds - (now - history[0])))
                return JSONResponse(
                    status_code=429,
                    content={"error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests. Please try again later.", "details": None}},
                    headers={"Retry-After": str(retry_after), "X-RateLimit-Limit": str(self.requests_per_window), "X-RateLimit-Remaining": "0"},
                )
            history.append(now)
            remaining = self.requests_per_window - len(history)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
