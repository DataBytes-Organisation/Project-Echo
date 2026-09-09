"""Low-cardinality HTTP metrics for Prometheus scrapes."""

import time

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQS = Counter("echo_http_requests_total", "HTTP requests", ["method", "route", "status"])
LAT = Histogram("echo_http_request_seconds", "HTTP latency", ["route"])

metrics_router = APIRouter()

SKIP_PATHS = frozenset({"/metrics"})


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    return "unmatched"


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = _route_template(request)
            LAT.labels(route=route).observe(time.perf_counter() - start)
            REQS.labels(method=request.method, route=route, status=str(status)).inc()


@metrics_router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
