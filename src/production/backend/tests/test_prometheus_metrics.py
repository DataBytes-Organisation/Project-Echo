"""C8.2 validation tests for Prometheus backend metrics."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.metrics import PrometheusMiddleware, metrics_router


app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.include_router(metrics_router)


@app.get("/test-success")
def success_route():
    return {"status": "ok"}


@app.get("/test-items/{item_id}")
def dynamic_route(item_id: str):
    return {"item_id": item_id}


client = TestClient(app)


def sample_value(name, labels):
    return REGISTRY.get_sample_value(name, labels)


def test_successful_request_increments_counter_and_latency():
    labels = {
        "method": "GET",
        "route": "/test-success",
        "status": "200",
    }

    before_requests = sample_value("echo_http_requests_total", labels) or 0
    before_latency_count = sample_value(
        "echo_http_request_seconds_count",
        {"route": "/test-success"},
    ) or 0

    response = client.get("/test-success")

    assert response.status_code == 200

    after_requests = sample_value("echo_http_requests_total", labels)
    after_latency_count = sample_value(
        "echo_http_request_seconds_count",
        {"route": "/test-success"},
    )

    assert after_requests == before_requests + 1
    assert after_latency_count == before_latency_count + 1


def test_unmatched_routes_use_low_cardinality_label():
    labels = {
        "method": "GET",
        "route": "unmatched",
        "status": "404",
    }

    before = sample_value("echo_http_requests_total", labels) or 0

    assert client.get("/does-not-exist-one").status_code == 404
    assert client.get("/does-not-exist-two").status_code == 404

    after = sample_value("echo_http_requests_total", labels)

    assert after == before + 2


def test_dynamic_routes_use_route_template_not_raw_ids():
    route = "/test-items/{item_id}"

    labels = {
        "method": "GET",
        "route": route,
        "status": "200",
    }

    before = sample_value("echo_http_requests_total", labels) or 0

    assert client.get("/test-items/abc123").status_code == 200
    assert client.get("/test-items/xyz789").status_code == 200

    after = sample_value("echo_http_requests_total", labels)

    assert after == before + 2

    assert sample_value(
        "echo_http_requests_total",
        {
            "method": "GET",
            "route": "/test-items/abc123",
            "status": "200",
        },
    ) is None


def test_metrics_endpoint_is_not_self_instrumented():
    for _ in range(3):
        assert client.get("/metrics").status_code == 200

    assert sample_value(
        "echo_http_requests_total",
        {
            "method": "GET",
            "route": "/metrics",
            "status": "200",
        },
    ) is None
