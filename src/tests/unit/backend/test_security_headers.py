"""Tests for the security response headers middleware.

The middleware is plain ASGI, so each test drives it directly with a scope, a
receive callable and a send callable, and inspects the response start message
it produces. This needs no HTTP client and therefore no extra dependency.
"""

import asyncio

import pytest

from app.middleware.security_headers import (
    API_CSP,
    BASE_HEADERS,
    DOCS_CSP,
    HSTS_VALUE,
    SecurityHeadersMiddleware,
)


async def dummy_app(scope, receive, send):
    """A minimal inner application returning a JSON response."""
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"server", b"uvicorn"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": b"{}"})


def response_headers(path="/detections", hsts=False, inner=dummy_app, scope_type="http"):
    """Run the middleware and return the response headers as a lower-cased dict."""
    middleware = SecurityHeadersMiddleware(inner, hsts_enabled=hsts)
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": scope_type,
        "method": "GET",
        "path": path,
        "headers": [],
        "scheme": "http",
    }
    asyncio.run(middleware(scope, receive, send))

    start = next(m for m in messages if m["type"] == "http.response.start")
    return {k.decode().lower(): v.decode() for k, v in start["headers"]}


# ---------------------------------------------------------------------------
# The headers are present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,value", sorted(BASE_HEADERS.items()))
def test_base_header_is_present(name, value):
    assert response_headers()[name.lower()] == value


def test_content_security_policy_is_present():
    assert "content-security-policy" in response_headers()


def test_existing_response_headers_are_preserved():
    assert response_headers()["content-type"] == "application/json"


def test_server_header_set_by_the_application_is_removed():
    """A Server header emitted through ASGI is stripped.

    This does not cover uvicorn's own "server: uvicorn". uvicorn writes that
    in its HTTP protocol layer, after all ASGI middleware has run, so no
    middleware can remove it and no unit test at this level can detect it.
    It is suppressed with the --no-server-header flag in API.Dockerfile and
    in the docker-compose command, and verified by inspecting the headers of
    a response from the running container.
    """
    assert "server" not in response_headers()


# ---------------------------------------------------------------------------
# Content-Security-Policy differs for the documentation pages
# ---------------------------------------------------------------------------


def test_api_paths_get_the_restrictive_policy():
    assert response_headers("/detections")["content-security-policy"] == API_CSP


def test_api_policy_allows_nothing_to_load():
    assert "default-src 'none'" in API_CSP


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_paths_get_the_relaxed_policy(path):
    """Swagger UI and ReDoc load assets from a CDN and would render blank
    under the API policy."""
    assert response_headers(path)["content-security-policy"] == DOCS_CSP


def test_docs_policy_allows_the_cdn():
    assert "https://cdn.jsdelivr.net" in DOCS_CSP


def test_docs_policy_still_forbids_framing():
    assert "frame-ancestors 'none'" in DOCS_CSP


# ---------------------------------------------------------------------------
# Strict-Transport-Security is opt-in
# ---------------------------------------------------------------------------


def test_hsts_is_absent_by_default():
    """Local development is served over plain HTTP; sending HSTS there would
    make the browser refuse to connect."""
    assert "strict-transport-security" not in response_headers()


def test_hsts_is_present_when_enabled():
    assert response_headers(hsts=True)["strict-transport-security"] == HSTS_VALUE


def test_hsts_covers_subdomains():
    assert "includeSubDomains" in HSTS_VALUE


# ---------------------------------------------------------------------------
# Behaviour in edge cases
# ---------------------------------------------------------------------------


def test_a_route_may_override_a_header():
    """setdefault, not assignment: a deliberate per-route value must win."""

    async def app_setting_its_own(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"x-frame-options", b"SAMEORIGIN")],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    assert response_headers(inner=app_setting_its_own)["x-frame-options"] == "SAMEORIGIN"


def test_error_responses_also_receive_the_headers():
    """Headers must not depend on the status code, or a 500 would be exempt."""

    async def failing_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    assert response_headers(inner=failing_app)["x-content-type-options"] == "nosniff"


def test_non_http_scopes_pass_straight_through():
    """Websocket and lifespan messages have no response headers to set."""
    seen = {}

    async def websocket_app(scope, receive, send):
        seen["type"] = scope["type"]

    middleware = SecurityHeadersMiddleware(websocket_app)

    async def receive():
        return {}

    async def send(message):
        raise AssertionError("send should not be called")

    asyncio.run(middleware({"type": "websocket", "path": "/live"}, receive, send))
    assert seen["type"] == "websocket"


def test_headers_for_is_independent_between_calls():
    """The returned dict must not be shared, or one request could mutate the
    headers of the next."""
    middleware = SecurityHeadersMiddleware(dummy_app)
    first = middleware.headers_for("/a")
    first["X-Frame-Options"] = "MUTATED"
    assert middleware.headers_for("/b")["X-Frame-Options"] == "DENY"
