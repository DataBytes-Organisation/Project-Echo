"""Security response headers for the Project Echo backend.

Browsers apply a number of protections only when the server asks for them, by
sending particular response headers. The API currently sends none of them.
This middleware adds them to every response, including responses produced by
the exception handlers.

The Node.js equivalent is Helmet, which the task description names as an
example. There is no Helmet for Python; the same job is done here by a small
piece of ASGI middleware, which avoids adding a dependency to a requirements
file that is already large.

Headers sent on every response:

    X-Content-Type-Options        stops the browser guessing that a response
                                  is a script and executing it
    X-Frame-Options               stops the response being embedded in a
                                  frame on another site
    Referrer-Policy               stops the request URL being disclosed to
                                  other sites
    Cross-Origin-Opener-Policy    isolates the browsing context
    X-Permitted-Cross-Domain-...  stops legacy plugin cross-domain access
    Permissions-Policy            switches off device APIs this API never uses
    Content-Security-Policy       restricts what the response may load
    Strict-Transport-Security     optional, see config.hsts_enabled

Related task: security header middleware, fourth item of Global
Configurations and Security Headers.
"""

from starlette.datastructures import MutableHeaders

# Paths served by FastAPI's interactive documentation. Swagger UI and ReDoc
# load their JavaScript and CSS from a CDN, so the restrictive policy applied
# to API responses would leave these pages blank.
DOCS_PATHS = frozenset(
    ["/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"]
)

# API responses are JSON and are never rendered as a document, so nothing
# needs to be loadable at all.
API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"

# The documentation pages do render, and need their assets. Kept as narrow as
# Swagger UI allows: 'unsafe-inline' is required for its bootstrap styles and
# initialisation script, and cannot be removed without vendoring the assets
# locally.
DOCS_CSP = (
    "default-src 'none'; "
    "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' https://cdn.jsdelivr.net; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'"
)

BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "X-Permitted-Cross-Domain-Policies": "none",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

HSTS_VALUE = "max-age=31536000; includeSubDomains"


class SecurityHeadersMiddleware:
    """ASGI middleware that adds security headers to every HTTP response.

    Written as plain ASGI middleware rather than a BaseHTTPMiddleware
    subclass. BaseHTTPMiddleware buffers the response body, which would
    interfere with the streaming and websocket endpoints in this application;
    this class only rewrites the response start message and leaves the body
    untouched.
    """

    def __init__(self, app, hsts_enabled=False, docs_paths=DOCS_PATHS):
        self.app = app
        self.hsts_enabled = hsts_enabled
        self.docs_paths = frozenset(docs_paths)

    def headers_for(self, path):
        """Return the headers to apply to a response for ``path``."""
        headers = dict(BASE_HEADERS)
        headers["Content-Security-Policy"] = (
            DOCS_CSP if path in self.docs_paths else API_CSP
        )
        if self.hsts_enabled:
            headers["Strict-Transport-Security"] = HSTS_VALUE
        return headers

    async def __call__(self, scope, receive, send):
        # Websocket and lifespan messages have no response headers to set.
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in self.headers_for(path).items():
                    # setdefault, not assignment: a route that has already
                    # set one of these deliberately keeps its own value.
                    headers.setdefault(name, value)
                # Remove the Server header if the application set one; it
                # discloses the server software to no benefit.
                #
                # Note that this does NOT remove uvicorn's own "server:
                # uvicorn". uvicorn adds that in its HTTP protocol layer,
                # after every ASGI middleware has run, so it is not visible
                # here and cannot be deleted from this position. It is
                # suppressed instead by starting uvicorn with
                # --no-server-header, which is done in API.Dockerfile and in
                # the docker-compose command.
                if "server" in headers:
                    del headers["server"]
            await send(message)

        await self.app(scope, receive, send_with_headers)


def add_security_headers(app, hsts_enabled=False):
    """Attach SecurityHeadersMiddleware to ``app``."""
    app.add_middleware(SecurityHeadersMiddleware, hsts_enabled=hsts_enabled)
    return app
