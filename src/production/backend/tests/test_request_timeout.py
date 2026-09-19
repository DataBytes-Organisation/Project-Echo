import asyncio
import json
import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError


REQUIRED_ENV = {
    "MONGODB_URI": "mongodb://user:pass@localhost:27017/EchoNet",
    "USER_MONGODB_URI": "mongodb://user:pass@localhost:27017/UserSample",
    "JWT_SECRET": "test-secret",
}

for key, value in REQUIRED_ENV.items():
    os.environ.setdefault(key, value)

from app.config import Settings, settings
from app.main import app as project_app
from app.middleware.request_timeout import RequestTimeoutMiddleware


async def request_app(app, method, path):
    """Call an ASGI application directly and return status, headers and body."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("test-client", 50000),
        "server": ("test-server", 80),
    }
    messages = []
    request_sent = False
    response_complete = asyncio.Event()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await response_complete.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            response_complete.set()

    await app(scope, receive, send)

    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in start["headers"]
    }
    return start["status"], headers, body


async def integrated_normal_read():
    return {"status": "ok"}


async def integrated_slow_read():
    await asyncio.sleep(0.05)
    return {"status": "finished"}


async def integrated_slow_write():
    await asyncio.sleep(0.02)
    return {"status": "written"}


# These routes exist only in the test process. Wrapping the real Project Echo
# app with a shorter timeout exercises its routers, exception handlers and
# middleware without making the production timeout test take 15 seconds.
project_app.add_api_route(
    "/__tests__/timeout/normal",
    integrated_normal_read,
    methods=["GET"],
    include_in_schema=False,
)
project_app.add_api_route(
    "/__tests__/timeout/slow-read",
    integrated_slow_read,
    methods=["GET"],
    include_in_schema=False,
)
project_app.add_api_route(
    "/__tests__/timeout/slow-write",
    integrated_slow_write,
    methods=["POST"],
    include_in_schema=False,
)


def integrated_timeout_app(timeout_seconds=0.01):
    return RequestTimeoutMiddleware(
        project_app,
        timeout_seconds=timeout_seconds,
    )


class RequestTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_read_returns_standard_504_response(self):
        status, _, body = await request_app(
            integrated_timeout_app(),
            "GET",
            "/__tests__/timeout/slow-read",
        )

        self.assertEqual(status, 504)
        self.assertEqual(
            json.loads(body),
            {
                "status": "failed",
                "error": {
                    "code": "GATEWAY_TIMEOUT",
                    "message": "Request timed out.",
                    "details": None,
                }
            },
        )

    async def test_normal_read_is_not_falsely_timed_out(self):
        status, headers, body = await request_app(
            integrated_timeout_app(),
            "GET",
            "/__tests__/timeout/normal",
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})
        self.assertIn("x-correlation-id", headers)

    async def test_write_request_is_not_cancelled(self):
        status, headers, body = await request_app(
            integrated_timeout_app(),
            "POST",
            "/__tests__/timeout/slow-write",
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "written"})
        self.assertIn("x-correlation-id", headers)

    def test_timeout_setting_accepts_environment_override(self):
        env = {**REQUIRED_ENV, "REQUEST_TIMEOUT_SECONDS": "2.5"}
        with patch.dict("os.environ", env, clear=True):
            configured = Settings()

        self.assertEqual(configured.request_timeout_seconds, 2.5)

    def test_timeout_setting_rejects_non_positive_value(self):
        env = {**REQUIRED_ENV, "REQUEST_TIMEOUT_SECONDS": "0"}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(ValidationError):
                Settings()

    def test_project_app_uses_central_timeout_setting(self):
        middleware = next(
            item
            for item in project_app.user_middleware
            if item.cls is RequestTimeoutMiddleware
        )
        self.assertEqual(
            middleware.kwargs["timeout_seconds"],
            settings.request_timeout_seconds,
        )


if __name__ == "__main__":
    unittest.main()
