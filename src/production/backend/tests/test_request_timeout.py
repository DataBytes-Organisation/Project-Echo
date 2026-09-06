import asyncio
import json
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from pydantic import ValidationError


REQUIRED_ENV = {
    "MONGODB_URI": "mongodb://user:pass@localhost:27017/EchoNet",
    "USER_MONGODB_URI": "mongodb://user:pass@localhost:27017/UserSample",
    "JWT_SECRET": "test-secret",
}

for key, value in REQUIRED_ENV.items():
    os.environ.setdefault(key, value)

from app.config import Settings, settings
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


def timeout_test_app(timeout_seconds=0.01):
    app = FastAPI()

    @app.get("/normal")
    async def normal():
        return {"status": "ok"}

    @app.get("/slow")
    async def slow():
        await asyncio.sleep(0.05)
        return {"status": "finished"}

    @app.post("/slow-write")
    async def slow_write():
        await asyncio.sleep(0.02)
        return {"status": "written"}

    app.add_middleware(
        RequestTimeoutMiddleware,
        timeout_seconds=timeout_seconds,
    )
    return app


class RequestTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_read_returns_standard_504_response(self):
        status, _, body = await request_app(timeout_test_app(), "GET", "/slow")

        self.assertEqual(status, 504)
        self.assertEqual(
            json.loads(body),
            {
                "error": {
                    "code": "GATEWAY_TIMEOUT",
                    "message": "Request timed out.",
                    "details": None,
                }
            },
        )

    async def test_normal_read_is_not_falsely_timed_out(self):
        status, _, body = await request_app(timeout_test_app(), "GET", "/normal")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})

    async def test_write_request_is_not_cancelled(self):
        status, _, body = await request_app(timeout_test_app(), "POST", "/slow-write")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "written"})

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
        # The C4 base branch currently passes optional mail settings into a
        # dependency that requires strings. Patch harmless values only while
        # importing the complete application so this C11 wiring test remains
        # independent of test discovery order.
        with patch.object(settings, "mail_username", "test@example.com"):
            with patch.object(settings, "mail_password", "test-password"):
                with patch.object(settings, "mail_from", "test@example.com"):
                    from app.main import app as project_app

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
