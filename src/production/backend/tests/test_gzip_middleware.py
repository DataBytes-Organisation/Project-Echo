import asyncio
import gzip
import json
import unittest

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.main import app
from app.errors import StandardizeErrorResponseMiddleware, error_response
from starlette.middleware.gzip import GZipMiddleware


async def request_app(path, accept_encoding=None, application=app):
    """Call the ASGI app directly and return its status, headers, and body."""
    request_headers = []
    if accept_encoding is not None:
        request_headers.append((b"accept-encoding", accept_encoding.encode("ascii")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": request_headers,
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

    await application(scope, receive, send)

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


def middleware_test_app():
    """Build a small app that exercises the production middleware ordering."""
    test_app = FastAPI()
    test_app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,
        compresslevel=6,
    )
    test_app.add_middleware(StandardizeErrorResponseMiddleware)

    @test_app.get("/large-error")
    async def large_error():
        return error_response(
            400,
            "The request could not be completed.",
            {"context": "x" * 2000},
        )

    @test_app.get("/large-success")
    async def large_success():
        return JSONResponse({"content": "x" * 2000})

    @test_app.get("/small-success")
    async def small_success():
        return JSONResponse({"content": "ok"})

    return test_app


class GZipMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def test_gzip_middleware_configuration(self):
        middleware = next(
            item for item in app.user_middleware if item.cls is GZipMiddleware
        )

        self.assertEqual(middleware.kwargs["minimum_size"], 1000)
        self.assertEqual(middleware.kwargs["compresslevel"], 6)

    def test_gzip_runs_inside_error_standardization_middleware(self):
        middleware_classes = [item.cls for item in app.user_middleware]

        self.assertLess(
            middleware_classes.index(StandardizeErrorResponseMiddleware),
            middleware_classes.index(GZipMiddleware),
        )

    async def test_large_response_is_compressed_for_supported_client(self):
        status, headers, body = await request_app("/openapi.json", "gzip")

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("content-encoding"), "gzip")
        self.assertIn("accept-encoding", headers.get("vary", "").lower())
        self.assertGreater(len(gzip.decompress(body)), 1000)

    async def test_small_response_is_not_compressed(self):
        status, headers, body = await request_app("/", "gzip")

        self.assertEqual(status, 200)
        self.assertNotIn("content-encoding", headers)
        self.assertLess(len(body), 1000)

    async def test_client_without_gzip_gets_normal_large_response(self):
        status, headers, body = await request_app("/openapi.json")

        self.assertEqual(status, 200)
        self.assertNotIn("content-encoding", headers)
        self.assertGreater(len(body), 1000)

    async def test_large_compressed_error_body_is_complete_and_readable(self):
        test_app = middleware_test_app()

        status, headers, body = await request_app(
            "/large-error", "gzip", application=test_app
        )

        self.assertEqual(status, 400)
        self.assertEqual(headers.get("content-encoding"), "gzip")
        self.assertGreater(len(body), 0)

        payload = json.loads(gzip.decompress(body))
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"]["code"], "BAD_REQUEST")
        self.assertEqual(
            payload["error"]["message"],
            "The request could not be completed.",
        )
        self.assertEqual(payload["error"]["details"]["context"], "x" * 2000)

    async def test_isolated_large_success_is_compressed(self):
        test_app = middleware_test_app()

        status, headers, body = await request_app(
            "/large-success", "gzip", application=test_app
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("content-encoding"), "gzip")
        self.assertEqual(json.loads(gzip.decompress(body))["content"], "x" * 2000)

    async def test_isolated_small_success_is_not_compressed(self):
        test_app = middleware_test_app()

        status, headers, body = await request_app(
            "/small-success", "gzip", application=test_app
        )

        self.assertEqual(status, 200)
        self.assertNotIn("content-encoding", headers)
        self.assertEqual(json.loads(body), {"content": "ok"})


if __name__ == "__main__":
    unittest.main()
