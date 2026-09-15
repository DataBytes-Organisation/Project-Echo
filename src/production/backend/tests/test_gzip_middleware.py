import asyncio
import gzip
import unittest

from app.main import app
from starlette.middleware.gzip import GZipMiddleware


async def request_app(path, accept_encoding=None):
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


class GZipMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def test_gzip_middleware_configuration(self):
        middleware = next(
            item for item in app.user_middleware if item.cls is GZipMiddleware
        )

        self.assertEqual(middleware.kwargs["minimum_size"], 1000)
        self.assertEqual(middleware.kwargs["compresslevel"], 6)

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


if __name__ == "__main__":
    unittest.main()
