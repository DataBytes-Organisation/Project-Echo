"""Automated tests for the Engine-to-Backend integration client."""

from __future__ import annotations

import json
import socket
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib import error

from client import BackendRequestError, load_payload, send_detection


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ResponseContext:
    """Small context-manager response used by the standard-library mocks."""

    def __init__(self, status: int, body: object) -> None:
        self.status = status
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "ResponseContext":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def make_http_error(status: int, body: str) -> error.HTTPError:
    """Create an HTTP error with a readable response body."""

    response_body = MagicMock()
    response_body.read.return_value = body.encode("utf-8")
    http_error = error.HTTPError(
        url="http://localhost:9000/engine/event",
        code=status,
        msg=body,
        hdrs=None,
        fp=response_body,
    )
    return http_error


class IntegrationHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend_url = "http://localhost:9000/engine/event"
        self.valid_fixtures = sorted(
            (FIXTURES / "valid_payloads").glob("valid_detection_*.json")
        )
        self.invalid_fixtures = sorted(
            (FIXTURES / "invalid_payloads").glob("invalid_detection_*.json")
        )
        self.valid_payload = load_payload(self.valid_fixtures[0])

    @patch("client.request.urlopen")
    def test_successful_simulated_request(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = ResponseContext(201, {"_id": "test-event-id"})

        self.assertEqual(len(self.valid_fixtures), 5)
        for fixture in self.valid_fixtures:
            with self.subTest(fixture=fixture.name):
                response = send_detection(self.backend_url, load_payload(fixture))
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.body["_id"], "test-event-id")

    @patch("client.request.urlopen")
    def test_invalid_payload_returns_422(self, mock_urlopen: MagicMock) -> None:
        def reject_payload(*args: object, **kwargs: object) -> None:
            raise make_http_error(422, '{"detail":"payload validation failed"}')

        mock_urlopen.side_effect = reject_payload

        self.assertEqual(len(self.invalid_fixtures), 5)
        for fixture in self.invalid_fixtures:
            with self.subTest(fixture=fixture.name):
                with self.assertRaises(BackendRequestError) as raised:
                    send_detection(self.backend_url, load_payload(fixture))
                self.assertEqual(raised.exception.status_code, 422)

    @patch("client.request.urlopen", side_effect=socket.timeout("timed out"))
    def test_timeout_is_reported(self, mock_urlopen: MagicMock) -> None:
        with self.assertRaisesRegex(BackendRequestError, "timed out"):
            send_detection(self.backend_url, self.valid_payload, timeout_seconds=0.1)

    @patch("client.request.urlopen")
    def test_connection_failure_is_reported(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = error.URLError("connection refused")

        with self.assertRaisesRegex(BackendRequestError, "Could not connect"):
            send_detection(self.backend_url, self.valid_payload)

    @patch("client.request.urlopen")
    def test_backend_500_is_reported(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = make_http_error(500, "storage failure")

        with self.assertRaises(BackendRequestError) as raised:
            send_detection(self.backend_url, self.valid_payload)

        self.assertEqual(raised.exception.status_code, 500)

    @patch("client.request.urlopen")
    def test_unexpected_success_status_is_rejected(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = ResponseContext(200, {"message": "unexpected"})

        with self.assertRaisesRegex(BackendRequestError, "Unexpected Backend status 200"):
            send_detection(self.backend_url, self.valid_payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
