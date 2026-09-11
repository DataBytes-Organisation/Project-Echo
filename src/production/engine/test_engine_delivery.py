"""
Automated tests for Engine Backend delivery reliability.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

from test_iot_integration import EchoEngine

import echo_engine as engine_module


def _response(status_code, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    return response


class TestEngineBackendDelivery(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.config["API_URL"] = (
            "http://mock-backend/engine/event"
        )
        self.engine.config["API_TIMEOUT_SECONDS"] = 5
        self.engine.config["API_RETRY_COUNT"] = 2

        self.audio_event = {
            "sourceType": "simulator",
            "timestamp": "2026-09-10T10:00:00Z",
            "sensorId": "sensor-001",
            "microphoneLLA": {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
            "animalEstLLA": None,
            "animalTrueLLA": None,
            "animalLLAUncertainty": None,
            "audioClip": "base64-test-audio",
            "sampleRate": 16000,
        }

    def _send(self):
        self.engine.echo_api_send_detection_event(
            self.audio_event,
            32000,
            "Magpie",
            91.5,
        )

    def test_successful_http_201(self):
        with patch.object(
            engine_module.requests,
            "post",
            return_value=_response(201, "created"),
        ) as mock_post:
            self._send()

        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 5)

    def test_timeout_passed_into_requests_post(self):
        self.engine.config["API_TIMEOUT_SECONDS"] = 8

        with patch.object(
            engine_module.requests,
            "post",
            return_value=_response(201, "created"),
        ) as mock_post:
            self._send()

        mock_post.assert_called_once_with(
            "http://mock-backend/engine/event",
            json=mock_post.call_args.kwargs["json"],
            timeout=8,
        )

    def test_http_400_does_not_retry(self):
        with patch.object(
            engine_module.requests,
            "post",
            return_value=_response(400, "bad request"),
        ) as mock_post:
            with self.assertRaises(RuntimeError) as raised:
                self._send()

        self.assertEqual(mock_post.call_count, 1)
        self.assertIn("HTTP 400", str(raised.exception))
        self.assertIn("bad request", str(raised.exception))

    def test_http_500_retries_and_eventually_succeeds(self):
        with patch.object(
            engine_module.requests,
            "post",
            side_effect=[
                _response(500, "unavailable"),
                _response(201, "created"),
            ],
        ) as mock_post:
            self._send()

        self.assertEqual(mock_post.call_count, 2)

    def test_http_500_exhausts_retries_and_raises(self):
        with patch.object(
            engine_module.requests,
            "post",
            return_value=_response(500, "unavailable"),
        ) as mock_post:
            with self.assertRaises(RuntimeError) as raised:
                self._send()

        self.assertEqual(mock_post.call_count, 3)
        self.assertIn("after 3 attempts", str(raised.exception))
        self.assertIn("HTTP 500", str(raised.exception))
        self.assertIn("unavailable", str(raised.exception))

    def test_timeout_retries(self):
        with patch.object(
            engine_module.requests,
            "post",
            side_effect=[
                requests.exceptions.Timeout("timed out"),
                requests.exceptions.Timeout("timed out"),
                _response(201, "created"),
            ],
        ) as mock_post:
            self._send()

        self.assertEqual(mock_post.call_count, 3)

    def test_timeout_exhausts_retries_and_raises(self):
        with patch.object(
            engine_module.requests,
            "post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ) as mock_post:
            with self.assertRaises(RuntimeError) as raised:
                self._send()

        self.assertEqual(mock_post.call_count, 3)
        self.assertIn("timeout", str(raised.exception))
        self.assertIn("after 3 attempts", str(raised.exception))

    def test_connection_error_retries(self):
        with patch.object(
            engine_module.requests,
            "post",
            side_effect=[
                requests.exceptions.ConnectionError("refused"),
                _response(201, "created"),
            ],
        ) as mock_post:
            self._send()

        self.assertEqual(mock_post.call_count, 2)

    def test_connection_error_exhausts_retries_and_raises(self):
        with patch.object(
            engine_module.requests,
            "post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ) as mock_post:
            with self.assertRaises(RuntimeError) as raised:
                self._send()

        self.assertEqual(mock_post.call_count, 3)
        self.assertIn("connection failure", str(raised.exception))
        self.assertIn("after 3 attempts", str(raised.exception))

    def test_retry_count_is_bounded(self):
        self.engine.config["API_RETRY_COUNT"] = 2

        with patch.object(
            engine_module.requests,
            "post",
            return_value=_response(503, "unavailable"),
        ) as mock_post:
            with self.assertRaises(RuntimeError):
                self._send()

        self.assertEqual(mock_post.call_count, 3)

    def test_unexpected_error_is_not_retried(self):
        with patch.object(
            engine_module.requests,
            "post",
            side_effect=ValueError("unexpected"),
        ) as mock_post:
            with self.assertRaises(ValueError):
                self._send()

        self.assertEqual(mock_post.call_count, 1)

    def test_http_422_does_not_retry(self):
        with patch.object(
            engine_module.requests,
            "post",
            return_value=_response(422, "validation error"),
        ) as mock_post:
            with self.assertRaises(RuntimeError) as raised:
                self._send()

        self.assertEqual(mock_post.call_count, 1)
        self.assertIn("HTTP 422", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
