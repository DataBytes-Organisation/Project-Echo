"""
Automated tests for the Engine prediction output contract.
"""

import unittest
from unittest.mock import MagicMock, patch

from test_iot_integration import EchoEngine

import echo_engine as engine_module


class TestEnginePredictionOutput(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.config["API_URL"] = (
            "http://mock-backend/engine/event"
        )

        self.audio_event = {
            "timestamp": "2026-08-06T10:30:00Z",
            "sensorId": "sensor-001",
            "microphoneLLA": [
                -37.8136,
                144.9631,
                0.0,
            ],
            "animalEstLLA": [
                -37.8136,
                144.9631,
                0.0,
            ],
            "animalTrueLLA": [
                -37.8136,
                144.9631,
                0.0,
            ],
            "animalLLAUncertainty": 5.0,
            "audioClip": "base64-test-audio",
        }

    def test_complete_prediction_payload_sent(self):
        mock_response = MagicMock()
        mock_response.text = "accepted"

        with patch.object(
            engine_module.requests,
            "post",
            return_value=mock_response
        ) as mock_post:
            self.engine.echo_api_send_detection_event(
                self.audio_event,
                48000,
                "Magpie",
                91.5,
            )

        expected_payload = {
            "timestamp": "2026-08-06T10:30:00Z",
            "species": "Magpie",
            "confidence": 91.5,
            "sensorId": "sensor-001",
            "microphoneLLA": [
                -37.8136,
                144.9631,
                0.0,
            ],
            "animalEstLLA": [
                -37.8136,
                144.9631,
                0.0,
            ],
            "animalTrueLLA": [
                -37.8136,
                144.9631,
                0.0,
            ],
            "animalLLAUncertainty": 5.0,
            "audioClip": "base64-test-audio",
            "sampleRate": 48000,
        }

        mock_post.assert_called_once_with(
            "http://mock-backend/engine/event",
            json=expected_payload
        )

    def test_backend_url_read_from_configuration(self):
        expected_url = (
            "http://different-backend/test/event"
        )
        self.engine.config["API_URL"] = expected_url

        mock_response = MagicMock()
        mock_response.text = "accepted"

        with patch.object(
            engine_module.requests,
            "post",
            return_value=mock_response
        ) as mock_post:
            self.engine.echo_api_send_detection_event(
                self.audio_event,
                48000,
                "Kookaburra",
                87.2,
            )

        actual_url = mock_post.call_args.args[0]

        self.assertEqual(actual_url, expected_url)

    def test_missing_required_output_field_rejected(self):
        invalid_event = self.audio_event.copy()
        del invalid_event["timestamp"]

        with patch.object(
            engine_module.requests,
            "post"
        ) as mock_post:
            with self.assertRaises(KeyError):
                self.engine.echo_api_send_detection_event(
                    invalid_event,
                    48000,
                    "Magpie",
                    91.5,
                )

        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()