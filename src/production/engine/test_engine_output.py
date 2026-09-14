"""
Automated tests for the Engine prediction output contract.
"""

import os
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
        self.engine.config["API_TIMEOUT_SECONDS"] = 5
        self.engine.config["API_RETRY_COUNT"] = 2
        self._clear_api_key = patch.dict(
            os.environ,
            {"ENGINE_API_KEY": ""},
        )
        self._clear_api_key.start()
        self.addCleanup(self._clear_api_key.stop)

        self.audio_event = {
            "sourceType": "simulator",
            "timestamp": "2026-08-06T10:30:00Z",
            "sensorId": "sensor-001",
            "microphoneLLA": {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
            "animalEstLLA": {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
            "animalTrueLLA": {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
            "animalLLAUncertainty": 5.0,
            "audioClip": "base64-test-audio",
        }

    def test_complete_prediction_payload_sent(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
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
            "sourceType": "simulator",
            "timestamp": "2026-08-06T10:30:00Z",
            "species": "Magpie",
            "confidence": 91.5,
            "sensorId": "sensor-001",
            "microphoneLLA": {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
            "animalEstLLA": {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
            "animalTrueLLA": {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
            "animalLLAUncertainty": 5.0,
            "audioClip": "base64-test-audio",
            "sampleRate": 48000,
            "source_model": "classic",
        }

        mock_post.assert_called_once_with(
            "http://mock-backend/engine/event",
            json=expected_payload,
            timeout=5,
        )

    def test_mqtt_sample_rate_used_when_provided(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "accepted"
        event_with_sample_rate = dict(self.audio_event)
        event_with_sample_rate["sampleRate"] = 16000
        event_with_sample_rate["sourceType"] = "real"

        with patch.object(
            engine_module.requests,
            "post",
            return_value=mock_response
        ) as mock_post:
            self.engine.echo_api_send_detection_event(
                event_with_sample_rate,
                32000,
                "Magpie",
                91.5,
            )

        posted_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(posted_payload["sampleRate"], 16000)
        self.assertEqual(posted_payload["sourceType"], "real")

    def test_inference_sample_rate_used_when_mqtt_omits_it(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "accepted"

        with patch.object(
            engine_module.requests,
            "post",
            return_value=mock_response
        ) as mock_post:
            self.engine.echo_api_send_detection_event(
                self.audio_event,
                32000,
                "Magpie",
                91.5,
            )

        posted_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(posted_payload["sampleRate"], 32000)

    def test_nullable_animal_location_fields_preserved(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "accepted"
        event_with_nulls = dict(self.audio_event)
        event_with_nulls["sourceType"] = "real"
        event_with_nulls["animalEstLLA"] = None
        event_with_nulls["animalTrueLLA"] = None
        event_with_nulls["animalLLAUncertainty"] = None
        event_with_nulls["sampleRate"] = 16000

        with patch.object(
            engine_module.requests,
            "post",
            return_value=mock_response
        ) as mock_post:
            self.engine.echo_api_send_detection_event(
                event_with_nulls,
                32000,
                "Magpie",
                91.5,
            )

        posted_payload = mock_post.call_args.kwargs["json"]
        self.assertIsNone(posted_payload["animalEstLLA"])
        self.assertIsNone(posted_payload["animalTrueLLA"])
        self.assertIsNone(posted_payload["animalLLAUncertainty"])
        self.assertEqual(posted_payload["sampleRate"], 16000)

    def test_backend_url_read_from_configuration(self):
        expected_url = (
            "http://different-backend/test/event"
        )
        self.engine.config["API_URL"] = expected_url

        mock_response = MagicMock()
        mock_response.status_code = 201
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

    def test_source_model_follows_active_inference_model(self):
        self.engine.config["ACTIVE_INFERENCE_MODEL"] = (
            "efficientnetv2_tflite"
        )
        mock_response = MagicMock()
        mock_response.status_code = 201
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

        posted_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            posted_payload["source_model"],
            "efficientnetv2_tflite",
        )

    def test_source_model_unknown_when_active_inference_model_missing_or_empty(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "accepted"

        for model_name in (None, ""):
            with self.subTest(ACTIVE_INFERENCE_MODEL=model_name):
                if model_name is None:
                    self.engine.config.pop("ACTIVE_INFERENCE_MODEL", None)
                else:
                    self.engine.config["ACTIVE_INFERENCE_MODEL"] = model_name

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

                posted_payload = mock_post.call_args.kwargs["json"]
                self.assertEqual(posted_payload["source_model"], "unknown")


if __name__ == "__main__":
    unittest.main()
