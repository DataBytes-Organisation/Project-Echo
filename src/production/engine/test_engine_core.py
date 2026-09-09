"""
Additional automated tests for core Echo Engine functions.

Run from src/production/engine:
    python -m pytest test_engine_core.py -v
"""

import base64
import sys
import unittest
from unittest.mock import MagicMock

from test_iot_integration import (
    EchoEngine,
    _make_msg,
    _valid_payload,
)


class TestCorePrediction(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.class_names = ["Kookaburra", "Magpie"]

    def test_prediction_returns_species_and_percentage(self):
        mock_tf = sys.modules["tensorflow"]

        mock_tf.argmax.return_value.numpy.return_value = 1

        mock_probability = MagicMock()
        mock_probability.numpy.return_value = 0.873

        mock_softmax = MagicMock()
        mock_softmax.__getitem__.return_value = mock_probability
        mock_tf.nn.softmax.return_value = mock_softmax

        species, confidence = self.engine.predict_class([0.1, 0.9])

        self.assertEqual(species, "Magpie")
        self.assertEqual(confidence, 87.3)

    def test_invalid_class_index_raises_error(self):
        mock_tf = sys.modules["tensorflow"]
        mock_tf.argmax.return_value.numpy.return_value = 10

        with self.assertRaises(IndexError):
            self.engine.predict_class([0.1, 0.9])


class TestAudioEncoding(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()

    def test_audio_base64_round_trip(self):
        original_audio = b"RIFF-test-audio-data"

        encoded_audio = self.engine.audio_to_string(original_audio)
        decoded_audio = self.engine.string_to_audio(encoded_audio)

        self.assertEqual(decoded_audio, original_audio)

    def test_recorded_audio_base64_round_trip(self):
        original_audio = b"small-recorded-audio-fixture"

        encoded_audio = base64.b64encode(
            original_audio
        ).decode("utf-8")

        decoded_audio = self.engine.recorded_string_to_audio(
            encoded_audio
        )

        self.assertEqual(decoded_audio, original_audio)


class TestEdgePredictionHandling(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.echo_api_send_detection_event = MagicMock()

    def test_valid_edge_prediction_forwarded(self):
        payload = {
            "type": "prediction",
            "species": "Magpie",
            "confidence": 91.5,
            "sensor_id": "edge-device-01",
            "timestamp": "1234567890",
            "gps_data": {
                "lat": -37.8136,
                "lon": 144.9631,
            },
            "gps_uncertainty": 4.0,
        }

        self.engine.on_iot_message(
            None,
            None,
            _make_msg(payload)
        )

        self.engine.echo_api_send_detection_event.assert_called_once()

        arguments = (
            self.engine.echo_api_send_detection_event
            .call_args[0]
        )

        audio_event = arguments[0]
        sample_rate = arguments[1]
        species = arguments[2]
        confidence = arguments[3]

        self.assertEqual(sample_rate, 0)
        self.assertEqual(species, "Magpie")
        self.assertEqual(confidence, 91.5)
        self.assertEqual(
            audio_event["sensorId"],
            "edge-device-01"
        )
        self.assertEqual(
            audio_event["microphoneLLA"],
            [-37.8136, 144.9631, 0.0]
        )

    def test_edge_prediction_without_gps_rejected(self):
        payload = {
            "type": "prediction",
            "species": "Magpie",
            "confidence": 91.5,
        }

        self.engine.on_iot_message(
            None,
            None,
            _make_msg(payload)
        )

        self.engine.echo_api_send_detection_event.assert_not_called()


class TestIoTNegativeScenarios(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.combined_pipeline = MagicMock()
        self.engine.echo_api_send_detection_event = MagicMock()

    def test_malformed_json_rejected(self):
        message = MagicMock()
        message.payload = b'{"audio_file":'

        self.engine.on_iot_message(None, None, message)

        self.engine.combined_pipeline.assert_not_called()
        self.engine.echo_api_send_detection_event.assert_not_called()

    def test_invalid_base64_audio_rejected(self):
        payload = _valid_payload(
            audio_file="not_base64"
        )

        self.engine.on_iot_message(
            None,
            None,
            _make_msg(payload)
        )

        self.engine.combined_pipeline.assert_not_called()
        self.engine.echo_api_send_detection_event.assert_not_called()

    def test_preprocessing_error_prevents_api_request(self):
        self.engine.combined_pipeline.side_effect = ValueError(
            "Invalid or corrupt WAV audio"
        )

        self.engine.on_iot_message(
            None,
            None,
            _make_msg(_valid_payload())
        )

        self.engine.echo_api_send_detection_event.assert_not_called()


if __name__ == "__main__":
    unittest.main()