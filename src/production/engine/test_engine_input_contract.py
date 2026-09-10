"""
Tests for Engine MQTT input normalisation and standard-schema routing.
"""

import unittest
from unittest.mock import MagicMock, patch

from test_iot_integration import EchoEngine, _make_msg

import echo_engine as engine_module


STANDARD_LLA = {
    "latitude": -37.8136,
    "longitude": 144.9631,
    "altitude": 10.0,
}


def _standard_esp32_payload(**overrides):
    payload = {
        "sourceType": "real",
        "timestamp": "2026-09-10T10:00:00Z",
        "sensorId": "esp32-node-01",
        "microphoneLLA": dict(STANDARD_LLA),
        "animalEstLLA": None,
        "animalTrueLLA": None,
        "animalLLAUncertainty": None,
        "audioClip": "VGVzdCBhdWRpbw==",
        "sampleRate": 16000,
    }
    payload.update(overrides)
    return payload


def _legacy_simulator_payload(**overrides):
    payload = {
        "timestamp": "2026-09-10T10:00:00Z",
        "sensorId": "sensor-001",
        "microphoneLLA": [-37.8136, 144.9631, 0.0],
        "animalEstLLA": [-37.8136, 144.9631, 0.0],
        "animalTrueLLA": [-37.8136, 144.9631, 0.0],
        "animalLLAUncertainty": 5.0,
        "audioClip": "VGVzdCBhdWRpbw==",
        "mode": "Animal_Mode",
        "audioFile": "kookaburra/sample.wav",
    }
    payload.update(overrides)
    return payload


class TestMqttInputNormalisation(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()

    def test_standard_schema_accepted_unchanged(self):
        payload = _standard_esp32_payload()
        normalised = self.engine.normalise_mqtt_audio_event(payload)

        self.assertEqual(normalised["sourceType"], "real")
        self.assertEqual(normalised["timestamp"], payload["timestamp"])
        self.assertEqual(normalised["sensorId"], payload["sensorId"])
        self.assertEqual(normalised["audioClip"], payload["audioClip"])
        self.assertEqual(normalised["sampleRate"], 16000)
        self.assertIsNone(normalised["animalEstLLA"])
        self.assertIsNone(normalised["animalTrueLLA"])
        self.assertIsNone(normalised["animalLLAUncertainty"])
        self.assertEqual(
            normalised["microphoneLLA"],
            STANDARD_LLA,
        )
        self.assertNotIn("audioFile", normalised)
        self.assertNotIn("mode", normalised)

    def test_legacy_array_lla_converted_to_objects(self):
        normalised = self.engine.normalise_mqtt_audio_event(
            _legacy_simulator_payload()
        )

        expected_lla = {
            "latitude": -37.8136,
            "longitude": 144.9631,
            "altitude": 0.0,
        }
        self.assertEqual(normalised["microphoneLLA"], expected_lla)
        self.assertEqual(normalised["animalEstLLA"], expected_lla)
        self.assertEqual(normalised["animalTrueLLA"], expected_lla)

    def test_legacy_simulator_defaults_source_type(self):
        normalised = self.engine.normalise_mqtt_audio_event(
            _legacy_simulator_payload()
        )

        self.assertEqual(normalised["sourceType"], "simulator")
        self.assertEqual(normalised["mode"], "Animal_Mode")
        self.assertEqual(
            normalised["audioFile"],
            "kookaburra/sample.wav",
        )
        self.assertNotIn("sampleRate", normalised)

    def test_provided_source_type_is_preserved_on_legacy_shape(self):
        normalised = self.engine.normalise_mqtt_audio_event(
            _legacy_simulator_payload(sourceType="real")
        )

        self.assertEqual(normalised["sourceType"], "real")

    def test_malformed_lla_array_length_raises_controlled_error(self):
        payload = _legacy_simulator_payload(
            microphoneLLA=[-37.8136, 144.9631]
        )

        with self.assertRaises(ValueError) as raised:
            self.engine.normalise_mqtt_audio_event(payload)

        self.assertIn("microphoneLLA", str(raised.exception))
        self.assertIn("Malformed LLA", str(raised.exception))

    def test_malformed_lla_object_missing_keys_raises_controlled_error(self):
        payload = _standard_esp32_payload(
            microphoneLLA={"latitude": -37.8136, "longitude": 144.9631}
        )

        with self.assertRaises(ValueError) as raised:
            self.engine.normalise_mqtt_audio_event(payload)

        self.assertIn("microphoneLLA", str(raised.exception))
        self.assertIn("Malformed LLA", str(raised.exception))

    def test_null_microphone_lla_rejected(self):
        payload = _standard_esp32_payload(microphoneLLA=None)

        with self.assertRaises(ValueError) as raised:
            self.engine.normalise_mqtt_audio_event(payload)

        self.assertIn("microphoneLLA", str(raised.exception))


class TestOnMessageContractRouting(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.string_to_audio = MagicMock(return_value=b"audio")
        self.engine.efficientnetv2_tflite_predict_from_audio_bytes = (
            MagicMock(
                return_value=(
                    "Magpie",
                    91.5,
                    None,
                    32000,
                    [],
                )
            )
        )
        self.engine.echo_api_send_detection_event = MagicMock()

    def test_standard_payload_without_audiofile_uses_efficientnet(self):
        self.engine.on_message(
            None,
            None,
            _make_msg(_standard_esp32_payload()),
        )

        self.engine.efficientnetv2_tflite_predict_from_audio_bytes.assert_called_once()
        self.engine.echo_api_send_detection_event.assert_called_once()

        audio_event = (
            self.engine.echo_api_send_detection_event.call_args[0][0]
        )
        self.assertEqual(audio_event["sourceType"], "real")
        self.assertEqual(audio_event["sampleRate"], 16000)
        self.assertEqual(
            audio_event["microphoneLLA"]["latitude"],
            -37.8136,
        )
        self.assertIsNone(audio_event["animalEstLLA"])
        self.assertNotIn("audioFile", audio_event)

    def test_legacy_simulator_payload_uses_efficientnet(self):
        self.engine.on_message(
            None,
            None,
            _make_msg(_legacy_simulator_payload()),
        )

        self.engine.efficientnetv2_tflite_predict_from_audio_bytes.assert_called_once()
        audio_event = (
            self.engine.echo_api_send_detection_event.call_args[0][0]
        )
        self.assertEqual(audio_event["sourceType"], "simulator")
        self.assertEqual(
            audio_event["audioFile"],
            "kookaburra/sample.wav",
        )
        self.assertEqual(
            audio_event["microphoneLLA"],
            {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
        )

    def test_recording_mode_still_uses_efficientnet(self):
        self.engine.on_message(
            None,
            None,
            _make_msg(
                _legacy_simulator_payload(
                    audioFile="Recording_Mode",
                    mode="Recording_Mode",
                )
            ),
        )

        self.engine.efficientnetv2_tflite_predict_from_audio_bytes.assert_called_once()
        audio_event = (
            self.engine.echo_api_send_detection_event.call_args[0][0]
        )
        self.assertEqual(audio_event["sourceType"], "simulator")
        self.assertEqual(audio_event["audioFile"], "Recording_Mode")

    def test_malformed_lla_does_not_call_inference(self):
        self.engine.on_message(
            None,
            None,
            _make_msg(
                _legacy_simulator_payload(
                    microphoneLLA=[-37.8136]
                )
            ),
        )

        self.engine.efficientnetv2_tflite_predict_from_audio_bytes.assert_not_called()
        self.engine.echo_api_send_detection_event.assert_not_called()


class TestOnMessagePostsStandardEvent(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.config["API_URL"] = (
            "http://mock-backend/engine/event"
        )
        self.engine.string_to_audio = MagicMock(return_value=b"audio")
        self.engine.efficientnetv2_tflite_predict_from_audio_bytes = (
            MagicMock(
                return_value=(
                    "Magpie",
                    91.5,
                    None,
                    32000,
                    [],
                )
            )
        )

    def test_esp32_sample_rate_and_source_type_reach_backend_payload(self):
        mock_response = MagicMock()
        mock_response.text = "accepted"

        with patch.object(
            engine_module.requests,
            "post",
            return_value=mock_response,
        ) as mock_post:
            self.engine.on_message(
                None,
                None,
                _make_msg(_standard_esp32_payload()),
            )

        posted_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(posted_payload["sourceType"], "real")
        self.assertEqual(posted_payload["sampleRate"], 16000)
        self.assertEqual(posted_payload["species"], "Magpie")
        self.assertEqual(posted_payload["confidence"], 91.5)
        self.assertIsNone(posted_payload["animalEstLLA"])
        self.assertEqual(
            posted_payload["microphoneLLA"]["longitude"],
            144.9631,
        )

    def test_legacy_simulator_uses_inference_sample_rate(self):
        mock_response = MagicMock()
        mock_response.text = "accepted"

        with patch.object(
            engine_module.requests,
            "post",
            return_value=mock_response,
        ) as mock_post:
            self.engine.on_message(
                None,
                None,
                _make_msg(_legacy_simulator_payload()),
            )

        posted_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(posted_payload["sourceType"], "simulator")
        self.assertEqual(posted_payload["sampleRate"], 32000)
        self.assertEqual(
            posted_payload["animalEstLLA"],
            {
                "latitude": -37.8136,
                "longitude": 144.9631,
                "altitude": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
