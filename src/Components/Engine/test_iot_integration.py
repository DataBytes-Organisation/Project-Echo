"""
Unit tests for IoT MQTT integration in echo_engine.py

Mocks heavy dependencies (TensorFlow, librosa, MQTT, GCP, etc.) so tests
run without the full Docker stack or GPU.

Run from src/Components/Engine/:
    python -m pytest test_iot_integration.py -v
"""

import sys
import os
import io
import json
import base64
import pickle
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ---------------------------------------------------------------------------
# Mock heavy third-party modules BEFORE importing echo_engine
# ---------------------------------------------------------------------------
_HEAVY_MODULES = [
    "tensorflow", "tensorflow.keras", "tensorflow.keras.models",
    "librosa", "librosa.feature",
    "paho", "paho.mqtt", "paho.mqtt.client",
    "google.cloud", "google.cloud.storage",
    "pymongo", "diskcache", "soundfile",
    "geopy", "geopy.distance",
    "sklearn", "sklearn.preprocessing",
    "helpers", "helpers.melspectrogram_to_cam",
    "yamnet_dir", "yamnet_dir.params", "yamnet_dir.yamnet",
]

for mod in _HEAVY_MODULES:
    sys.modules[mod] = MagicMock()

sys.modules["tensorflow"].__version__ = "2.15.0"
sys.modules["librosa"].__version__ = "0.10.0"
sys.modules["pymongo"].MongoClient.return_value.list_database_names.return_value = []

# ---------------------------------------------------------------------------
# Patch builtins.open and pickle.load for module-level file reads
# ---------------------------------------------------------------------------
import builtins
_real_open = builtins.open

_ENGINE_CONFIG = {
    "AUDIO_SAMPLE_RATE": 48000, "AUDIO_CLIP_DURATION": 5,
    "AUDIO_NFFT": 2048, "AUDIO_STRIDE": 200, "AUDIO_MELS": 260,
    "AUDIO_FMIN": 20, "AUDIO_FMAX": 13000, "AUDIO_TOP_DB": 80,
    "AUDIO_WINDOW": 500, "WEATHER_SAMPLE_RATE": 16000, "WEATHER_CLIP_DURATION": 2,
    "MODEL_INPUT_IMAGE_WIDTH": 260, "MODEL_INPUT_IMAGE_HEIGHT": 260,
    "MODEL_INPUT_IMAGE_CHANNELS": 3,
    "MQTT_CLIENT_URL": "localhost", "MQTT_CLIENT_PORT": 1883,
    "MQTT_PUBLISH_URL": "projectecho/engine/2",
    "MODEL_SERVER": "http://localhost:8501/v1/models/echo_model:predict",
    "WEATHER_SERVER": "http://localhost:8501/v1/models/weather_model:predict",
    "GCLOUD_PROJECT": "test", "BUCKET_NAME": "test", "DB_HOSTNAME": "localhost",
    "IOT_MQTT_BROKER": "broker.hivemq.com",
    "IOT_MQTT_PORT": 1883,
    "IOT_MQTT_TOPIC": "iot/data/test",
}

_ENGINE_CREDS = {"DB_USERNAME": "test", "DB_PASSWORD": "test"}


def _patched_open(path, *args, **kwargs):
    s = str(path)
    if "class_names.pkl" in s or "label_encoder.pkl" in s:
        return io.BytesIO(pickle.dumps(["species_a", "species_b"]))
    if "echo_engine.json" in s:
        return io.BytesIO(json.dumps(_ENGINE_CONFIG).encode())
    if "echo_credentials.json" in s:
        return io.BytesIO(json.dumps(_ENGINE_CREDS).encode())
    return _real_open(path, *args, **kwargs)


builtins.open = _patched_open
from echo_engine_iot import EchoEngine  # noqa: E402
builtins.open = _real_open


# ===========================================================================
# Helpers
# ===========================================================================

def _make_msg(payload: dict) -> MagicMock:
    """Create a mock MQTT message with a JSON payload."""
    msg = MagicMock()
    msg.payload = json.dumps(payload).encode()
    return msg


def _valid_payload(**overrides) -> dict:
    """Return a valid IoT payload, with optional overrides."""
    payload = {
        "audio_file": base64.b64encode(b"\x00" * 1000).decode(),
        "gps_data": {"lat": -37.8136, "lon": 144.9631},
        "sensor_id": "rpi_node_3",
        "gps_uncertainty": 5.0,
    }
    payload.update(overrides)
    return payload


# ===========================================================================
# Tests: payload validation
# ===========================================================================

class TestIoTPayloadValidation(unittest.TestCase):
    """Verify the handler rejects bad payloads before hitting the ML pipeline."""

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.class_names = ["Kookaburra", "Magpie"]
        self.engine.combined_pipeline = MagicMock()
        self.engine.predict_class = MagicMock()
        self.engine.echo_api_send_detection_event = MagicMock()
        self.engine.audio_to_string = MagicMock(return_value="base64audio")

    def test_missing_audio_file_rejected(self):
        payload = {"gps_data": {"lat": -37.8, "lon": 145.0}}
        self.engine.on_iot_message(None, None, _make_msg(payload))
        self.engine.combined_pipeline.assert_not_called()

    def test_empty_audio_file_rejected(self):
        payload = _valid_payload(audio_file="")
        self.engine.on_iot_message(None, None, _make_msg(payload))
        self.engine.combined_pipeline.assert_not_called()

    def test_missing_gps_data_rejected(self):
        payload = {"audio_file": base64.b64encode(b"\x00").decode()}
        self.engine.on_iot_message(None, None, _make_msg(payload))
        self.engine.combined_pipeline.assert_not_called()

    def test_missing_lat_rejected(self):
        payload = _valid_payload()
        payload["gps_data"] = {"lon": 145.0}
        self.engine.on_iot_message(None, None, _make_msg(payload))
        self.engine.combined_pipeline.assert_not_called()

    def test_missing_lon_rejected(self):
        payload = _valid_payload()
        payload["gps_data"] = {"lat": -37.8}
        self.engine.on_iot_message(None, None, _make_msg(payload))
        self.engine.combined_pipeline.assert_not_called()


# ===========================================================================
# Tests: happy-path handler logic
# ===========================================================================

class TestIoTMessageHandler(unittest.TestCase):
    """Verify the handler correctly calls the pipeline and sends events."""

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.class_names = ["Kookaburra", "Magpie"]

        self.engine.combined_pipeline = MagicMock(return_value=(
            np.zeros((260, 260, 3)),
            np.zeros(48000 * 5, dtype=np.float32),
            48000,
        ))
        self.engine.predict_class = MagicMock(return_value=("Kookaburra", 95.5))
        self.engine.echo_api_send_detection_event = MagicMock()
        self.engine.audio_to_string = MagicMock(return_value="base64audio")

        # tf.expand_dims returns a mock tensor with .numpy().tolist()
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value.tolist.return_value = [[[[0.0] * 3] * 260] * 260]
        sys.modules["tensorflow"].expand_dims.return_value = mock_tensor

        # requests.post to model server
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({"outputs": [[0.1, 0.9]]})
        self.requests_patcher = patch("echo_engine.requests.post", return_value=mock_resp)
        self.mock_post = self.requests_patcher.start()

    def tearDown(self):
        self.requests_patcher.stop()

    def test_valid_payload_calls_pipeline(self):
        self.engine.on_iot_message(None, None, _make_msg(_valid_payload()))
        self.engine.combined_pipeline.assert_called_once()

    def test_valid_payload_calls_api(self):
        self.engine.on_iot_message(None, None, _make_msg(_valid_payload()))
        self.engine.echo_api_send_detection_event.assert_called_once()

    def test_sensor_id_from_payload(self):
        self.engine.on_iot_message(None, None, _make_msg(_valid_payload(sensor_id="rpi_node_3")))
        event = self.engine.echo_api_send_detection_event.call_args[0][0]
        self.assertEqual(event["sensorId"], "rpi_node_3")

    def test_default_sensor_id(self):
        payload = _valid_payload()
        del payload["sensor_id"]
        self.engine.on_iot_message(None, None, _make_msg(payload))
        event = self.engine.echo_api_send_detection_event.call_args[0][0]
        self.assertEqual(event["sensorId"], "unknown_iot_sensor")

    def test_gps_coordinates_in_event(self):
        self.engine.on_iot_message(None, None, _make_msg(_valid_payload()))
        event = self.engine.echo_api_send_detection_event.call_args[0][0]
        self.assertEqual(event["microphoneLLA"], [-37.8136, 144.9631, 0.0])
        self.assertEqual(event["animalEstLLA"], [-37.8136, 144.9631, 0.0])
        self.assertEqual(event["animalTrueLLA"], [-37.8136, 144.9631, 0.0])

    def test_gps_uncertainty_from_payload(self):
        self.engine.on_iot_message(None, None, _make_msg(_valid_payload(gps_uncertainty=5.0)))
        event = self.engine.echo_api_send_detection_event.call_args[0][0]
        self.assertEqual(event["animalLLAUncertainty"], 5.0)

    def test_default_gps_uncertainty(self):
        payload = _valid_payload()
        del payload["gps_uncertainty"]
        self.engine.on_iot_message(None, None, _make_msg(payload))
        event = self.engine.echo_api_send_detection_event.call_args[0][0]
        self.assertEqual(event["animalLLAUncertainty"], 10.0)

    def test_species_and_confidence_forwarded(self):
        self.engine.predict_class.return_value = ("Magpie", 87.3)
        self.engine.on_iot_message(None, None, _make_msg(_valid_payload()))
        args = self.engine.echo_api_send_detection_event.call_args[0]
        self.assertEqual(args[2], "Magpie")
        self.assertEqual(args[3], 87.3)

    def test_model_server_url_from_config(self):
        self.engine.on_iot_message(None, None, _make_msg(_valid_payload()))
        post_url = self.mock_post.call_args[0][0]
        self.assertEqual(post_url, _ENGINE_CONFIG["MODEL_SERVER"])


# ===========================================================================
# Tests: config and startup order
# ===========================================================================

class TestIoTConfig(unittest.TestCase):
    """Verify IoT config keys are read from echo_engine.json."""

    def test_config_has_iot_keys(self):
        engine = EchoEngine()
        self.assertIn("IOT_MQTT_BROKER", engine.config)
        self.assertIn("IOT_MQTT_PORT", engine.config)
        self.assertIn("IOT_MQTT_TOPIC", engine.config)


class TestIoTStartupOrder(unittest.TestCase):
    """Verify IoT listener starts AFTER class_names are loaded (race condition fix)."""

    def test_iot_listener_starts_after_class_names(self):
        engine = EchoEngine()
        call_order = []

        engine.gcp_load_species_list = MagicMock(
            side_effect=lambda: (call_order.append("class_names"), ["species"])[1]
        )
        engine.start_iot_mqtt_listener = MagicMock(
            side_effect=lambda: call_order.append("iot_listener")
        )

        # Mocked paho.Client().loop_forever() returns immediately
        engine.execute()

        self.assertEqual(call_order, ["class_names", "iot_listener"])


if __name__ == "__main__":
    unittest.main()
