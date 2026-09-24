"""Sprint 2: offline Engine integration tests (Python 3.9+).

Place beside echo_engine.py, then run:
    python -m pytest test_engine_end_to_end.py -v --tb=short

Only NumPy and the Python standard library are required by this file.
It also runs with: python -m unittest test_engine_end_to_end -v

REAL: uploaded production handlers, Base64 decoding, padding, normalisation,
tensor layout, inference orchestration, softmax, class mapping and API payload
construction. STUBBED: librosa's DSP operations, TFLite runtime and HTTP client.
The WAV decoder double reads genuine generated PCM WAV bytes using wave.
It does not implement resampling or validate librosa's numerical behaviour.

These are mocked-boundary integration tests, NOT live end-to-end validation,
model-accuracy tests, or proof that the Backend accepts the payload. Constructor,
model loading, MQTT transport, Recording_Mode_V2, authentication and individual
HTTP-status rules remain covered elsewhere. This suite does exercise timeout
retry exhaustion through the complete handler and verifies recovery on the next
message. No production source, configuration, or existing test is changed.
"""

import base64
import importlib.util
import io
import json
import os
import pickle
import sys
import unittest
import warnings
import wave
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np


SAMPLE_RATE = 32000
BACKEND_URL = "https://backend.invalid/test/detections"


def generate_wav():
    """Small deterministic mono PCM16 recording, held entirely in memory."""
    time_points = np.arange(1600) / SAMPLE_RATE
    pcm = (0.25 * np.sin(2 * np.pi * 440 * time_points) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(SAMPLE_RATE)
        recording.writeframes(pcm.tobytes())
    return buffer.getvalue(), pcm.astype(np.float32) / 32768.0


def make_message(payload):
    """Match the MQTT callback's message.payload interface without a broker."""
    return SimpleNamespace(payload=json.dumps(payload).encode("utf-8"))


def load_isolated_engine():
    """Load actual production source with scoped dependency replacements.

    Unlike the legacy bootstrap, this never installs a permanent builtins.open
    replacement or leaves its dependency doubles in sys.modules. A private
    module avoids changing the echo_engine object used by the old test suite.
    """
    source = Path(__file__).resolve().with_name("echo_engine.py")
    if not source.is_file():
        raise FileNotFoundError("Place test_engine_end_to_end.py beside echo_engine.py")

    dependency_names = (
        "tensorflow", "tensorflow.keras", "tensorflow.keras.models",
        "librosa", "librosa.feature", "pandas", "soundfile", "diskcache",
        "paho", "paho.mqtt", "paho.mqtt.client",
        "google", "google.cloud", "google.cloud.storage", "pymongo",
        "geopy", "geopy.distance", "sklearn", "sklearn.preprocessing",
        "helpers", "helpers.melspectrogram_to_cam",
        "yamnet_dir", "yamnet_dir.params", "yamnet_dir.yamnet",
    )
    dependencies = {name: MagicMock(name=name) for name in dependency_names}
    # Keep parent imports and dotted imports connected to the same doubles.
    for name, dependency in dependencies.items():
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(dependencies[parent], attribute, dependency)
    dependencies["tensorflow"].__version__ = "test-double"
    dependencies["librosa"].__version__ = "test-double"

    http = ModuleType("requests")
    http.post = Mock(side_effect=AssertionError("Unexpected HTTP boundary call"))

    class RequestException(Exception):
        """Base request exception used by the isolated HTTP test double."""

    class RequestTimeout(RequestException):
        """Timeout raised by the isolated HTTP test double."""

    class RequestConnectionError(RequestException):
        """Connection failure raised by the isolated HTTP test double."""

    http.exceptions = SimpleNamespace(
        RequestException=RequestException,
        Timeout=RequestTimeout,
        ConnectionError=RequestConnectionError,
    )
    dependencies["requests"] = http

    spec = importlib.util.spec_from_file_location("_sprint2_engine_under_test", source)
    module = importlib.util.module_from_spec(spec)

    def asset_open(path, *args, **kwargs):
        if Path(path).name in {"class_names.pkl", "label_encoder.pkl"}:
            return io.BytesIO(pickle.dumps(["Kookaburra", "Magpie"]))
        raise AssertionError("Unexpected Engine file access: " + str(path))

    # A module-local open shadows the builtin only in this Engine instance.
    module.open = asset_open
    with patch.dict(sys.modules, dependencies), patch.dict(os.environ):
        with patch.object(sys, "path", list(sys.path)), warnings.catch_warnings():
            with redirect_stdout(io.StringIO()):
                spec.loader.exec_module(module)
    return module


class TestEngineEndToEnd(unittest.TestCase):
    """Connect real Engine methods; substitute only the selected boundaries."""

    def setUp(self):
        self.module = load_isolated_engine()
        # Explicitly bypass startup; no credentials, database, or model assets.
        self.engine = self.module.EchoEngine.__new__(self.module.EchoEngine)
        self.engine.config = {
            "API_URL": BACKEND_URL,
            "API_TIMEOUT_SECONDS": 5,
            "API_RETRY_COUNT": 2,
            "ACTIVE_INFERENCE_MODEL": "efficientnetv2",
        }
        clear_api_key = patch.dict(os.environ, {"ENGINE_API_KEY": ""})
        clear_api_key.start()
        self.addCleanup(clear_api_key.stop)
        self.engine.eff_preprocess_config = {
            "target_sr": SAMPLE_RATE, "duration_s": 5.0,
            "n_mels": 128, "hop_length": 512, "fmin": 20, "fmax": 16000,
        }
        self.engine.eff_input_details = [{
            "index": 0, "shape": np.array([1, 1, 128, 313]), "dtype": np.float32,
        }]
        self.engine.eff_output_details = [{"index": 1}]
        self.engine.eff_class_mapping = {
            "index_to_label": {"0": "Kookaburra", "1": "Magpie"},
        }
        self.wav_bytes, self.decoded_audio = generate_wav()
        self.audio_base64 = base64.b64encode(self.wav_bytes).decode("ascii")
        self.steps = []
        self.trace = io.StringIO()
        self.addCleanup(self.trace.close)
        capture = redirect_stdout(self.trace)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

        def decode_pcm(file_object, sr, mono):
            self.steps.append("decode_audio")
            self.assertEqual(file_object.getvalue(), self.wav_bytes)
            self.assertEqual((sr, mono), (SAMPLE_RATE, True))
            with wave.open(file_object, "rb") as recording:
                self.assertEqual(recording.getframerate(), sr)
                self.assertEqual(recording.getnchannels(), 1)
                self.assertEqual(recording.getsampwidth(), 2)
                pcm = recording.readframes(recording.getnframes())
            return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, sr

        # Deliberately synthetic DSP output: tests Engine wiring, not librosa.
        self.synthetic_mel = np.linspace(1, 10, 128 * 313, dtype=np.float32).reshape(128, 313)

        def mel_spectrogram(**kwargs):
            self.steps.append("mel_spectrogram")
            self.assertEqual(kwargs["sr"], SAMPLE_RATE)
            self.assertEqual(kwargs["n_mels"], 128)
            self.assertEqual(kwargs["hop_length"], 512)
            self.assertEqual((kwargs["fmin"], kwargs["fmax"]), (20.0, 16000.0))
            self.assertEqual(kwargs["y"].dtype, np.float32)
            self.assertEqual(kwargs["y"].shape, (SAMPLE_RATE * 5,))
            np.testing.assert_array_equal(kwargs["y"][:1600], self.decoded_audio)
            np.testing.assert_array_equal(kwargs["y"][1600:], 0.0)
            return self.synthetic_mel.copy()

        self.synthetic_db = np.linspace(-80, 0, 128 * 313, dtype=np.float32).reshape(128, 313)

        def power_to_db(mel, ref):
            self.steps.append("power_to_db")
            np.testing.assert_array_equal(mel, self.synthetic_mel)
            self.assertIs(ref, np.max)
            return self.synthetic_db.copy()

        self.module.librosa.load.side_effect = decode_pcm
        self.module.librosa.feature.melspectrogram.side_effect = mel_spectrogram
        self.module.librosa.power_to_db.side_effect = power_to_db

        interpreter = Mock(name="tflite_runtime")
        interpreter.set_tensor.side_effect = lambda *args: self.steps.append("set_tensor")
        interpreter.invoke.side_effect = lambda: self.steps.append("invoke")

        def model_output(index):
            self.steps.append("get_tensor")
            self.assertEqual(index, 1)
            # Logits giving an independently known 20%/80% probability split.
            return np.array([[0.0, np.log(4.0)]], dtype=np.float32)

        interpreter.get_tensor.side_effect = model_output
        self.engine.eff_interpreter = interpreter

        def post(url, **kwargs):
            self.steps.append("backend_post")
            self.assertEqual(url, BACKEND_URL)
            self.assertIn("json", kwargs)
            return SimpleNamespace(status_code=201, text="fixture accepted")

        self.module.requests.post.side_effect = post

    def standard_payload(self, source_type):
        return {
            "sourceType": source_type,
            "timestamp": "2026-08-30T12:00:00Z",
            "sensorId": "sprint2-fixture-01",
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
            "audioClip": self.audio_base64,
            "sampleRate": SAMPLE_RATE,
        }

    def legacy_recording_payload(self):
        return {
            "audioFile": "Recording_Mode",
            "mode": "Recording_Mode",
            "audioClip": self.audio_base64,
            "sensorId": "sprint2-fixture-01",
            "timestamp": "2026-08-30T12:00:00Z",
            "microphoneLLA": [-37.8136, 144.9631, 0.0],
            "animalEstLLA": [-37.8136, 144.9631, 0.0],
            "animalTrueLLA": [-37.8136, 144.9631, 0.0],
            "animalLLAUncertainty": 5.0,
        }

    def send_standard(self, source_type="real"):
        self.engine.on_message(
            None,
            None,
            make_message(self.standard_payload(source_type)),
        )

    def assert_complete_flow(self, expected_source_type):
        """Verify the full chain and exact final HTTP-boundary payload."""
        self.assertEqual(self.steps, [
            "decode_audio", "mel_spectrogram", "power_to_db",
            "set_tensor", "invoke", "get_tensor", "backend_post",
        ])
        runtime = self.engine.eff_interpreter
        runtime.set_tensor.assert_called_once()
        runtime.invoke.assert_called_once_with()
        runtime.get_tensor.assert_called_once_with(1)
        tensor_index, tensor = runtime.set_tensor.call_args.args
        self.assertEqual(tensor_index, 0)
        self.assertEqual(tensor.shape, (1, 1, 128, 313))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())
        self.assertAlmostEqual(float(tensor.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(tensor.std()), 1.0, places=5)
        expected_tensor = (self.synthetic_db - float(self.synthetic_db.mean())) / (
            float(self.synthetic_db.std()) + 1e-6
        )
        np.testing.assert_allclose(tensor[0, 0], expected_tensor, rtol=1e-6, atol=1e-6)

        expected_payload = {
            "sourceType": expected_source_type,
            "timestamp": "2026-08-30T12:00:00Z",
            "sensorId": "sprint2-fixture-01",
            "species": "Magpie",
            "confidence": 80.0,
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
            "audioClip": self.audio_base64,
            "sampleRate": SAMPLE_RATE,
            "source_model": "efficientnetv2",
        }
        self.module.requests.post.assert_called_once_with(
            BACKEND_URL,
            json=expected_payload,
            timeout=5,
            headers=None,
        )

    def test_standard_real_audio_reaches_backend_through_real_engine_methods(self):
        self.send_standard("real")
        self.assert_complete_flow("real")

    def test_standard_simulator_audio_reaches_backend_through_real_engine_methods(self):
        self.send_standard("simulator")
        self.assert_complete_flow("simulator")

    def test_legacy_recording_mode_reaches_backend_after_normalisation(self):
        message = make_message(self.legacy_recording_payload())
        self.engine.on_message(None, None, message)
        self.assert_complete_flow("simulator")

    def test_current_preprocessing_failure_stops_inference_and_backend(self):
        self.module.librosa.load.side_effect = ValueError("fixture decode failed")
        self.send_standard()
        self.module.librosa.load.assert_called_once()
        self.engine.eff_interpreter.set_tensor.assert_not_called()
        self.engine.eff_interpreter.invoke.assert_not_called()
        self.module.requests.post.assert_not_called()
        self.assertIn("An error occurred: fixture decode failed", self.trace.getvalue())

    def test_local_inference_failure_stops_backend_submission(self):
        self.engine.eff_interpreter.invoke.side_effect = RuntimeError("fixture invoke failed")
        self.send_standard()
        self.engine.eff_interpreter.set_tensor.assert_called_once()
        self.engine.eff_interpreter.invoke.assert_called_once_with()
        self.engine.eff_interpreter.get_tensor.assert_not_called()
        self.module.requests.post.assert_not_called()
        self.assertIn("An error occurred: fixture invoke failed", self.trace.getvalue())

    def test_invalid_model_output_shape_stops_backend_submission(self):
        self.engine.eff_interpreter.get_tensor.side_effect = None
        self.engine.eff_interpreter.get_tensor.return_value = np.array([0, 1], dtype=np.float32)
        self.send_standard()
        self.engine.eff_interpreter.get_tensor.assert_called_once_with(1)
        self.module.requests.post.assert_not_called()
        self.assertIn("Unexpected EfficientNetV2 output shape", self.trace.getvalue())

    def test_missing_winning_class_label_stops_backend_submission(self):
        del self.engine.eff_class_mapping["index_to_label"]["1"]
        self.send_standard()
        self.engine.eff_interpreter.get_tensor.assert_called_once_with(1)
        self.module.requests.post.assert_not_called()
        self.assertIn("No species label exists for model output index 1", self.trace.getvalue())

    def test_backend_exception_is_caught_and_next_message_can_run(self):
        timeout_error = self.module.requests.exceptions.Timeout(
            "fixture backend timed out"
        )
        self.module.requests.post.side_effect = timeout_error
        self.send_standard()
        self.engine.eff_interpreter.invoke.assert_called_once_with()
        self.assertEqual(self.module.requests.post.call_count, 3)
        self.assertIn(
            "An error occurred: Backend delivery failed after 3 attempts: timeout",
            self.trace.getvalue(),
        )

        self.module.requests.post.side_effect = None
        self.module.requests.post.return_value = SimpleNamespace(status_code=201, text="accepted")
        self.send_standard()
        self.assertEqual(self.module.requests.post.call_count, 4)
        self.assertEqual(self.engine.eff_interpreter.invoke.call_count, 2)
        self.assertEqual(self.module.requests.post.call_args.kwargs["json"]["species"], "Magpie")
        self.assertEqual(self.trace.getvalue().count("An error occurred:"), 1)
        self.assertIn("accepted", self.trace.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
