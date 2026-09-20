"""Simulator-only R2 tests; no credentials, engine or cloud required.

Run: python -m unittest discover -s src/tests/unit/storage -p test_simulator_r2.py -v
"""

import base64
import datetime
import importlib
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


PRODUCTION = Path(__file__).resolve().parents[3] / "production"
SIMULATOR_SRC = PRODUCTION / "simulator" / "src"
STORE_DIR = PRODUCTION / "infrastructure" / "store"


def load_comms(file_path):
    """Load real simulator code, isolating unrelated network dependencies."""
    source = (SIMULATOR_SRC / "comms_manager.py").read_text(encoding="utf-8")
    module = types.ModuleType("simulator_r2_under_test")
    module.__file__ = str(file_path)
    dependencies = {
        name: MagicMock()
        for name in ("paho", "paho.mqtt", "paho.mqtt.client", "pymongo", "requests")
    }
    with patch.object(sys, "path", [str(SIMULATOR_SRC), str(STORE_DIR), *sys.path]):
        # Keep the real storage module cached when dependency mocks are removed.
        importlib.import_module("cloudflare_r2")
        with patch.dict(sys.modules, dependencies):
            exec(compile(source, str(file_path), "exec"), module.__dict__)
    return module


comms = load_comms(SIMULATOR_SRC / "comms_manager.py")


class SimulatorR2Tests(unittest.TestCase):
    def setUp(self):
        settings = {
            "R2_ACCOUNT_ID": "test-account",
            "R2_BUCKET_NAME": "project-echo-simulator-prod",
            "R2_ACCESS_KEY_ID": "test-key",
            "R2_SECRET_ACCESS_KEY": "test-secret",
            "R2_DATASET_PREFIX": "",
            "MQTT_PUBLISH_URL": "projectecho/engine/2",
        }
        environment = patch.dict(os.environ, settings, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

        self.key = "Gymnorhina tibicen/recording-001.wav"
        self.audio = b"RIFF-original-audio-bytes"
        self.client = MagicMock()
        self.client.list_objects_v2.return_value = {
            "Contents": [{"Key": self.key, "Size": len(self.audio)}]
        }
        self.client.get_object.side_effect = lambda **kwargs: {
            "Body": io.BytesIO(self.audio)
        }
        client_factory = patch.object(comms.R2Storage, "_create_client", return_value=self.client)
        client_factory.start()
        self.addCleanup(client_factory.stop)

        self.manager = comms.CommsManager()
        self.manager.clock = MagicMock()
        self.manager.clock.get_time.return_value = datetime.datetime(2026, 9, 19, 12)
        self.manager.mqtt_client = MagicMock()
        self.manager.mqtt_client.publish.return_value = (0, 1)
        self.animal = MagicMock()
        self.animal.getSpecies.return_value.getName.return_value = "Gymnorhina tibicen"
        self.animal.getLLA.return_value = (-37.0, 145.0, 10.0)
        self.mic = MagicMock()
        self.mic.getLLA.return_value = (-37.1, 145.1, 11.0)
        self.mic.getID.return_value = "mic-1"

    def send_audio(self):
        with patch("builtins.print"):
            self.manager.mqtt_send_random_audio_msg(
                self.animal, (-37.2, 145.2, 12.0), self.mic, 5.0
            )

    def test_root_species_and_original_key_are_preserved(self):
        species = self.manager.r2_load_species_list()
        self.assertEqual([item.getName() for item in species], ["Gymnorhina tibicen"])
        self.assertEqual(self.manager.audio_keys, {"Gymnorhina tibicen": [self.key]})
        self.client.list_objects_v2.assert_called_once_with(
            Bucket="project-echo-simulator-prod", Prefix=""
        )

    def test_optional_prefix_does_not_become_a_species_name(self):
        os.environ["R2_DATASET_PREFIX"] = "datasets/simulator"
        prefixed_key = "datasets/simulator/" + self.key
        self.client.list_objects_v2.return_value = {
            "Contents": [{"Key": prefixed_key, "Size": len(self.audio)}]
        }
        self.manager.r2_load_species_list()
        self.send_audio()
        self.client.list_objects_v2.assert_called_once_with(
            Bucket="project-echo-simulator-prod", Prefix="datasets/simulator/"
        )
        self.client.get_object.assert_called_once_with(
            Bucket="project-echo-simulator-prod", Key=prefixed_key
        )

    def test_all_listing_pages_are_loaded(self):
        self.client.list_objects_v2.side_effect = [
            {"Contents": [{"Key": self.key, "Size": 50}],
             "IsTruncated": True, "NextContinuationToken": "page-2"},
            {"Contents": [{"Key": "Species B/two.wav", "Size": 50}]},
        ]
        species = self.manager.r2_load_species_list()
        self.assertEqual([item.getName() for item in species],
                         ["Gymnorhina tibicen", "Species B"])
        self.assertEqual(self.client.list_objects_v2.call_args.kwargs,
                         {"Bucket": "project-echo-simulator-prod", "Prefix": "",
                          "ContinuationToken": "page-2"})

    def test_download_preserves_audio_and_existing_mqtt_contract(self):
        self.manager.r2_load_species_list()
        self.send_audio()
        topic, payload = self.manager.mqtt_client.publish.call_args.args
        self.assertEqual(topic, "projectecho/engine/2")
        self.assertEqual(json.loads(payload), {
            "timestamp": "2026-09-19T12:00:00",
            "sensorId": "mic-1",
            "microphoneLLA": [-37.1, 145.1, 11.0],
            "animalEstLLA": [-37.2, 145.2, 12.0],
            "animalTrueLLA": [-37.0, 145.0, 10.0],
            "animalLLAUncertainty": 5.0,
            "audioClip": base64.b64encode(self.audio).decode("utf-8"),
            "mode": "Animal_Mode",
            "audioFile": "recording-001.wav",
        })
        self.client.get_object.assert_called_once_with(
            Bucket="project-echo-simulator-prod", Key=self.key
        )
        self.assertEqual([call[0] for call in self.client.mock_calls],
                         ["list_objects_v2", "get_object"])

    def test_missing_credentials_fail_before_accessing_storage(self):
        del os.environ["R2_SECRET_ACCESS_KEY"]
        with self.assertRaisesRegex(ValueError, "R2_SECRET_ACCESS_KEY"):
            self.manager.r2_load_species_list()
        self.assertEqual(self.client.mock_calls, [])

    def test_empty_dataset_or_folder_markers_fail_clearly(self):
        for objects in ([], [{"Key": "Species/", "Size": 0},
                             {"Key": "Species/empty.wav", "Size": 0}]):
            with self.subTest(objects=objects):
                self.client.list_objects_v2.return_value = {"Contents": objects}
                with self.assertRaisesRegex(RuntimeError, "No species audio"):
                    self.manager.r2_load_species_list()
        self.manager.mqtt_client.publish.assert_not_called()

    def test_uninitialised_or_unknown_species_do_not_publish(self):
        with self.assertRaisesRegex(RuntimeError, "Load the R2 species"):
            self.send_audio()
        self.manager.r2_load_species_list()
        self.animal.getSpecies.return_value.getName.return_value = "Unknown"
        with self.assertRaisesRegex(KeyError, "Unknown"):
            self.send_audio()
        self.client.get_object.assert_not_called()
        self.manager.mqtt_client.publish.assert_not_called()

    def test_listing_failure_is_not_reported_as_success(self):
        self.client.list_objects_v2.side_effect = PermissionError("Access denied")
        with self.assertRaisesRegex(PermissionError, "Access denied"):
            self.manager.r2_load_species_list()
        self.assertIsNone(self.manager.r2_storage)

    def test_download_failure_does_not_publish(self):
        self.manager.r2_load_species_list()
        self.client.get_object.side_effect = PermissionError("Access denied")
        with self.assertRaisesRegex(PermissionError, "Access denied"):
            self.send_audio()
        self.manager.mqtt_client.publish.assert_not_called()

    def test_recording_messages_are_forwarded_without_storage_access(self):
        message = MagicMock(payload=b"unchanged-recording-message")
        with patch("builtins.print"):
            self.manager.mqtt_send_recording_msg(message, "Recording_Mode")
        self.manager.mqtt_client.publish.assert_called_once_with(
            "projectecho/engine/2", message.payload
        )
        self.assertEqual(self.client.mock_calls, [])

    def test_docker_layout_import_does_not_require_repository_parents(self):
        module = load_comms(Path(Path.cwd().anchor) / "app" / "comms_manager.py")
        self.assertIs(module.R2Storage, comms.R2Storage)

    def test_local_import_resolves_shared_storage_directory(self):
        self.assertEqual(comms._store_dir, STORE_DIR)


if __name__ == "__main__":
    unittest.main()
