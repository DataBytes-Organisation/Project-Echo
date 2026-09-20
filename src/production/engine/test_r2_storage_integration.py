"""Actual engine/simulator R2 storage paths with cloud and MQTT mocked."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from test_engine_end_to_end import load_isolated_engine


SRC = Path(__file__).resolve().parents[2]
STORE = SRC / "production" / "infrastructure" / "store"
STORAGE_TESTS = SRC / "tests" / "unit" / "storage"
with patch.object(sys, "path", [str(STORE), str(STORAGE_TESTS), *sys.path]):
    from cloudflare_r2 import R2Storage
    from test_simulator_r2 import comms


class EngineSimulatorR2Tests(unittest.TestCase):
    def setUp(self):
        self.module = load_isolated_engine()
        # Model loading and database setup are not storage operations.
        self.engine = self.module.EchoEngine.__new__(self.module.EchoEngine)
        self.engine.config = {
            "MQTT_CLIENT_URL": "unused.invalid", "MQTT_CLIENT_PORT": 1883,
            "MQTT_PUBLISH_URL": "projectecho/engine/2",
        }
        self.simulator = comms.CommsManager()
        settings = {
            "R2_ACCOUNT_ID": "test-account", "R2_BUCKET_NAME": "test-bucket",
            "R2_ACCESS_KEY_ID": "test-key", "R2_SECRET_ACCESS_KEY": "test-secret",
            "R2_DATASET_PREFIX": "",
        }
        environment = patch.dict(os.environ, settings, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.client = MagicMock()
        self.client.list_objects_v2.return_value = {"Contents": [
            {"Key": "Species B/two.wav", "Size": 100},
            {"Key": "Species A/one.wav", "Size": 100},
            {"Key": "Empty/", "Size": 0},
        ]}
        client_factory = patch.object(R2Storage, "_create_client", return_value=self.client)
        client_factory.start()
        self.addCleanup(client_factory.stop)

    def test_engine_and_simulator_use_same_bucket_and_root_species(self):
        engine_species = self.engine.r2_load_species_list()
        simulator_species = [s.getName() for s in self.simulator.r2_load_species_list()]
        self.assertEqual(engine_species, ["Species A", "Species B"])
        self.assertEqual(engine_species, simulator_species)
        for request in self.client.list_objects_v2.call_args_list:
            self.assertEqual(request.kwargs, {"Bucket": "test-bucket", "Prefix": ""})
        self.client.get_object.assert_not_called()

    def test_both_components_respect_the_same_optional_prefix(self):
        os.environ["R2_DATASET_PREFIX"] = "datasets/simulator"
        self.client.list_objects_v2.return_value = {"Contents": [
            {"Key": "datasets/simulator/Species A/one.wav", "Size": 100},
        ]}
        self.assertEqual(self.engine.r2_load_species_list(), ["Species A"])
        self.assertEqual([s.getName() for s in self.simulator.r2_load_species_list()],
                         ["Species A"])
        for request in self.client.list_objects_v2.call_args_list:
            self.assertEqual(request.kwargs,
                             {"Bucket": "test-bucket", "Prefix": "datasets/simulator/"})

    def test_engine_reads_every_page_in_sorted_order(self):
        self.client.list_objects_v2.side_effect = [
            {"Contents": [{"Key": "Species B/two.wav", "Size": 100}],
             "IsTruncated": True, "NextContinuationToken": "page-2"},
            {"Contents": [{"Key": "Species A/one.wav", "Size": 100}]},
        ]
        self.assertEqual(self.engine.r2_load_species_list(), ["Species A", "Species B"])
        self.assertEqual(self.client.list_objects_v2.call_args.kwargs["ContinuationToken"],
                         "page-2")

    def test_empty_dataset_fails_in_both_components(self):
        self.client.list_objects_v2.return_value = {}
        for component in (self.engine, self.simulator):
            with self.subTest(component=type(component).__name__):
                with self.assertRaisesRegex(RuntimeError, "No species audio"):
                    component.r2_load_species_list()

    def test_missing_credentials_fail_without_cloud_access(self):
        del os.environ["R2_SECRET_ACCESS_KEY"]
        with self.assertRaisesRegex(ValueError, "R2_SECRET_ACCESS_KEY"):
            self.engine.r2_load_species_list()
        self.assertEqual(self.client.mock_calls, [])

    def test_permission_failure_is_not_silently_accepted(self):
        self.client.list_objects_v2.side_effect = PermissionError("Access denied")
        with self.assertRaisesRegex(PermissionError, "Access denied"):
            self.engine.r2_load_species_list()

    def test_engine_loads_r2_species_before_starting_iot_listener(self):
        calls = []
        load_species = self.engine.r2_load_species_list

        def checked_load():
            calls.append("r2_species")
            return load_species()

        with patch.object(self.engine, "r2_load_species_list", side_effect=checked_load):
            with patch.object(self.engine, "start_iot_mqtt_listener",
                              side_effect=lambda: calls.append("iot_listener")):
                with redirect_stdout(io.StringIO()):
                    self.engine.execute()
        self.assertEqual(calls, ["r2_species", "iot_listener"])
        self.assertEqual(self.engine.class_names, ["Species A", "Species B"])
        self.module.paho.Client.return_value.loop_forever.assert_called_once()

    def test_failed_storage_does_not_start_iot_listener(self):
        self.client.list_objects_v2.side_effect = PermissionError("Access denied")
        with patch.object(self.engine, "start_iot_mqtt_listener") as listener:
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(PermissionError):
                    self.engine.execute()
        listener.assert_not_called()
        self.module.paho.Client.return_value.loop_forever.assert_not_called()


if __name__ == "__main__":
    unittest.main()
