"""
Automated tests for Echo Engine configuration loading.

Run from src/production/engine:
    python -m pytest test_engine_configuration.py -v
"""

import io
import json
import unittest
from unittest.mock import patch

from test_iot_integration import (
    EchoEngine,
    _ENGINE_CONFIG,
    _ENGINE_CREDS,
)

_REAL_OPEN = open


def controlled_open(config_value):
    """
    Return an open function that supplies controlled Engine
    configuration and credentials.
    """

    def open_file(path, *args, **kwargs):
        file_path = str(path)

        if file_path.endswith("echo_engine.json"):
            if isinstance(config_value, Exception):
                raise config_value

            return io.StringIO(config_value)

        if file_path.endswith("echo_credentials.json"):
            return io.StringIO(json.dumps(_ENGINE_CREDS))

        return _REAL_OPEN(path, *args, **kwargs)

    return open_file


class TestEngineConfigurationLoading(unittest.TestCase):

    def test_required_configuration_keys_exist(self):
        engine = EchoEngine()

        required_keys = {
            "AUDIO_CLIP_DURATION",
            "AUDIO_SAMPLE_RATE",
            "AUDIO_NFFT",
            "AUDIO_STRIDE",
            "AUDIO_MELS",
            "MODEL_INPUT_IMAGE_WIDTH",
            "MODEL_INPUT_IMAGE_HEIGHT",
            "MODEL_INPUT_IMAGE_CHANNELS",
            "MODEL_SERVER",
            "MQTT_CLIENT_URL",
            "MQTT_CLIENT_PORT",
            "IOT_MQTT_BROKER",
            "IOT_MQTT_PORT",
            "IOT_MQTT_TOPIC",
        }

        missing_keys = required_keys - set(engine.config)

        self.assertEqual(
            missing_keys,
            set(),
            f"Missing configuration keys: {missing_keys}"
        )

    def test_configuration_value_types(self):
        engine = EchoEngine()

        self.assertIsInstance(
            engine.config["AUDIO_SAMPLE_RATE"],
            int
        )
        self.assertIsInstance(
            engine.config["AUDIO_CLIP_DURATION"],
            int
        )
        self.assertIsInstance(
            engine.config["MODEL_SERVER"],
            str
        )
        self.assertIsInstance(
            engine.config["IOT_MQTT_PORT"],
            int
        )
        self.assertIsInstance(
            engine.config["IOT_MQTT_TOPIC"],
            str
        )

    def test_valid_custom_configuration_loaded(self):
        test_config = _ENGINE_CONFIG.copy()
        test_config["MODEL_SERVER"] = (
            "http://mock-model-server/test"
        )

        fake_open = controlled_open(
            json.dumps(test_config)
        )

        with patch("builtins.open", side_effect=fake_open):
            engine = EchoEngine()

        self.assertEqual(
            engine.config["MODEL_SERVER"],
            "http://mock-model-server/test"
        )
        self.assertEqual(
            engine.config["AUDIO_SAMPLE_RATE"],
            48000
        )

    def test_missing_configuration_handled(self):
        fake_open = controlled_open(
            FileNotFoundError("echo_engine.json not found")
        )

        with patch("builtins.open", side_effect=fake_open):
            with patch("builtins.print") as mock_print:
                engine = EchoEngine()

        self.assertFalse(hasattr(engine, "config"))

        printed_messages = " ".join(
            str(call.args[0])
            for call in mock_print.call_args_list
            if call.args
        )

        self.assertIn(
            "Could not engine config",
            printed_messages
        )

    def test_malformed_json_configuration_handled(self):
        fake_open = controlled_open(
            '{"AUDIO_SAMPLE_RATE": 48000, invalid}'
        )

        with patch("builtins.open", side_effect=fake_open):
            with patch("builtins.print") as mock_print:
                engine = EchoEngine()

        self.assertFalse(hasattr(engine, "config"))

        printed_messages = " ".join(
            str(call.args[0])
            for call in mock_print.call_args_list
            if call.args
        )

        self.assertIn(
            "Could not engine config",
            printed_messages
        )


if __name__ == "__main__":
    unittest.main()