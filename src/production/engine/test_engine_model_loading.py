"""
Automated tests for Echo Engine model loading.
"""

import unittest
from unittest.mock import MagicMock, patch

# Importing this first prepares the mocked heavy dependencies.
from test_iot_integration import EchoEngine  # noqa: F401

import echo_engine as engine_module


class TestEngineModelLoading(unittest.TestCase):

    def test_valid_model_is_returned(self):
        fake_model = MagicMock(name="fake_keras_model")

        with patch.object(
            engine_module,
            "load_model",
            return_value=fake_model
        ):
            result = engine_module.load_keras_model_file(
                "test_model.h5"
            )

        self.assertIs(result, fake_model)

    def test_missing_model_file_raises_clear_error(self):
        with patch.object(
            engine_module,
            "load_model",
            side_effect=OSError("File does not exist")
        ):
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Model file could not be loaded"
            ):
                engine_module.load_keras_model_file(
                    "missing_model.h5"
                )

    def test_corrupt_model_file_raises_clear_error(self):
        with patch.object(
            engine_module,
            "load_model",
            side_effect=ValueError("Invalid model format")
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Invalid or corrupt model file"
            ):
                engine_module.load_keras_model_file(
                    "corrupt_model.h5"
                )

    def test_empty_model_path_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Model path must be a non-empty string"
        ):
            engine_module.load_keras_model_file("")

    def test_configured_keras_model_path(self):
        engine_module.load_model.assert_any_call(
            "yamnet_dir/model_3_82_16000.h5"
        )

    def test_yamnet_assets_loaded_from_expected_paths(self):
        engine_module.yamnet.load_weights.assert_any_call(
            "yamnet_dir/yamnet.h5"
        )

        engine_module.tf.saved_model.load.assert_any_call(
            "yamnet_dir/model"
        )


if __name__ == "__main__":
    unittest.main()