"""
Automated tests for Echo Engine audio preprocessing.

These tests use a small generated WAV fixture and do not
require the complete audio dataset.
"""

import io
import unittest
import wave
from unittest.mock import MagicMock

import numpy as np

# Prepares mocked heavy dependencies before importing the Engine.
from test_iot_integration import EchoEngine  # noqa: F401

import echo_engine_iot as engine_module


def generate_small_wav(
    duration_seconds=0.10,
    sample_rate=8000,
    frequency=440
):
    """Generate a small mono WAV fixture in memory."""

    sample_count = int(duration_seconds * sample_rate)

    time_values = np.arange(sample_count) / sample_rate

    samples = (
        0.25
        * np.sin(2 * np.pi * frequency * time_values)
        * 32767
    ).astype(np.int16)

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())

    return wav_buffer.getvalue()


class NumpyTensorResult:
    """Small TensorFlow-like wrapper providing numpy()."""

    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class TestEnginePreprocessing(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()

        self.processed_audio = np.zeros(
            48000 * 5,
            dtype=np.float32
        )

        self.engine.load_random_subsection = MagicMock(
            return_value=self.processed_audio
        )

        # Simulate librosa loading valid audio.
        engine_module.librosa.load.reset_mock()
        engine_module.librosa.load.side_effect = None
        engine_module.librosa.load.return_value = (
            np.zeros(800, dtype=np.float32),
            48000,
        )

        # Simulate a mel-spectrogram with sufficient time frames.
        mel_spectrogram = np.linspace(
            0.0,
            1.0,
            260 * 1201,
            dtype=np.float32
        ).reshape(260, 1201)

        (
            engine_module.librosa.feature
            .melspectrogram
            .reset_mock()
        )

        (
            engine_module.librosa.feature
            .melspectrogram
            .return_value
        ) = mel_spectrogram

        engine_module.librosa.power_to_db.reset_mock()
        engine_module.librosa.power_to_db.return_value = (
            mel_spectrogram
        )

        # Configure mocked TensorFlow operations to use NumPy.
        engine_module.tf.expand_dims.side_effect = (
            lambda value, axis: np.expand_dims(value, axis)
        )

        engine_module.tf.repeat.side_effect = (
            lambda value, repeats, axis:
            np.repeat(value, repeats, axis=axis)
        )

        def ensure_shape(value, expected_shape):
            if list(value.shape) != list(expected_shape):
                raise ValueError(
                    f"Unexpected shape: {value.shape}"
                )
            return value

        engine_module.tf.ensure_shape.side_effect = ensure_shape

        resized_image = np.linspace(
            0.0,
            1.0,
            260 * 260 * 3,
            dtype=np.float32
        ).reshape(260, 260, 3)

        engine_module.tf.image.resize.side_effect = (
            lambda value, size, method: resized_image.copy()
        )

        engine_module.tf.reduce_min.side_effect = np.min
        engine_module.tf.reduce_max.side_effect = np.max

    def test_valid_wav_preprocessing_output(self):
        wav_fixture = generate_small_wav()

        image, audio, sample_rate = (
            self.engine.combined_pipeline(
                wav_fixture,
                "Recording_Mode"
            )
        )

        self.assertEqual(image.shape, (260, 260, 3))
        self.assertEqual(audio.shape, (48000 * 5,))
        self.assertEqual(sample_rate, 48000)
        self.assertTrue(np.isfinite(image).all())
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_preprocessing_uses_configured_sample_rate(self):
        wav_fixture = generate_small_wav()

        self.engine.combined_pipeline(
            wav_fixture,
            "Recording_Mode"
        )

        load_arguments = (
            engine_module.librosa.load.call_args
        )

        loaded_file = load_arguments.args[0]
        configured_rate = load_arguments.kwargs["sr"]

        self.assertIsInstance(loaded_file, io.BytesIO)
        self.assertEqual(
            configured_rate,
            self.engine.config["AUDIO_SAMPLE_RATE"]
        )

    def test_corrupt_audio_rejected(self):
        engine_module.librosa.load.side_effect = ValueError(
            "Invalid WAV file"
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid WAV file"
        ):
            self.engine.combined_pipeline(
                b"This is not WAV audio",
                "Recording_Mode"
            )

    def test_empty_audio_rejected(self):
        engine_module.librosa.load.side_effect = ValueError(
            "Audio file is empty"
        )

        with self.assertRaisesRegex(
            ValueError,
            "Audio file is empty"
        ):
            self.engine.combined_pipeline(
                b"",
                "Recording_Mode"
            )


class TestAudioSubsection(unittest.TestCase):

    def setUp(self):
        self.engine = EchoEngine()
        self.engine.config["AUDIO_SAMPLE_RATE"] = 4

    def test_short_audio_is_padded(self):
        short_audio = np.array(
            [0.5, -0.5],
            dtype=np.float32
        )

        engine_module.tf.shape.side_effect = (
            lambda value: np.array(value.shape)
        )

        engine_module.tf.zeros.side_effect = (
            lambda shape, dtype:
            np.zeros(shape, dtype=dtype)
        )

        engine_module.tf.concat.side_effect = (
            lambda values, axis:
            NumpyTensorResult(
                np.concatenate(values, axis=axis)
            )
        )

        result = self.engine.load_random_subsection(
            short_audio,
            duration_secs=1
        )

        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(
            result,
            np.array(
                [0.5, -0.5, 0.0, 0.0],
                dtype=np.float32
            )
        )


if __name__ == "__main__":
    unittest.main()