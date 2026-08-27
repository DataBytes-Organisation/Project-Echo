"""
pytest tests for the Sprint 2 Engine Inference Wrapper integration.
"""

import pytest

from src.prototypes.engine.Inference_wrapper.prototype_engine import PrototypeEngine


@pytest.fixture
def valid_event():
    return {
        "timestamp": "2026-08-08T10:00:00Z",
        "sensorId": "mic_01",
        "microphoneLLA": [
            -38.143,
            144.361,
            15
        ],
        "animalEstLLA": [
            -38.142,
            144.360,
            15
        ],
        "animalTrueLLA": [
            -38.142,
            144.360,
            15
        ],
        "animalLLAUncertainty": 8.5,
        "audioClip": "VGhpc3Byb3BseWRhdGE="
    }


def test_valid_prediction(valid_event):
    response = PrototypeEngine.process_prediction(
        valid_event,
        "Koala",
        96.42,
        48000
    )

    assert response is not None
    assert response.to_dict() is not None


def test_missing_sensor_id(valid_event):
    missing_sensor = valid_event.copy()
    del missing_sensor["sensorId"]

    response = PrototypeEngine.process_prediction(
        missing_sensor,
        "Koala",
        96.42,
        48000
    )

    assert response is not None


def test_invalid_audio(valid_event):
    invalid_audio = valid_event.copy()
    invalid_audio["audioClip"] = ""

    response = PrototypeEngine.process_prediction(
        invalid_audio,
        "Koala",
        96.42,
        48000
    )

    assert response is not None


def test_inference_failure(valid_event):
    response = PrototypeEngine.process_prediction(
        valid_event,
        None,
        None,
        48000
    )

    assert response is not None