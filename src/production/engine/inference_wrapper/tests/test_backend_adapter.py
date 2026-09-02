from ..inference_wrapper import InferenceWrapper
from ..backend_adapter import BackendAdapter


def test_success_response_converts_to_backend_payload():
    response = InferenceWrapper.build_success(
        timestamp="2026-08-30T10:00:00Z",
        species="Koala",
        confidence=96.42,
        sensorId="mic_01",
        microphoneLLA={
            "latitude": -38.143,
            "longitude": 144.361,
            "altitude": 15
        },
        animalEstLLA={
            "latitude": -38.142,
            "longitude": 144.360,
            "altitude": 15
        },
        animalTrueLLA=None,
        animalLLAUncertainty=8.5,
        audioClip="test-audio",
        sampleRate=48000
    )

    payload = BackendAdapter.to_backend_payload(response)

    assert payload["timestamp"] == "2026-08-30T10:00:00Z"
    assert payload["species"] == "Koala"
    assert payload["confidence"] == 96.42
    assert payload["sensorId"] == "mic_01"

    assert payload["microphoneLLA"] == [
        -38.143,
        144.361,
        15
    ]

    assert payload["animalEstLLA"] == [
        -38.142,
        144.360,
        15
    ]

    assert payload["animalTrueLLA"] is None

    assert payload["animalLLAUncertainty"] == 8.5
    assert payload["audioClip"] == "test-audio"
    assert payload["sampleRate"] == 48000


def test_failure_response_converts_to_backend_payload():
    response = InferenceWrapper.build_failure(
        timestamp="2026-08-30T10:00:00Z",
        sensorId="mic_01",
        sampleRate=48000,
        error_code="INVALID_AUDIO",
        error_message="Audio clip is empty."
    )

    payload = BackendAdapter.to_backend_payload(response)

    assert payload["timestamp"] == "2026-08-30T10:00:00Z"
    assert payload["species"] is None
    assert payload["confidence"] is None
    assert payload["sensorId"] == "mic_01"

    assert payload["microphoneLLA"] is None
    assert payload["animalEstLLA"] is None
    assert payload["animalTrueLLA"] is None

    assert payload["audioClip"] is None
    assert payload["sampleRate"] == 48000