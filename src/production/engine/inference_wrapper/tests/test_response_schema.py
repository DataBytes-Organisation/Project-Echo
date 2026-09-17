from ..response_schema import InferenceResponse


def test_success_response_schema_to_dict():
    response = InferenceResponse(
        status="success",
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
        sampleRate=48000,
        error=None
    )

    payload = response.to_dict()

    assert payload["status"] == "success"
    assert payload["timestamp"] == "2026-08-30T10:00:00Z"
    assert payload["species"] == "Koala"
    assert payload["confidence"] == 96.42
    assert payload["sensorId"] == "mic_01"

    assert payload["microphoneLLA"]["latitude"] == -38.143
    assert payload["microphoneLLA"]["longitude"] == 144.361
    assert payload["microphoneLLA"]["altitude"] == 15

    assert payload["animalEstLLA"]["latitude"] == -38.142
    assert payload["animalTrueLLA"] is None

    assert payload["animalLLAUncertainty"] == 8.5
    assert payload["audioClip"] == "test-audio"
    assert payload["sampleRate"] == 48000
    assert payload["error"] is None


def test_failure_response_schema_to_dict():
    response = InferenceResponse(
        status="failed",
        timestamp="2026-08-30T10:00:00Z",
        species=None,
        confidence=None,
        sensorId="mic_01",
        microphoneLLA=None,
        animalEstLLA=None,
        animalTrueLLA=None,
        animalLLAUncertainty=None,
        audioClip=None,
        sampleRate=48000,
        error={
            "code": "INVALID_AUDIO",
            "message": "Audio clip is empty."
        }
    )

    payload = response.to_dict()

    assert payload["status"] == "failed"
    assert payload["species"] is None
    assert payload["confidence"] is None
    assert payload["sensorId"] == "mic_01"

    assert payload["microphoneLLA"] is None
    assert payload["animalEstLLA"] is None
    assert payload["animalTrueLLA"] is None

    assert payload["animalLLAUncertainty"] is None
    assert payload["audioClip"] is None
    assert payload["sampleRate"] == 48000

    assert payload["error"]["code"] == "INVALID_AUDIO"
    assert payload["error"]["message"] == "Audio clip is empty."