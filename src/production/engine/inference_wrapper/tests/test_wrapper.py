from ..inference_wrapper import InferenceWrapper


def test_build_success_response():
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

    assert response.status == "success"
    assert response.species == "Koala"
    assert response.confidence == 96.42
    assert response.sensorId == "mic_01"
    assert response.error is None


def test_build_failure_response():
    response = InferenceWrapper.build_failure(
        timestamp="2026-08-30T10:00:00Z",
        sensorId="mic_01",
        sampleRate=48000,
        error_code="INVALID_AUDIO",
        error_message="Audio clip is empty."
    )

    assert response.status == "failed"
    assert response.species is None
    assert response.confidence is None
    assert response.sensorId == "mic_01"
    assert response.error["code"] == "INVALID_AUDIO"
    assert response.error["message"] == "Audio clip is empty."