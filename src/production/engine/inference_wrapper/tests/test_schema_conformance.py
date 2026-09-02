from ..inference_wrapper import InferenceWrapper
from ..backend_adapter import BackendAdapter


EXPECTED_BACKEND_FIELDS = {
    "timestamp",
    "species",
    "confidence",
    "sensorId",
    "microphoneLLA",
    "animalEstLLA",
    "animalTrueLLA",
    "animalLLAUncertainty",
    "audioClip",
    "sampleRate",
}


def test_backend_payload_field_conformance():
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
        animalTrueLLA={
            "latitude": -38.142,
            "longitude": 144.360,
            "altitude": 15
        },
        animalLLAUncertainty=8.5,
        audioClip="test-audio",
        sampleRate=48000
    )

    backend_payload = BackendAdapter.to_backend_payload(response)

    assert set(backend_payload.keys()) == EXPECTED_BACKEND_FIELDS


def test_backend_payload_gps_format():
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
        animalTrueLLA={
            "latitude": -38.142,
            "longitude": 144.360,
            "altitude": 15
        },
        animalLLAUncertainty=8.5,
        audioClip="test-audio",
        sampleRate=48000
    )

    backend_payload = BackendAdapter.to_backend_payload(response)

    assert isinstance(backend_payload["microphoneLLA"], list)
    assert len(backend_payload["microphoneLLA"]) == 3

    assert isinstance(backend_payload["animalEstLLA"], list)
    assert len(backend_payload["animalEstLLA"]) == 3

    assert isinstance(backend_payload["animalTrueLLA"], list)
    assert len(backend_payload["animalTrueLLA"]) == 3