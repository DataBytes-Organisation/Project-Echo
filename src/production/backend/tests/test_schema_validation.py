from fastapi.testclient import TestClient
from app.main import app
import datetime

client = TestClient(app)


def _valid_payload(**overrides):
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "sensorId": "S123",
        "species": "Koala",
        "sourceType": "real",
        "microphoneLLA": {"latitude": -33.0, "longitude": 150.0, "altitude": 20.0},
        "animalEstLLA": {"latitude": -33.01, "longitude": 150.01, "altitude": 20.0},
        "animalTrueLLA": {"latitude": -33.02, "longitude": 150.02, "altitude": 20.0},
        "animalLLAUncertainty": 5,
        "audioClip": "base64encodedaudio==",
        "confidence": 95.5,
        "sampleRate": 48000,
    }
    payload.update(overrides)
    return payload


def test_valid_payload_normalisation():
    """Verify that a valid engine payload is processed successfully."""
    response = client.post("/engine/event", json=_valid_payload())
    # Passed validation if not 422 (201 created, or another non-validation status).
    assert response.status_code != 422
    assert response.status_code == 201


def test_missing_required_field():
    """Verify that a payload missing a required field (species) is rejected."""
    payload = _valid_payload()
    del payload["species"]
    response = client.post("/engine/event", json=payload)
    assert response.status_code == 422


def test_out_of_bounds_confidence():
    """Verify that a payload with confidence > 100 is rejected."""
    response = client.post("/engine/event", json=_valid_payload(confidence=150.0))
    assert response.status_code == 422


def test_invalid_data_type():
    """Verify that a payload with an incorrect data type is rejected."""
    response = client.post(
        "/engine/event",
        json=_valid_payload(sampleRate="forty-eight thousand"),
    )
    assert response.status_code == 422
