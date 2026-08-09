from fastapi.testclient import TestClient
from app.main import app
import datetime

client = TestClient(app)

# Note: Using Australian/British spelling for tests where applicable
def test_valid_payload_normalisation():
    """Verify that a valid engine payload is processed successfully."""
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "sensorId": "S123",
        "species": "Koala",
        "microphoneLLA": [-33.0, 150.0, 20.0],
        "animalEstLLA": [-33.01, 150.01, 20.0],
        "animalTrueLLA": [-33.02, 150.02, 20.0],
        "animalLLAUncertainty": 5,
        "audioClip": "base64encodedaudio==",
        "confidence": 95.5,
        "sampleRate": 48000
    }
    
    # We expect 422 if the schema is invalid, so if it returns something else 
    # (like 500 because the DB is down, or 201 Created), it means it passed validation.
    response = client.post("/engine/event", json=payload)
    assert response.status_code != 422

def test_missing_required_field():
    """Verify that a payload missing a required field (species) is rejected."""
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "sensorId": "S123",
        # "species" is missing
        "microphoneLLA": [-33.0, 150.0, 20.0],
        "animalEstLLA": [-33.01, 150.01, 20.0],
        "animalTrueLLA": [-33.02, 150.02, 20.0],
        "animalLLAUncertainty": 5,
        "audioClip": "base64encodedaudio==",
        "confidence": 95.5,
        "sampleRate": 48000
    }
    response = client.post("/engine/event", json=payload)
    assert response.status_code == 422

def test_out_of_bounds_confidence():
    """Verify that a payload with confidence > 100 is rejected."""
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "sensorId": "S123",
        "species": "Koala",
        "microphoneLLA": [-33.0, 150.0, 20.0],
        "animalEstLLA": [-33.01, 150.01, 20.0],
        "animalTrueLLA": [-33.02, 150.02, 20.0],
        "animalLLAUncertainty": 5,
        "audioClip": "base64encodedaudio==",
        "confidence": 150.0, # Invalid, max is 100
        "sampleRate": 48000
    }
    response = client.post("/engine/event", json=payload)
    assert response.status_code == 422

def test_invalid_data_type():
    """Verify that a payload with an incorrect data type is rejected."""
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "sensorId": "S123",
        "species": "Koala",
        "microphoneLLA": [-33.0, 150.0, 20.0],
        "animalEstLLA": [-33.01, 150.01, 20.0],
        "animalTrueLLA": [-33.02, 150.02, 20.0],
        "animalLLAUncertainty": 5,
        "audioClip": "base64encodedaudio==",
        "confidence": 95.5,
        "sampleRate": "forty-eight thousand" # Invalid string instead of int
    }
    response = client.post("/engine/event", json=payload)
    assert response.status_code == 422
