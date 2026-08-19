from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_valid_post_request():
    # Valid JSON request
    response = client.post("/engine/event", json={
        "timestamp": "2026-08-01T00:00:00Z",
        "sensorId": "test",
        "species": "Bird",
        "microphoneLLA": [0.0, 0.0, 0.0],
        "animalEstLLA": [0.0, 0.0, 0.0],
        "animalTrueLLA": [0.0, 0.0, 0.0],
        "animalLLAUncertainty": 0,
        "audioClip": "base64",
        "confidence": 95,
        "sampleRate": 48000,
        "source_model": "unknown"
    })
    # Since the db might not be running or this payload might fail DB insert without it, 
    # we just want to ensure it isn't rejected with a 415 by the middleware.
    assert response.status_code != 415

def test_invalid_post_request_text_plain():
    # Invalid content-type (text/plain)
    response = client.post("/engine/event", data="This is some text", headers={"Content-Type": "text/plain"})
    assert response.status_code == 415
    assert "Unsupported media type" in response.text

def test_invalid_put_request_no_content_type():
    # Missing content type but has body
    response = client.put("/engine/event", data="body-content", headers={"Content-Length": "12"})
    assert response.status_code == 415

def test_get_request_ignores_middleware():
    # GET request with no body, should not trigger 415
    response = client.get("/health")
    assert response.status_code == 200
