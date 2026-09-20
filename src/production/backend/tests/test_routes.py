from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Verify that the root endpoint is accessible and returns a 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "echo api" in response.text.lower()

def test_openapi_spec_exists():
    """Verify that the OpenAPI export endpoint functions correctly."""
    response = client.get("/openapi-export")
    assert response.status_code == 200
    assert "paths" in response.json()

def test_engine_event_route_exists():
    """
    Verify that the ML Engine webhook route is mounted.
    A POST request without a body should result in 422 Unprocessable Entity,
    but it should definitely not be a 404 Not Found.
    """
    response = client.post("/engine/event")
    assert response.status_code != 404
    assert response.status_code == 422

def test_detections_route_exists():
    """Verify that the HMI detections retrieval route is mounted."""
    # It might return 500 if DB is not connected, but not 404
    response = client.get("/detections")
    assert response.status_code != 404


