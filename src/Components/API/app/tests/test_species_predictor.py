import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from app.main import app
from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_predict_species_success():
    with patch('app.routers.species_predictor.predict_species') as mock_predict, \
         patch('app.routers.species_predictor.Predictions.insert_one') as mock_insert:
        mock_predict.return_value = {"species": "Test Bird", "confidence": 0.99}
        mock_insert.return_value = MagicMock()
        with open("test_audio.wav", "wb") as f:
            f.write(b"fake audio data")
        with open("test_audio.wav", "rb") as audio_file:
            response = client.post("/predict", files={"audio": ("test_audio.wav", audio_file, "audio/wav")})
        assert response.status_code == 200
        assert response.json()["predicted_species"] == "Test Bird"
        assert "confidence" in response.json()


def test_predict_species_no_audio():
    response = client.post("/predict", files={})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_recent_predictions():
    with patch('app.routers.species_predictor.Predictions.find') as mock_find:
        mock_find.return_value.sort.return_value.limit.return_value = [
            {"_id": "1", "filename": "a.wav", "predicted_species": "Test Bird", "confidence": 0.99, "timestamp": "2024-01-01T00:00:00", "user_id": "user1"}
        ]
        response = client.get("/predictions/recent?limit=1")
        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["items"][0]["predicted_species"] == "Test Bird"
