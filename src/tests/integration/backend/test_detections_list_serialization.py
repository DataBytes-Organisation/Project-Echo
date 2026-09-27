"""Regression test for a real bug found while building the similar-detection
retrieval feature: GET /detections (the list endpoint) crashed with a 500
whenever it serialized real Mongo-backed documents, because
DetectionListResponses had no Config.json_encoders for ObjectId, unlike
Detection/DetectionCreate which each declare their own. This went unnoticed
because src/production/backend/tests/test_detection_retrieval.py calls
detections_service.list_detections() directly as a plain function, which
never exercises the actual FastAPI response_model JSON-encoding step this
test goes through via TestClient.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[4] / "src" / "production" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from bson import ObjectId  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from app.main import app  # noqa: E402
from app import detections as detections_service  # noqa: E402


class FakeCursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, field, direction):
        return self

    def skip(self, count):
        self.documents = self.documents[count:]
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(self.documents)


class FakeDetectionsCollection:
    def __init__(self, documents):
        self.documents = documents

    def count_documents(self, query):
        return len(self.documents)

    def find(self, query):
        return FakeCursor(self.documents)


def _make_document(species):
    return {
        "_id": ObjectId(),
        "timestamp": datetime.now(timezone.utc),
        "sensorId": "test-sensor",
        "species": species,
        "microphoneLLA": [-33.1, 150.0, 10],
        "animalEstLLA": [-33.1, 150.0, 10],
        "animalTrueLLA": [-33.1, 150.0, 10],
        "animalLLAUncertainty": 5,
        "audioClip": "dGVzdA==",
        "confidence": 90.0,
        "sampleRate": 48000,
    }


client = TestClient(app)


class TestDetectionsListSerialization:
    def test_list_endpoint_serialises_real_object_ids_without_500(self, monkeypatch):
        documents = [_make_document("SpeciesA"), _make_document("SpeciesB")]
        monkeypatch.setattr(detections_service, "Detections", FakeDetectionsCollection(documents))

        response = client.get("/detections?page=1&page_size=10")

        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 2
        assert {item["species"] for item in body["items"]} == {"SpeciesA", "SpeciesB"}

    def test_list_endpoint_with_a_single_item_also_works(self, monkeypatch):
        # The bug only reproduced with the real response_model encoding path,
        # not with a single hand-encoded Detection, so cover both counts.
        monkeypatch.setattr(detections_service, "Detections", FakeDetectionsCollection([_make_document("SpeciesA")]))

        response = client.get("/detections?page=1&page_size=10")

        assert response.status_code == 200
        assert len(response.json()["items"]) == 1
