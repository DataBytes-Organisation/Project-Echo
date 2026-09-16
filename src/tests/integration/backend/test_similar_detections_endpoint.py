"""Integration test for GET /detections/{id}/similar, using FastAPI's
TestClient - simulates a real HTTP request through the real app (routing,
response handling), with Mongo replaced by an in-memory fake so no live
database is needed. Follows the same FakeCollection pattern already used in
src/production/backend/tests/test_detection_retrieval.py, rather than the
DB-dependent pattern in test_public_routes.py, since this endpoint reads
detection documents directly.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[4] / "src" / "production" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from bson import ObjectId  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.routers import similar_detections  # noqa: E402


class FakeDetectionsCollection:
    def __init__(self, documents):
        self.documents = {doc["_id"]: doc for doc in documents}

    def find_one(self, query):
        return self.documents.get(query.get("_id"))

    def find(self, query, projection=None):
        results = []
        for doc in self.documents.values():
            if "embedding" in query and doc.get("embedding") is None:
                continue
            results.append(doc)
        return results


client = TestClient(app)

SPECIES_A_ID = ObjectId()
SPECIES_A_NEIGHBOUR_ID = ObjectId()
SPECIES_B_ID = ObjectId()
NO_EMBEDDING_ID = ObjectId()


def _make_documents():
    return [
        {"_id": SPECIES_A_ID, "species": "SpeciesA", "embedding": [1.0, 0.0]},
        {"_id": SPECIES_A_NEIGHBOUR_ID, "species": "SpeciesA", "embedding": [0.99, 0.01]},
        {"_id": SPECIES_B_ID, "species": "SpeciesB", "embedding": [0.0, 1.0]},
        {"_id": NO_EMBEDDING_ID, "species": "SpeciesC", "embedding": None},
    ]


class TestSimilarDetectionsEndpoint:
    def test_returns_closest_matches_ranked_by_similarity(self, monkeypatch):
        monkeypatch.setattr(similar_detections, "Detections", FakeDetectionsCollection(_make_documents()))

        response = client.get(f"/detections/{SPECIES_A_ID}/similar")

        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["detection_id"] == str(SPECIES_A_NEIGHBOUR_ID)
        assert body["ambiguous"] is True  # top-5 includes both SpeciesA and SpeciesB here

    def test_unknown_detection_id_returns_404(self, monkeypatch):
        monkeypatch.setattr(similar_detections, "Detections", FakeDetectionsCollection(_make_documents()))

        response = client.get(f"/detections/{ObjectId()}/similar")

        assert response.status_code == 404

    def test_malformed_detection_id_returns_400(self, monkeypatch):
        monkeypatch.setattr(similar_detections, "Detections", FakeDetectionsCollection(_make_documents()))

        response = client.get("/detections/not-a-real-object-id/similar")

        assert response.status_code == 400

    def test_detection_with_no_embedding_returns_422(self, monkeypatch):
        monkeypatch.setattr(similar_detections, "Detections", FakeDetectionsCollection(_make_documents()))

        response = client.get(f"/detections/{NO_EMBEDDING_ID}/similar")

        assert response.status_code == 422
