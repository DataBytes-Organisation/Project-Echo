from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import detections as detections_service
from app.routers import detections as detections_router
from app.schemas import DetectionCreate


def valid_payload(sensor_id="bulk-test-1"):
    return {
        "timestamp": "2026-09-18T00:30:00Z",
        "sensorId": sensor_id,
        "species": "Uperoleia mimula",
        "microphoneLLA": [-38.8081, 143.5913, 10.0],
        "animalEstLLA": [-38.8082, 143.5929, 4.6],
        "animalTrueLLA": [-38.8082, 143.5929, 10.0],
        "animalLLAUncertainty": 2,
        "audioClip": "bulk-test-audio",
        "confidence": 98.5,
        "sampleRate": 32000,
    }


class FakeDetectionsCollection:
    def __init__(self):
        self.inserted_documents = []

    def insert_many(self, documents):
        self.inserted_documents = list(documents)
        return SimpleNamespace(
            inserted_ids=[
                f"fake-id-{index}"
                for index in range(len(documents))
            ]
        )


def test_bulk_service_inserts_multiple_detections(monkeypatch):
    fake_collection = FakeDetectionsCollection()

    monkeypatch.setattr(
        detections_service,
        "Detections",
        fake_collection,
    )

    detections = [
        DetectionCreate(**valid_payload("sensor-1")),
        DetectionCreate(**valid_payload("sensor-2")),
    ]

    ids = detections_service.create_detections_bulk(detections)

    assert len(ids) == 2
    assert len(fake_collection.inserted_documents) == 2
    assert fake_collection.inserted_documents[0]["sensorId"] == "sensor-1"
    assert fake_collection.inserted_documents[1]["sensorId"] == "sensor-2"


def test_bulk_endpoint_accepts_valid_batch(monkeypatch):
    monkeypatch.setattr(
        detections_router,
        "enforce_and_consume",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        detections_service,
        "create_detections_bulk",
        lambda items: ["id-1", "id-2"],
    )

    result = detections_router.create_detections_bulk_endpoint(
        [
            valid_payload("sensor-1"),
            valid_payload("sensor-2"),
        ]
    )

    assert result["received"] == 2
    assert result["inserted"] == 2
    assert result["rejected"] == 0
    assert result["inserted_ids"] == ["id-1", "id-2"]


def test_bulk_endpoint_rejects_invalid_item_but_inserts_valid(monkeypatch):
    monkeypatch.setattr(
        detections_router,
        "enforce_and_consume",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        detections_service,
        "create_detections_bulk",
        lambda items: ["id-1"],
    )

    invalid = valid_payload("bad-sensor")
    invalid["species"] = ""

    result = detections_router.create_detections_bulk_endpoint(
        [
            valid_payload("good-sensor"),
            invalid,
        ]
    )

    assert result["received"] == 2
    assert result["inserted"] == 1
    assert result["rejected"] == 1
    assert result["errors"][0]["index"] == 1


def test_bulk_endpoint_rejects_empty_batch():
    with pytest.raises(HTTPException) as exc:
        detections_router.create_detections_bulk_endpoint([])

    assert exc.value.status_code == 400


def test_bulk_endpoint_rejects_more_than_100():
    payload = [
        valid_payload(f"sensor-{index}")
        for index in range(101)
    ]

    with pytest.raises(HTTPException) as exc:
        detections_router.create_detections_bulk_endpoint(payload)

    assert exc.value.status_code == 413