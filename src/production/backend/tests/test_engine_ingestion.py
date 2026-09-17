from types import SimpleNamespace
import asyncio

import pytest
from pydantic import ValidationError

from app.schemas import EventSchema
from app.routers import engine


def valid_payload():
    return {
        "timestamp": "2026-08-31T01:00:00Z",
        "sensorId": "unit-test-sensor",
        "species": "Uperoleia mimula",
        "sourceType": "simulator",
        "microphoneLLA": {"latitude": -38.8081, "longitude": 143.5913, "altitude": 10.0},
        "animalEstLLA": {"latitude": -38.8082, "longitude": 143.5929, "altitude": 4.6},
        "animalTrueLLA": {"latitude": -38.8082, "longitude": 143.5929, "altitude": 10.0},
        "animalLLAUncertainty": 0.0,
        "audioClip": "unit-test-audio",
        "confidence": 99.36,
        "sampleRate": 32000,
    }


class FakeEventsCollection:
    def __init__(self):
        self.inserted_document = None
        self.inserted_id = "fake-event-id"

    def insert_one(self, document):
        self.inserted_document = document.copy()
        self.inserted_document["_id"] = self.inserted_id
        return SimpleNamespace(inserted_id=self.inserted_id)

    def aggregate(self, pipeline):
        return [self.inserted_document]


def test_valid_engine_event_schema():
    event = EventSchema(**valid_payload())

    assert event.sensorId == "unit-test-sensor"
    assert event.species == "Uperoleia mimula"
    assert event.confidence == 99.36
    assert event.sampleRate == 32000


def test_create_event_uses_shared_persistence(monkeypatch):
    captured = {}

    def fake_persist(event):
        captured["event"] = event
        return "fake-event-id"

    def fake_build_stream_payload(inserted_id):
        return {
            "_id": inserted_id,
            "sensorId": "unit-test-sensor",
        }

    async def fake_broadcast(payload):
        captured["broadcast"] = payload

    monkeypatch.setattr(
        engine,
        "persist_event",
        fake_persist,
    )

    monkeypatch.setattr(
        engine,
        "_build_stream_payload",
        fake_build_stream_payload,
    )

    monkeypatch.setattr(
        engine.detection_stream_manager,
        "broadcast",
        fake_broadcast,
    )

    event = EventSchema(**valid_payload())
    response = asyncio.run(engine.create_event(event))

    assert captured["event"].sensorId == "unit-test-sensor"
    assert captured["event"].species == "Uperoleia mimula"
    assert captured["event"].confidence == 99.36

    assert captured["broadcast"]["_id"] == "fake-event-id"

    assert response == {
        "status": "success",
        "eventId": "fake-event-id",
    }


def test_empty_species_is_rejected():
    payload = valid_payload()
    payload["species"] = ""

    with pytest.raises(ValidationError):
        EventSchema(**payload)


def test_confidence_100_is_accepted():
    payload = valid_payload()
    payload["confidence"] = 100

    assert EventSchema(**payload).confidence == 100


def test_invalid_microphone_location_length_is_rejected():
    payload = valid_payload()
    payload["microphoneLLA"] = {"latitude": -38.8081, "longitude": 143.5913}

    with pytest.raises(ValidationError):
        EventSchema(**payload)


def test_real_event_can_omit_animal_location_fields():
    payload = valid_payload()
    payload.update({
        "sourceType": "real",
        "animalEstLLA": None,
        "animalTrueLLA": None,
        "animalLLAUncertainty": None,
    })

    event = EventSchema(**payload)

    assert event.animalEstLLA is None
    assert event.animalTrueLLA is None
    assert event.animalLLAUncertainty is None
