from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas import EventSchema
from app.routers import engine


def valid_payload():
    return {
        "timestamp": "2026-08-31T01:00:00Z",
        "sensorId": "unit-test-sensor",
        "species": "Uperoleia mimula",
        "microphoneLLA": [-38.8081, 143.5913, 10.0],
        "animalEstLLA": [-38.8082, 143.5929, 4.6],
        "animalTrueLLA": [-38.8082, 143.5929, 10.0],
        "animalLLAUncertainty": 0,
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


def test_create_event_inserts_into_events_collection(monkeypatch):
    fake_events = FakeEventsCollection()
    monkeypatch.setattr(engine, "Events", fake_events)

    event = EventSchema(**valid_payload())
    response = engine.create_event(event)

    assert fake_events.inserted_document is not None
    assert fake_events.inserted_document["sensorId"] == "unit-test-sensor"
    assert fake_events.inserted_document["species"] == "Uperoleia mimula"
    assert fake_events.inserted_document["confidence"] == 99.36

    assert response["_id"] == "fake-event-id"
    assert response["sensorId"] == "unit-test-sensor"
    assert response["species"] == "Uperoleia mimula"


def test_empty_species_is_rejected():
    payload = valid_payload()
    payload["species"] = ""

    with pytest.raises(ValidationError):
        EventSchema(**payload)


def test_confidence_100_is_rejected():
    payload = valid_payload()
    payload["confidence"] = 100

    with pytest.raises(ValidationError):
        EventSchema(**payload)


def test_invalid_microphone_location_length_is_rejected():
    payload = valid_payload()
    payload["microphoneLLA"] = [-38.8081, 143.5913]

    with pytest.raises(ValidationError):
        EventSchema(**payload)
