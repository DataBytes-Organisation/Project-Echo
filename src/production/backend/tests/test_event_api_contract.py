from app import errors, serializers
from app.routers import hmi


def test_failed_error_envelope_has_the_engine_ack_shape():
    payload = errors.error_body(422, "Request validation failed.", [{"loc": ["body"]}])

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["message"] == "Request validation failed."
    assert payload["error"]["details"] == [{"loc": ["body"]}]


def test_event_serializer_exposes_source_type():
    event = {
        "_id": "event-id",
        "timestamp": "2026-09-12T00:00:00Z",
        "sensorId": "sensor-1",
        "sourceType": "real",
        "species": "Litoria ewingii",
        "microphoneLLA": {"latitude": -37.8, "longitude": 144.9, "altitude": 10},
        "animalEstLLA": None,
        "animalTrueLLA": None,
        "animalLLAUncertainty": None,
        "confidence": 87.5,
    }

    assert serializers.eventEntity(event)["sourceType"] == "real"


def test_hmi_latest_events_filters_by_source_type(monkeypatch):
    event = {
        "_id": "event-id",
        "timestamp": "2026-09-12T00:00:00Z",
        "sensorId": "sensor-1",
        "sourceType": "real",
        "species": "Litoria ewingii",
        "microphoneLLA": {"latitude": -37.8, "longitude": 144.9, "altitude": 10},
        "animalEstLLA": None,
        "animalTrueLLA": None,
        "animalLLAUncertainty": None,
        "confidence": 87.5,
    }

    class FakeEvents:
        pipeline = None

        @classmethod
        def aggregate(cls, pipeline):
            cls.pipeline = pipeline
            return [event]

    monkeypatch.setattr(hmi, "Events", FakeEvents)

    response = hmi.show_latest_events(sourceType="real")

    assert FakeEvents.pipeline[0] == {"$match": {"sourceType": "real"}}
    assert response[0]["sourceType"] == "real"
