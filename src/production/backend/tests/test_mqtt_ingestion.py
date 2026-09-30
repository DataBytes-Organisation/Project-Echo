import json
from types import SimpleNamespace

from app.services import mqtt_client


def valid_payload():
    return {
        "timestamp": "2026-09-17T00:15:00Z",
        "sensorId": "mqtt-unit-test",
        "species": "Uperoleia mimula",
        "sourceType": "real",
        "microphoneLLA": {
            "latitude": -38.8081,
            "longitude": 143.5913,
            "altitude": 10.0,
        },
        "animalEstLLA": {
            "latitude": -38.8082,
            "longitude": 143.5929,
            "altitude": 4.6,
        },
        "animalTrueLLA": {
            "latitude": -38.8082,
            "longitude": 143.5929,
            "altitude": 10.0,
        },
        "animalLLAUncertainty": 2.5,
        "audioClip": "mqtt-unit-test-audio",
        "confidence": 99.36,
        "sampleRate": 32000,
        "source_model": "mqtt-unit-test-model",
    }


def test_valid_detection_is_persisted(monkeypatch):
    captured = {}

    def fake_persist(event):
        captured["event"] = event
        return "fake-event-id"

    monkeypatch.setattr(
        mqtt_client,
        "persist_event",
        fake_persist,
    )

    result = mqtt_client.persist_detection_payload(
        json.dumps(valid_payload()).encode()
    )

    assert result == "fake-event-id"
    assert captured["event"].sensorId == "mqtt-unit-test"
    assert captured["event"].species == "Uperoleia mimula"
    assert captured["event"].sourceType == "real"


def test_missing_species_is_rejected(monkeypatch):
    payload = valid_payload()
    payload.pop("species")

    called = {"value": False}

    def fake_persist(event):
        called["value"] = True
        return "should-not-be-called"

    monkeypatch.setattr(
        mqtt_client,
        "persist_event",
        fake_persist,
    )

    result = mqtt_client.persist_detection_payload(
        json.dumps(payload).encode()
    )

    assert result is None
    assert called["value"] is False


def test_invalid_json_is_rejected(monkeypatch):
    called = {"value": False}

    def fake_persist(event):
        called["value"] = True
        return "should-not-be-called"

    monkeypatch.setattr(
        mqtt_client,
        "persist_event",
        fake_persist,
    )

    result = mqtt_client.persist_detection_payload(
        b"{invalid-json"
    )

    assert result is None
    assert called["value"] is False


def test_persistence_failure_is_handled(monkeypatch):
    def fake_persist(event):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        mqtt_client,
        "persist_event",
        fake_persist,
    )

    result = mqtt_client.persist_detection_payload(
        json.dumps(valid_payload()).encode()
    )

    assert result is None


def test_detection_topic_routes_to_persistence(monkeypatch):
    received = []

    def fake_ingest(payload):
        received.append(payload)
        return "fake-event-id"

    monkeypatch.setattr(
        mqtt_client,
        "persist_detection_payload",
        fake_ingest,
    )

    message = SimpleNamespace(
        topic=mqtt_client.MQTT_DETECTION_TOPIC,
        payload=b'{"test": true}',
    )

    mqtt_client.on_message(
        None,
        None,
        message,
    )

    assert received == [b'{"test": true}']