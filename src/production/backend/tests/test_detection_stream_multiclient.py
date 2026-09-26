from types import SimpleNamespace
import asyncio

from app.schemas import EventSchema
from app.routers import engine
from app.services.detection_stream import DetectionStreamManager


def valid_payload():
    return {
        "timestamp": "2026-08-31T01:00:00Z",
        "sensorId": "b1-3-test-sensor",
        "species": "Uperoleia mimula",
        "sourceType": "simulator",
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
        "animalLLAUncertainty": 0.0,
        "audioClip": "b1-3-test-audio",
        "confidence": 94.5,
        "sampleRate": 32000,
    }


class FakeEventsCollection:
    def __init__(self):
        self.inserted_document = None
        self.inserted_id = "b1-3-event-001"

    def insert_one(self, document):
        self.inserted_document = document.copy()
        self.inserted_document["_id"] = self.inserted_id

        return SimpleNamespace(
            inserted_id=self.inserted_id
        )

    def aggregate(self, pipeline):
        return [self.inserted_document]


class FakeWebSocket:
    def __init__(
        self,
        *,
        persisted_events=None,
        fail_on_send=False,
    ):
        self.accepted = False
        self.sent_messages = []
        self.persisted_events = persisted_events
        self.fail_on_send = fail_on_send

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        if self.persisted_events is not None:
            assert (
                self.persisted_events.inserted_document
                is not None
            )

        if self.fail_on_send:
            raise RuntimeError(
                "simulated disconnected client"
            )

        self.sent_messages.append(payload)


def test_two_clients_receive_same_persisted_detection(
    monkeypatch,
):
    fake_events = FakeEventsCollection()

    client_a = FakeWebSocket(
        persisted_events=fake_events
    )
    client_b = FakeWebSocket(
        persisted_events=fake_events
    )

    async def scenario():
        # Create the manager while an event loop is running.
        manager = DetectionStreamManager()

        monkeypatch.setattr(
            engine,
            "Events",
            fake_events,
        )
        monkeypatch.setattr(
            engine,
            "invalidate_insights",
            lambda: None,
        )
        monkeypatch.setattr(
            engine,
            "detection_stream_manager",
            manager,
        )

        await manager.connect(client_a)
        await manager.connect(client_b)

        assert manager.connection_count == 2

        event = EventSchema(**valid_payload())

        response = await engine.create_event(event)

        return response, manager

    response, manager = asyncio.run(scenario())

    assert response == {
        "status": "success",
        "eventId": "b1-3-event-001",
    }

    assert manager.connection_count == 2

    assert len(client_a.sent_messages) == 1
    assert len(client_b.sent_messages) == 1

    message_a = client_a.sent_messages[0]
    message_b = client_b.sent_messages[0]

    assert message_a == message_b

    assert message_a["_id"] == "b1-3-event-001"
    assert message_a["sensorId"] == "b1-3-test-sensor"
    assert message_a["species"] == "Uperoleia mimula"
    assert message_a["sourceType"] == "simulator"
    assert message_a["confidence"] == 94.5


def test_disconnected_client_stops_receiving_without_affecting_other_client():
    departing_client = FakeWebSocket()
    remaining_client = FakeWebSocket()

    first_detection = {
        "_id": "event-001",
        "sensorId": "sensor-2",
        "species": "Dingo",
        "confidence": 91.0,
    }

    second_detection = {
        "_id": "event-002",
        "sensorId": "sensor-3",
        "species": "Sus Scrofa",
        "confidence": 88.0,
    }

    async def scenario():
        manager = DetectionStreamManager()

        await manager.connect(departing_client)
        await manager.connect(remaining_client)

        assert manager.connection_count == 2

        await manager.broadcast(first_detection)

        await manager.disconnect(
            departing_client
        )

        assert manager.connection_count == 1

        await manager.broadcast(second_detection)

        return manager

    manager = asyncio.run(scenario())

    assert departing_client.sent_messages == [
        first_detection
    ]

    assert remaining_client.sent_messages == [
        first_detection,
        second_detection,
    ]

    assert manager.connection_count == 1


def test_reconnected_client_receives_future_detections():
    original_client = FakeWebSocket()
    remaining_client = FakeWebSocket()
    reconnected_client = FakeWebSocket()

    while_disconnected = {
        "_id": "event-002",
        "sensorId": "sensor-2",
        "species": "Crimson Rosella",
        "confidence": 87.0,
    }

    after_reconnect = {
        "_id": "event-003",
        "sensorId": "sensor-3",
        "species": "Dingo",
        "confidence": 93.0,
    }

    async def scenario():
        manager = DetectionStreamManager()

        await manager.connect(original_client)
        await manager.connect(remaining_client)

        assert manager.connection_count == 2

        await manager.disconnect(
            original_client
        )

        assert manager.connection_count == 1

        await manager.broadcast(
            while_disconnected
        )

        # A refreshed/reconnected browser creates a new socket.
        await manager.connect(
            reconnected_client
        )

        assert manager.connection_count == 2

        await manager.broadcast(
            after_reconnect
        )

        return manager

    manager = asyncio.run(scenario())

    assert original_client.sent_messages == []

    assert remaining_client.sent_messages == [
        while_disconnected,
        after_reconnect,
    ]

    assert reconnected_client.sent_messages == [
        after_reconnect
    ]

    assert manager.connection_count == 2


def test_stale_client_does_not_block_healthy_clients():
    healthy_a = FakeWebSocket()
    stale_client = FakeWebSocket(
        fail_on_send=True
    )
    healthy_b = FakeWebSocket()

    detection = {
        "_id": "event-004",
        "sensorId": "sensor-2",
        "species": "Sus Scrofa",
        "confidence": 90.0,
    }

    async def scenario():
        manager = DetectionStreamManager()

        await manager.connect(healthy_a)
        await manager.connect(stale_client)
        await manager.connect(healthy_b)

        assert manager.connection_count == 3

        await manager.broadcast(detection)

        return manager

    manager = asyncio.run(scenario())

    assert healthy_a.sent_messages == [
        detection
    ]
    assert healthy_b.sent_messages == [
        detection
    ]
    assert stale_client.sent_messages == []

    assert manager.connection_count == 2