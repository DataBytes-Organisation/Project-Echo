import asyncio

from starlette.websockets import WebSocketDisconnect

from app.routers import live
from app.services.detection_stream import DetectionStreamManager


class FakeWebSocket:
    def __init__(self, *, token=None, authorization=None, disconnect=True):
        self.query_params = {} if token is None else {"token": token}
        self.headers = {} if authorization is None else {"authorization": authorization}
        self.disconnect = disconnect
        self.accepted = False
        self.closed_code = None
        self.sent_messages = []

    async def accept(self):
        self.accepted = True

    async def close(self, code):
        self.closed_code = code

    async def receive_text(self):
        if self.disconnect:
            raise WebSocketDisconnect()
        return "ping"

    async def send_json(self, payload):
        self.sent_messages.append(payload)


def test_valid_token_connects_then_disconnects_cleanly(monkeypatch):
    manager = DetectionStreamManager()
    websocket = FakeWebSocket(token="valid-token")

    monkeypatch.setattr(live, "decodeJWT", lambda token: {"id": "user-1"})
    monkeypatch.setattr(live, "detection_stream_manager", manager)

    asyncio.run(live.detection_stream(websocket))

    assert websocket.accepted is True
    assert websocket.closed_code is None
    assert manager.connection_count == 0


def test_invalid_token_is_rejected_before_connection(monkeypatch):
    manager = DetectionStreamManager()
    websocket = FakeWebSocket(token="invalid-token")

    monkeypatch.setattr(live, "decodeJWT", lambda token: None)
    monkeypatch.setattr(live, "detection_stream_manager", manager)

    asyncio.run(live.detection_stream(websocket))

    assert websocket.accepted is False
    assert websocket.closed_code == 1008
    assert manager.connection_count == 0


def test_manager_broadcasts_to_connected_clients_and_removes_stale_ones():
    manager = DetectionStreamManager()
    active_client = FakeWebSocket()

    class FailingWebSocket(FakeWebSocket):
        async def send_json(self, payload):
            raise RuntimeError("connection closed")

    stale_client = FailingWebSocket()

    async def exercise_stream():
        await manager.connect(active_client)
        await manager.connect(stale_client)
        await manager.broadcast({"sensorId": "stream-test", "confidence": 91.2})

    asyncio.run(exercise_stream())

    assert active_client.sent_messages == [
        {"sensorId": "stream-test", "confidence": 91.2}
    ]
    assert manager.connection_count == 1
