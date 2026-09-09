import asyncio
from typing import Any, Dict, Set

from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder


class DetectionStreamManager:
    """Tracks WebSocket clients and broadcasts persisted detections."""

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, detection: Dict[str, Any]) -> None:
        message = jsonable_encoder(detection)
        async with self._lock:
            connections = list(self._connections)

        stale: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)

        if stale:
            async with self._lock:
                for websocket in stale:
                    self._connections.discard(websocket)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


detection_stream_manager = DetectionStreamManager()
