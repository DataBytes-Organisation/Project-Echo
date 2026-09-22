import asyncio
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder


class DetectionStreamManager:
    """Tracks WebSocket clients and broadcasts persisted detections."""

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        # Lazily create the lock so import-time construction works on Python 3.9
        # when no event loop exists yet (common under pytest collection).
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._get_lock():
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._get_lock():
            self._connections.discard(websocket)

    async def broadcast(self, detection: Dict[str, Any]) -> None:
        message = jsonable_encoder(detection)
        async with self._get_lock():
            connections = list(self._connections)

        stale: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)

        if stale:
            async with self._get_lock():
                for websocket in stale:
                    self._connections.discard(websocket)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


detection_stream_manager = DetectionStreamManager()
