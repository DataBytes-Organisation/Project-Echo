from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.middleware.auth import decodeJWT
from app.services.detection_stream import detection_stream_manager

router = APIRouter()


def _extract_ws_token(websocket: WebSocket) -> Optional[str]:
    token = websocket.query_params.get("token")
    if token:
        return token

    auth_header = websocket.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip() or None

    return None


@router.websocket("/ws")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/detections")
async def detection_stream(websocket: WebSocket):
    token = _extract_ws_token(websocket)
    payload = decodeJWT(token) if token else None
    if not payload:
        # Reject unauthenticated wildlife location streams before accept.
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await detection_stream_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await detection_stream_manager.disconnect(websocket)
