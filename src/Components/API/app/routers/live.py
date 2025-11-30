from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio

router = APIRouter()

# --- WebSocket endpoint for real-time detections ---
@router.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            #Replace/add actual detection logic 
            detection = {"message": "This is a test detection. Replace with real data."}
            await websocket.send_json(detection)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
