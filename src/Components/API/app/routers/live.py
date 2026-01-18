from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio

router = APIRouter()

@router.websocket("/ws")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({"message": "WebSocket test connection successful!"})
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass

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
