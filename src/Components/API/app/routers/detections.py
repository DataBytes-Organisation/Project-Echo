from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import random
from datetime import datetime
import json

router = APIRouter()

# Keep track of connected clients
active_connections: list[WebSocket] = []

# Example species list for simulation
species_list = ["Magpie", "Kookaburra", "Lorikeet", "Cockatoo", "Curlew"]

async def broadcast(message: dict):
    """Send a message to all connected clients."""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            pass  # Ignore failed sends

def log_event(event: dict):
    """Append detection event to detections.log file."""
    with open("detections.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

@router.websocket("/detections")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"✅ Client connected. Total connections: {len(active_connections)}")

    try:
        while True:
            # Wait for any message from client
            await websocket.receive_text()

            # Simulate a detection event
            detected_species = random.choice(species_list)
            confidence = random.randint(70, 99)
            location = [
                round(random.uniform(144.90, 145.00), 4),  # longitude
                round(random.uniform(-37.80, -37.85), 4)   # latitude
            ]

            event = {
                "species": detected_species,
                "confidence": confidence,
                "location": location,
                "timestamp": datetime.utcnow().isoformat()
            }

            # 🔎 Log event in terminal
            print(f"📡 Detection event: {event}")

            # 📝 Save event to detections.log
            log_event(event)

            # Broadcast to all connected clients
            await broadcast(event)

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"❌ Client disconnected. Total connections: {len(active_connections)}")