"""Regression coverage for MongoDB IDs in detection-list responses."""

import asyncio
import json
import sys
import types
from datetime import datetime

from bson import ObjectId
from fastapi import FastAPI

# Importing schemas only requires these constants. Avoid connecting to a real
# MongoDB instance during this isolated serialization regression.
database_stub = types.ModuleType("app.database")
database_stub.GENDER = []
database_stub.STATES_CODE = []
database_stub.AUS_STATES = []
sys.modules.setdefault("app.database", database_stub)

from app.schemas import Detection, DetectionListResponses


def test_saved_detection_list_returns_valid_json_over_http():
    detection_id = ObjectId()

    saved_detection = Detection(
        _id=detection_id,
        timestamp=datetime(2026, 9, 1),
        sensorId="sensor-1",
        species="Koala",
        microphoneLLA=[-37.8, 144.9, 10],
        animalEstLLA=[-37.8, 144.9, 10],
        animalTrueLLA=[-37.8, 144.9, 10],
        animalLLAUncertainty=5,
        audioClip="test",
        confidence=95,
        sampleRate=48000,
    )

    app = FastAPI()

    @app.get(
        "/detections",
        response_model=DetectionListResponses,
    )
    def list_saved_detections():
        return {
            "items": [saved_detection],
            "total": 1,
            "page": 1,
            "page_size": 20,
        }

    messages = []

    async def receive():
        return {
            "type": "http.request",
            "body": b"",
            "more_body": False,
        }

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/detections",
        "raw_path": b"/detections",
        "query_string": b"",
        "headers": [],
        "client": ("test", 50000),
        "server": ("test", 80),
        "root_path": "",
    }

    asyncio.run(app(scope, receive, send))

    response_start = next(
        message
        for message in messages
        if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    payload = json.loads(response_body)

    assert response_start["status"] == 200
    assert payload["items"][0]["_id"] == str(detection_id)
