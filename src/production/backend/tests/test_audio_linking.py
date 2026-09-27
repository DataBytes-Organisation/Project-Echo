import asyncio
import json

import pytest
from bson import ObjectId
from fastapi import HTTPException

from app.routers import audio_upload_router
from app.services import mqtt_client


class FakeAudioUploads:
    def __init__(self, document=None):
        self.document = document

    def find_one(self, query):
        return self.document


def valid_detection_payload(audio_clip):
    return {
        "timestamp": "2026-09-17T03:20:00Z",
        "sensorId": "audio-link-unit-test",
        "species": "Uperoleia mimula",
        "sourceType": "real",
        "microphoneLLA": {
            "latitude": -38.8081,
            "longitude": 143.5913,
            "altitude": 10.0,
        },
        "animalEstLLA": None,
        "animalTrueLLA": None,
        "animalLLAUncertainty": 2.5,
        "audioClip": audio_clip,
        "confidence": 98.5,
        "sampleRate": 8000,
        "source_model": "audio-link-unit-test",
    }


def test_get_audio_returns_existing_object(monkeypatch):
    upload_id = str(ObjectId())
    storage_key = "audio_uploads/2026/09/23/stored.wav"
    storage = object()

    fake_collection = FakeAudioUploads(
        {
            "_id": ObjectId(upload_id),
            "filename": "stored.wav",
            "original_filename": "original.wav",
            "storage_key": storage_key,
            "content_type": "audio/wav",
        }
    )

    monkeypatch.setattr(
        audio_upload_router,
        "AudioUploads",
        fake_collection,
    )
    monkeypatch.setattr(
        audio_upload_router,
        "get_r2_storage",
        lambda: storage,
    )

    def fake_download(storage_arg, key):
        assert storage_arg is storage
        assert key == storage_key
        return b"audio-test-data"

    monkeypatch.setattr(
        audio_upload_router,
        "download_from_r2",
        fake_download,
    )

    response = asyncio.run(
        audio_upload_router.get_audio(upload_id)
    )

    assert response.status_code == 200
    assert response.body == b"audio-test-data"
    assert response.media_type == "audio/wav"
    assert (
        response.headers["content-disposition"]
        == 'inline; filename="original.wav"'
    )


def test_invalid_audio_id_returns_400():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            audio_upload_router.get_audio(
                "not-a-valid-object-id"
            )
        )

    assert exc.value.status_code == 400


def test_missing_audio_record_returns_404(monkeypatch):
    monkeypatch.setattr(
        audio_upload_router,
        "AudioUploads",
        FakeAudioUploads(None),
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            audio_upload_router.get_audio(
                str(ObjectId())
            )
        )

    assert exc.value.status_code == 404


def test_missing_audio_object_returns_404(monkeypatch):
    upload_id = str(ObjectId())
    storage_key = "audio_uploads/missing.wav"
    storage = object()

    fake_collection = FakeAudioUploads(
        {
            "_id": ObjectId(upload_id),
            "filename": "missing.wav",
            "original_filename": "missing.wav",
            "storage_key": storage_key,
            "content_type": "audio/wav",
        }
    )

    monkeypatch.setattr(
        audio_upload_router,
        "AudioUploads",
        fake_collection,
    )
    monkeypatch.setattr(
        audio_upload_router,
        "get_r2_storage",
        lambda: storage,
    )

    def missing_object(storage_arg, key):
        assert storage_arg is storage
        assert key == storage_key
        raise audio_upload_router.R2ObjectNotFound(
            "missing"
        )

    monkeypatch.setattr(
        audio_upload_router,
        "download_from_r2",
        missing_object,
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            audio_upload_router.get_audio(upload_id)
        )

    assert exc.value.status_code == 404


def test_detection_preserves_audio_upload_reference(monkeypatch):
    upload_id = str(ObjectId())
    captured = {}

    def fake_persist(event):
        captured["event"] = event
        return "fake-event-id"

    monkeypatch.setattr(
        mqtt_client,
        "persist_event",
        fake_persist,
    )

    payload = valid_detection_payload(upload_id)

    result = mqtt_client.persist_detection_payload(
        json.dumps(payload).encode()
    )

    assert result == "fake-event-id"
    assert captured["event"].audioClip == upload_id