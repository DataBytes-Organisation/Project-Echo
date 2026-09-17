import json

import pytest
from bson import ObjectId
from fastapi import HTTPException
from fastapi.responses import FileResponse

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


def test_get_audio_returns_existing_file(monkeypatch, tmp_path):
    upload_id = str(ObjectId())

    audio_file = tmp_path / "stored.wav"
    audio_file.write_bytes(b"audio-test-data")

    fake_collection = FakeAudioUploads(
        {
            "_id": ObjectId(upload_id),
            "filename": "stored.wav",
            "original_filename": "original.wav",
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
        "UPLOAD_DIR",
        str(tmp_path),
    )

    response = audio_upload_router.get_audio(upload_id)

    assert isinstance(response, FileResponse)
    assert response.path == str(audio_file)
    assert response.media_type == "audio/wav"


def test_invalid_audio_id_returns_400():
    with pytest.raises(HTTPException) as exc:
        audio_upload_router.get_audio("not-a-valid-object-id")

    assert exc.value.status_code == 400


def test_missing_audio_record_returns_404(monkeypatch):
    monkeypatch.setattr(
        audio_upload_router,
        "AudioUploads",
        FakeAudioUploads(None),
    )

    with pytest.raises(HTTPException) as exc:
        audio_upload_router.get_audio(str(ObjectId()))

    assert exc.value.status_code == 404


def test_missing_audio_file_returns_404(monkeypatch, tmp_path):
    upload_id = str(ObjectId())

    fake_collection = FakeAudioUploads(
        {
            "_id": ObjectId(upload_id),
            "filename": "missing.wav",
            "original_filename": "missing.wav",
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
        "UPLOAD_DIR",
        str(tmp_path),
    )

    with pytest.raises(HTTPException) as exc:
        audio_upload_router.get_audio(upload_id)

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