from datetime import datetime
from types import SimpleNamespace

from bson import ObjectId
from fastapi import BackgroundTasks, HTTPException
import pytest

from app.services import audio_jobs
from app.routers import audio_jobs as audio_jobs_router


def test_serialize_job_returns_public_job_tracking_fields():
    job = {
        "_id": ObjectId(),
        "upload_id": "upload-id",
        "filename": "recording.wav",
        "status": "completed",
        "attempts": 1,
        "created_at": datetime(2026, 9, 20, 1, 0),
        "updated_at": datetime(2026, 9, 20, 1, 1),
        "result": {"species": "Crimson Rosella", "confidence": 0.92},
    }

    payload = audio_jobs.serialize_job(job)

    assert payload["job_id"] == str(job["_id"])
    assert payload["status"] == "completed"
    assert payload["result"]["species"] == "Crimson Rosella"
    assert payload["error"] is None


def test_invalid_job_id_is_rejected():
    try:
        audio_jobs._job_id("not-an-object-id")
    except ValueError as error:
        assert str(error) == "Invalid job ID"
    else:
        raise AssertionError("An invalid job ID must be rejected")


def test_retry_rejects_an_invalid_job_id_with_bad_request():
    with pytest.raises(HTTPException) as error:
        audio_jobs_router.retry_audio_job("not-an-object-id", BackgroundTasks())

    assert error.value.status_code == 400
    assert error.value.detail == "Invalid job ID"


def test_process_job_marks_unexpected_prediction_errors_as_failed(monkeypatch, tmp_path):
    job_id = ObjectId()
    updates = []
    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(
        audio_jobs.AudioProcessingJobs,
        "find_one_and_update",
        lambda *_args, **_kwargs: {"_id": job_id, "file_path": str(audio_path)},
    )
    monkeypatch.setattr(
        audio_jobs,
        "predict_with_failure_detection",
        lambda _audio: (_ for _ in ()).throw(RuntimeError("unexpected model error")),
    )
    monkeypatch.setattr(
        audio_jobs.AudioProcessingJobs,
        "update_one",
        lambda *args, **kwargs: updates.append((args, kwargs)),
    )

    audio_jobs.process_job(str(job_id))

    assert updates[-1][0][1]["$set"]["status"] == "failed"
    assert updates[-1][0][1]["$set"]["error"] == "unexpected model error"


def test_store_audio_job_removes_upload_metadata_when_job_creation_fails(monkeypatch, tmp_path):
    upload_id = ObjectId()
    file_path = tmp_path / "recording.wav"
    deleted = []

    monkeypatch.setattr(
        audio_jobs_router.AudioUploads,
        "insert_one",
        lambda _upload: SimpleNamespace(inserted_id=upload_id),
    )
    monkeypatch.setattr(
        audio_jobs_router.AudioUploads,
        "delete_one",
        lambda query: deleted.append(query),
    )
    monkeypatch.setattr(
        audio_jobs_router.audio_jobs,
        "create_job",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("job insert failed")),
    )

    with pytest.raises(RuntimeError, match="job insert failed"):
        audio_jobs_router._store_audio_job(
            str(file_path), b"audio", "recording.wav", "recording.wav", "audio/wav", None
        )

    assert deleted == [{"_id": upload_id}]
    assert not file_path.exists()
