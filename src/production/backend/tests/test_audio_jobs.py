from datetime import datetime

from bson import ObjectId

from app.services import audio_jobs


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
