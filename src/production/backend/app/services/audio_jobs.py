"""Background audio prediction jobs with persistent status tracking."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from bson import ObjectId
from pymongo import ReturnDocument

from app.database import AudioProcessingJobs, Predictions
from app.services.model_adapter import MultiModalPredictionError, predict_with_failure_detection


VALID_STATUSES = {"queued", "processing", "completed", "failed"}


def _job_id(job_id: str) -> ObjectId:
    if not ObjectId.is_valid(job_id):
        raise ValueError("Invalid job ID")
    return ObjectId(job_id)


def create_job(upload_id: str, file_path: str, filename: str, user_id: Optional[str]) -> Dict[str, Any]:
    now = datetime.utcnow()
    document = {
        "upload_id": upload_id,
        "file_path": file_path,
        "filename": filename,
        "user_id": user_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "attempts": 0,
    }
    result = AudioProcessingJobs.insert_one(document)
    document["_id"] = result.inserted_id
    return document


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return AudioProcessingJobs.find_one({"_id": _job_id(job_id)})


def serialize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": str(job["_id"]),
        "upload_id": job["upload_id"],
        "filename": job["filename"],
        "status": job["status"],
        "attempts": job.get("attempts", 0),
        "created_at": job["created_at"].isoformat(),
        "updated_at": job["updated_at"].isoformat(),
        "result": job.get("result"),
        "error": job.get("error"),
    }


def process_job(job_id: str) -> None:
    """Run the configured prediction adapter and record a terminal job state."""
    oid = _job_id(job_id)
    job = AudioProcessingJobs.find_one_and_update(
        {"_id": oid, "status": "queued"},
        {"$set": {"status": "processing", "updated_at": datetime.utcnow()}, "$inc": {"attempts": 1}},
    )
    if not job:
        return

    try:
        with Path(job["file_path"]).open("rb") as audio_file:
            prediction = predict_with_failure_detection(audio_file)
        now = datetime.utcnow()
        Predictions.insert_one({
            "filename": job["filename"],
            "upload_id": job["upload_id"],
            "user_id": job.get("user_id"),
            "predicted_species": prediction["species"],
            "confidence": prediction["confidence"],
            "timestamp": now,
        })
        AudioProcessingJobs.update_one(
            {"_id": oid},
            {"$set": {"status": "completed", "updated_at": now, "result": prediction}, "$unset": {"error": ""}},
        )
    except (OSError, MultiModalPredictionError) as exc:
        AudioProcessingJobs.update_one(
            {"_id": oid},
            {"$set": {"status": "failed", "updated_at": datetime.utcnow(), "error": str(exc)}},
        )


def retry_job(job_id: str) -> Dict[str, Any]:
    oid = _job_id(job_id)
    updated = AudioProcessingJobs.find_one_and_update(
        {"_id": oid, "status": "failed"},
        {"$set": {"status": "queued", "updated_at": datetime.utcnow()}, "$unset": {"error": "", "result": ""}},
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise ValueError("Only failed jobs can be retried")
    return updated
