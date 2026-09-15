from __future__ import annotations

from typing import Any, Dict, List, Set

from bson import ObjectId

from app.database import AudioUploads, Detections, Events

C14_SEED_PREFIX = "C14.2-SEED-"
UNLINKED_AUDIO_PLACEHOLDER = "C14.2-unlinked-fixture"


def _is_upload_reference(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 24:
        return False
    try:
        ObjectId(value)
    except Exception:
        return False
    return True


def _collect_referenced_upload_ids() -> Set[str]:
    referenced: Set[str] = set()
    for collection in (Detections, Events):
        for doc in collection.find({}, {"audioClip": 1}):
            clip = doc.get("audioClip")
            if _is_upload_reference(clip):
                referenced.add(clip)
    return referenced


def run_consistency_checks() -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    upload_ids = {str(doc["_id"]) for doc in AudioUploads.find({}, {"_id": 1})}
    referenced_upload_ids = _collect_referenced_upload_ids()

    for collection_name, collection in (
        ("detections", Detections),
        ("events", Events),
    ):
        for doc in collection.find({}, {"_id": 1, "audioClip": 1, "sensorId": 1}):
            clip = doc.get("audioClip")
            if not _is_upload_reference(clip):
                continue
            if clip in upload_ids:
                continue

            issues.append(
                {
                    "check": "missing_audio_upload",
                    "collection": collection_name,
                    "record_id": str(doc["_id"]),
                    "sensorId": doc.get("sensorId"),
                    "audioClip": clip,
                    "reason": "audioClip references a missing audio_uploads record",
                }
            )

    for doc in AudioUploads.find({"fixture_marker": {"$regex": f"^{C14_SEED_PREFIX}"}}):
        upload_id = str(doc["_id"])
        if upload_id in referenced_upload_ids:
            continue

        issues.append(
            {
                "check": "orphan_audio_upload",
                "collection": "audio_uploads",
                "record_id": upload_id,
                "fixture_marker": doc.get("fixture_marker"),
                "reason": "seeded audio upload is not referenced by any detection or event",
            }
        )

    return issues


def apply_safe_fixes(issues: List[Dict[str, Any]]) -> Dict[str, int]:
    fixed = {
        "missing_audio_upload": 0,
        "orphan_audio_upload": 0,
    }

    for issue in issues:
        check = issue.get("check")

        if check == "missing_audio_upload":
            sensor_id = issue.get("sensorId") or ""
            if not str(sensor_id).startswith(C14_SEED_PREFIX):
                continue

            collection = Detections if issue["collection"] == "detections" else Events
            result = collection.update_one(
                {"_id": ObjectId(issue["record_id"])},
                {"$set": {"audioClip": UNLINKED_AUDIO_PLACEHOLDER}},
            )
            if result.modified_count:
                fixed["missing_audio_upload"] += 1

        elif check == "orphan_audio_upload":
            marker = issue.get("fixture_marker") or ""
            if not str(marker).startswith(C14_SEED_PREFIX):
                continue

            result = AudioUploads.delete_one({"_id": ObjectId(issue["record_id"])})
            if result.deleted_count:
                fixed["orphan_audio_upload"] += 1

    return fixed
