"""
C14.1 - Database Consistency Checker (report-only, with optional safe-fix)

Objective:
Detect orphaned/inconsistent references in Project Echo data, focusing on
detection-to-audio relationships (Detection.audioClip -> AudioUploads._id).

Usage:
    python -m tools.db_consistency_check --report-only
    python -m tools.db_consistency_check --safe-fix --confirm

Default mode is report-only. Safe-fix only runs with --confirm, and only
touches records that were flagged as inconsistent in this run.
"""

import argparse
import sys

from bson import ObjectId
from bson.errors import InvalidId

from app.database import AudioUploads, Detections, Events


def looks_like_upload_id(value):
    """Return True if value looks like a Mongo ObjectId string (24 hex chars)."""
    if not isinstance(value, str):
        return False
    try:
        ObjectId(value)
        return True
    except (InvalidId, TypeError):
        return False


def find_broken_audio_links():
    """
    Scan detections and events for audioClip values that look like an
    upload reference but do not match any real AudioUploads record.
    Returns a list of plain-dict issues, report-only (no writes).
    """
    known_upload_ids = set()
    for doc in AudioUploads.find({}, {"_id": 1}):
        known_upload_ids.add(str(doc["_id"]))

    issues = []
    for label, collection in (("detections", Detections), ("events", Events)):
        cursor = collection.find({}, {"_id": 1, "audioClip": 1, "sensorId": 1})
        for doc in cursor:
            audio_clip = doc.get("audioClip")

            if not looks_like_upload_id(audio_clip):
                # Not a reference-shaped value (e.g. empty, a URL, etc.) - skip.
                continue

            if audio_clip in known_upload_ids:
                continue

            issues.append({
                "collection": label,
                "record_id": str(doc["_id"]),
                "sensorId": doc.get("sensorId"),
                "audioClip": audio_clip,
                "reason": "audioClip does not match any AudioUploads record",
            })

    return issues


def print_report(issues):
    if not issues:
        print("No inconsistencies found.")
        return

    print(f"Found {len(issues)} inconsistent record(s):\n")
    for i, issue in enumerate(issues, start=1):
        print(f"{i}. [{issue['collection']}] record_id={issue['record_id']}")
        print(f"   sensorId={issue['sensorId']}")
        print(f"   audioClip={issue['audioClip']}")
        print(f"   reason: {issue['reason']}\n")


def apply_conservative_fix(issues):
    """
    Conservative fix: clear the broken audioClip reference so the record no
    longer points at a non-existent upload. Does not delete the detection or
    event itself, and does not guess a replacement value.
    """
    fixed_count = 0
    for issue in issues:
        collection = Detections if issue["collection"] == "detections" else Events
        result = collection.update_one(
            {"_id": ObjectId(issue["record_id"])},
            {"$set": {"audioClip": None}},
        )
        if result.modified_count:
            fixed_count += 1
    return fixed_count


def main():
    parser = argparse.ArgumentParser(description="Project Echo database consistency checker")
    parser.add_argument("--report-only", action="store_true", default=True,
                         help="Report issues without changing any data (default).")
    parser.add_argument("--safe-fix", action="store_true",
                         help="Clear broken audioClip references. Requires --confirm.")
    parser.add_argument("--confirm", action="store_true",
                         help="Required alongside --safe-fix to actually apply changes.")
    args = parser.parse_args()

    issues = find_broken_audio_links()
    print_report(issues)

    if args.safe_fix:
        if not args.confirm:
            print("\n--safe-fix was passed without --confirm. No changes made.")
            sys.exit(0)
        fixed = apply_conservative_fix(issues)
        print(f"\nSafe-fix applied: cleared audioClip on {fixed} record(s).")


if __name__ == "__main__":
    main()
