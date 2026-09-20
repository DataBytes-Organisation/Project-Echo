"""
C14.1 - Database Consistency Checker (report-only, with optional safe-fix)

Objective:
Detect orphaned/inconsistent references in Project Echo data, focusing on
detection-to-audio relationships (Detection.audioClip -> AudioUploads._id).

Scanning is delegated to the shared checker in app/services/consistency_checker.py
(introduced in PR #1033) so there is a single source of truth for what counts as
a broken link. This tool adds the optional, conservative repair step on top.

Usage:
    python -m tools.db_consistency_check --report-only
    python -m tools.db_consistency_check --safe-fix --confirm

Default mode is report-only. Safe-fix only runs with --confirm, and only
touches records that were flagged as inconsistent in this run.
"""

import argparse
import sys

from bson import ObjectId

from app.database import Detections, Events
from app.services.consistency_checker import run_consistency_checks

# audioClip is a required string in the app's models, so None is not valid.
# The app already uses an empty string to mean "no audio clip" (serializers.py).
NO_AUDIO_CLIP = ""

COLLECTIONS = {"detections": Detections, "events": Events}


def find_broken_audio_links():
    """Report-only (no writes). Uses the shared #1033 scan and keeps only broken
    detection/event -> audio upload links."""
    return [i for i in run_consistency_checks() if i.get("check") == "missing_audio_upload"]


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
    Conservative fix: replace a broken audioClip reference with an empty string,
    the app's own "no audio clip" value, so the record stays valid against the
    existing string-field models. Does not delete the detection or event.

    Each update also matches the audioClip value seen at scan time, so a record
    whose link was changed (for example fixed by someone else) after the scan
    is skipped, not overwritten.
    """
    fixed, skipped = [], []
    for issue in issues:
        collection = COLLECTIONS[issue["collection"]]
        result = collection.update_one(
            {"_id": ObjectId(issue["record_id"]), "audioClip": issue["audioClip"]},
            {"$set": {"audioClip": NO_AUDIO_CLIP}},
        )
        if result.modified_count:
            fixed.append(issue)
        else:
            skipped.append(issue)
    return fixed, skipped


def verify_repairs(fixed):
    """Read each repaired record back and re-run the shared scan."""
    bad_value = 0
    for issue in fixed:
        doc = COLLECTIONS[issue["collection"]].find_one(
            {"_id": ObjectId(issue["record_id"])}, {"audioClip": 1}
        )
        if doc is None or doc.get("audioClip") != NO_AUDIO_CLIP:
            bad_value += 1

    flagged_again = {i["record_id"] for i in find_broken_audio_links()}
    reflagged = sum(1 for i in fixed if i["record_id"] in flagged_again)
    return bad_value, reflagged


def main():
    parser = argparse.ArgumentParser(description="Project Echo database consistency checker")
    parser.add_argument("--report-only", action="store_true", default=True,
                         help="Report issues without changing any data (default).")
    parser.add_argument("--safe-fix", action="store_true",
                         help="Replace broken audioClip references with an empty string. Requires --confirm.")
    parser.add_argument("--confirm", action="store_true",
                         help="Required alongside --safe-fix to actually apply changes.")
    args = parser.parse_args()

    issues = find_broken_audio_links()
    print_report(issues)

    if args.safe_fix:
        if not args.confirm:
            print("\n--safe-fix was passed without --confirm. No changes made.")
            sys.exit(0)
        fixed, skipped = apply_conservative_fix(issues)
        print(f"\nSafe-fix applied: reset audioClip to an empty string on {len(fixed)} record(s).")
        if skipped:
            print(f"Skipped {len(skipped)} record(s) whose audioClip changed since the scan.")
        bad_value, reflagged = verify_repairs(fixed)
        print(f"Verification: {bad_value} record(s) with an unexpected value, "
              f"{reflagged} record(s) still flagged by the checker.")


if __name__ == "__main__":
    main()
