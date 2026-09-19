"""
Project Echo - C13 Backend Seed and Fixture Generator CLI.

Creates deterministic, schema-valid Project Echo detection fixtures for:
- local development
- B1 real-time streaming tests/demos
- B2 analytics verification
- automated/integration tests

Examples
--------
Dry run:
    python -m tools.seed_detections --count 20 --species "Sus Scrofa" --dry-run

Seed events for B1/B2:
    python -m tools.seed_detections --count 100 --days 14 --sensor-id 2

Seed the detections collection:
    python -m tools.seed_detections --count 50 --collection detections

Deterministic date-bounded dataset:
    python -m tools.seed_detections \
        --count 100 \
        --start 2026-09-01 \
        --end 2026-09-10 \
        --seed 374

Safe cleanup:
    python -m tools.seed_detections \
        --cleanup \
        --database EchoNet \
        --environment local \
        --collection events \
        --confirm-cleanup
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.database import client as project_mongo_client
from app.schemas import DetectionCreate


GENERATOR_NAME = "c13_seed_detections"
DEFAULT_DATABASE = "EchoNet"
DEFAULT_COLLECTION = "events"
DEFAULT_SEED = 374
DEFAULT_DAYS = 14

ALLOWED_COLLECTIONS = ("events", "detections")

DEFAULT_SPECIES = (
    "Sus Scrofa",
    "Crimson Rosella",
    "Dingo",
    "Uperoleia mimula",
)

DEFAULT_SENSOR_IDS = (
    "1",
    "2",
    "3",
    "4",
)

SAMPLE_RATES = (
    16000,
    32000,
    44100,
    48000,
)

# Representative Australian coordinates used only for synthetic fixtures.
BASE_LATITUDE = -33.1101
BASE_LONGITUDE = 150.0567
BASE_ALTITUDE = 23.0


class SeedGeneratorError(Exception):
    """Raised for predictable C13 generator or safety failures."""


def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def parse_datetime(value: str) -> datetime:
    """
    Parse ISO-8601 date/datetime input.

    Accepts examples such as:
      2026-09-01
      2026-09-01T12:30:00
      2026-09-01T12:30:00Z
      2026-09-01T12:30:00+10:00

    Returned values are always timezone-aware UTC datetimes.
    """
    text = value.strip()

    if not text:
        raise argparse.ArgumentTypeError("Date/time value cannot be empty.")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ISO-8601 date/time: {value}"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    return parsed


def resolve_date_range(
    start: Optional[datetime],
    end: Optional[datetime],
    days: Optional[int],
) -> Tuple[datetime, datetime]:
    """
    Resolve the fixture timestamp range.

    Rules:
    - --start + --end gives an explicit deterministic range.
    - --days creates a range ending at --end or current UTC time.
    - With no options, use the previous DEFAULT_DAYS.
    """
    if days is not None and days < 1:
        raise SeedGeneratorError("--days must be at least 1.")

    if start is not None and days is not None:
        raise SeedGeneratorError(
            "Use either --start/--end or --days; "
            "do not combine --start with --days."
        )

    if start is not None:
        if end is None:
            raise SeedGeneratorError(
                "--start requires --end so the fixture range is explicit."
            )

        if start > end:
            raise SeedGeneratorError(
                "--start must be earlier than or equal to --end."
            )

        return start, end

    resolved_end = end or utc_now()
    resolved_days = days or DEFAULT_DAYS
    resolved_start = resolved_end - timedelta(days=resolved_days)

    return resolved_start, resolved_end


def evenly_spaced_timestamp(
    index: int,
    count: int,
    start: datetime,
    end: datetime,
) -> datetime:
    """Produce deterministic timestamps distributed across the range."""
    if count <= 1:
        return start

    fraction = index / (count - 1)
    return start + ((end - start) * fraction)


def jitter_location(
    rng: random.Random,
    base_lat: float,
    base_lon: float,
    base_alt: float,
    horizontal_jitter: float = 0.01,
    vertical_jitter: float = 5.0,
) -> List[float]:
    """Create a deterministic synthetic LLA coordinate."""
    return [
        round(base_lat + rng.uniform(-horizontal_jitter, horizontal_jitter), 6),
        round(base_lon + rng.uniform(-horizontal_jitter, horizontal_jitter), 6),
        round(base_alt + rng.uniform(-vertical_jitter, vertical_jitter), 2),
    ]


def validate_detection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate fixture data through Project Echo's real DetectionCreate schema.

    The returned dictionary contains only schema-approved detection fields.
    """
    validated = DetectionCreate(**payload)
    return validated.dict(by_alias=True)


def generate_detection_payload(
    *,
    index: int,
    count: int,
    start: datetime,
    end: datetime,
    species_values: Sequence[str],
    sensor_ids: Sequence[str],
    rng: random.Random,
    run_id: str,
) -> Dict[str, Any]:
    """Generate one realistic and schema-valid Project Echo fixture."""
    timestamp = evenly_spaced_timestamp(
        index=index,
        count=count,
        start=start,
        end=end,
    )

    # Cycling rather than randomly choosing species/sensors makes aggregate
    # totals predictable while still allowing other values to vary by seed.
    species = species_values[index % len(species_values)]
    sensor_id = sensor_ids[index % len(sensor_ids)]

    microphone_lla = jitter_location(
        rng,
        BASE_LATITUDE,
        BASE_LONGITUDE,
        BASE_ALTITUDE,
    )

    animal_true_lla = jitter_location(
        rng,
        microphone_lla[0],
        microphone_lla[1],
        microphone_lla[2],
        horizontal_jitter=0.004,
        vertical_jitter=3.0,
    )

    animal_est_lla = jitter_location(
        rng,
        animal_true_lla[0],
        animal_true_lla[1],
        animal_true_lla[2],
        horizontal_jitter=0.0015,
        vertical_jitter=1.5,
    )

    raw_payload: Dict[str, Any] = {
        "timestamp": timestamp,
        "sensorId": sensor_id,
        "species": species,
        "microphoneLLA": microphone_lla,
        "animalEstLLA": animal_est_lla,
        "animalTrueLLA": animal_true_lla,
        "animalLLAUncertainty": rng.randint(1, 30),
        "audioClip": f"fixture://{run_id}/audio-{index + 1:05d}.wav",
        "confidence": round(rng.uniform(70.0, 99.9), 2),
        "sampleRate": SAMPLE_RATES[index % len(SAMPLE_RATES)],
        "source_model": "c13-fixture-generator",
    }

    return validate_detection(raw_payload)


def attach_fixture_metadata(
    detection: Dict[str, Any],
    *,
    run_id: str,
    seed: int,
    environment: str,
) -> Dict[str, Any]:
    """
    Add metadata only after DetectionCreate validation.

    This metadata lets cleanup identify generated records without touching
    real Project Echo detections.
    """
    document = dict(detection)

    document["_fixture"] = {
        "generated": True,
        "generator": GENERATOR_NAME,
        "runId": run_id,
        "seed": seed,
        "environment": environment,
        "createdAt": utc_now(),
    }

    return document


def generate_fixtures(
    *,
    count: int,
    start: datetime,
    end: datetime,
    species_values: Sequence[str],
    sensor_ids: Sequence[str],
    seed: int,
    run_id: str,
    environment: str,
) -> List[Dict[str, Any]]:
    """Generate a deterministic list of validated fixture documents."""
    if count < 1:
        raise SeedGeneratorError("--count must be at least 1.")

    if not species_values:
        raise SeedGeneratorError("At least one species is required.")

    if not sensor_ids:
        raise SeedGeneratorError("At least one sensor ID is required.")

    rng = random.Random(seed)

    fixtures: List[Dict[str, Any]] = []

    for index in range(count):
        validated = generate_detection_payload(
            index=index,
            count=count,
            start=start,
            end=end,
            species_values=species_values,
            sensor_ids=sensor_ids,
            rng=rng,
            run_id=run_id,
        )

        fixtures.append(
            attach_fixture_metadata(
                validated,
                run_id=run_id,
                seed=seed,
                environment=environment,
            )
        )

    return fixtures


def get_collection(
    *,
    database_name: str,
    collection_name: str,
) -> Collection:
    """
    Return a MongoDB collection using Project Echo's existing
    Backend database connection.

    Reusing app.database avoids creating a second MongoDB
    configuration path and keeps C13 aligned with the running API.
    """
    try:
        project_mongo_client.admin.command("ping")
    except PyMongoError as exc:
        raise SeedGeneratorError(
            "Unable to connect to MongoDB through the "
            "Project Echo Backend database configuration."
        ) from exc

    database = project_mongo_client[database_name]
    return database[collection_name]


def build_fixture_filter(
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a filter that can only target C13-generated records."""
    fixture_filter: Dict[str, Any] = {
        "_fixture.generated": True,
        "_fixture.generator": GENERATOR_NAME,
    }

    if run_id:
        fixture_filter["_fixture.runId"] = run_id

    return fixture_filter


def seed_database(
    *,
    documents: Sequence[Dict[str, Any]],
    database_name: str,
    collection_name: str,
) -> int:
    """Persist generated fixtures using one bounded MongoDB insert."""
    collection = get_collection(
        database_name=database_name,
        collection_name=collection_name,
    )

    try:
        result = collection.insert_many(
            list(documents),
            ordered=True,
        )
        return len(result.inserted_ids)
    except PyMongoError as exc:
        raise SeedGeneratorError(
            f"Failed to insert fixtures into "
            f"{database_name}.{collection_name}."
        ) from exc


def cleanup_fixtures(
    *,
    database_name: str,
    collection_name: str,
    run_id: Optional[str],
    dry_run: bool,
) -> int:
    """
    Delete only C13-generated records.

    Real project records cannot match the generator filter unless they were
    deliberately marked as C13 fixtures.
    """
    collection = get_collection(
        database_name=database_name,
        collection_name=collection_name,
    )

    fixture_filter = build_fixture_filter(run_id=run_id)

    try:
        if dry_run:
            return collection.count_documents(fixture_filter)

        result = collection.delete_many(fixture_filter)
        return result.deleted_count

    except PyMongoError as exc:
        raise SeedGeneratorError(
            f"Fixture cleanup failed for "
            f"{database_name}.{collection_name}."
        ) from exc


def serialisable_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datetimes so fixtures can be printed as readable JSON."""
    converted: Dict[str, Any] = {}

    for key, value in document.items():
        if isinstance(value, datetime):
            converted[key] = value.isoformat()
        elif isinstance(value, dict):
            converted[key] = serialisable_document(value)
        else:
            converted[key] = value

    return converted


def print_generation_summary(
    *,
    fixtures: Sequence[Dict[str, Any]],
    database_name: str,
    collection_name: str,
    run_id: str,
    seed: int,
    dry_run: bool,
) -> None:
    """Print concise, reproducible evidence for the C13 run."""
    species_counts = Counter(
        fixture["species"] for fixture in fixtures
    )

    sensor_counts = Counter(
        fixture["sensorId"] for fixture in fixtures
    )

    summary = {
        "mode": "dry-run" if dry_run else "insert",
        "generator": GENERATOR_NAME,
        "runId": run_id,
        "seed": seed,
        "database": database_name,
        "collection": collection_name,
        "count": len(fixtures),
        "speciesCounts": dict(species_counts),
        "sensorCounts": dict(sensor_counts),
    }

    print("\nC13 fixture generation summary")
    print(json.dumps(summary, indent=2))

    if fixtures:
        print("\nSample fixture")
        print(
            json.dumps(
                serialisable_document(fixtures[0]),
                indent=2,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.seed_detections",
        description=(
            "Generate deterministic, schema-valid Project Echo "
            "detection fixtures."
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of fixtures to generate (default: 20).",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "Generate records across the previous N days. "
            f"Default when no explicit range is supplied: {DEFAULT_DAYS}."
        ),
    )

    parser.add_argument(
        "--start",
        type=parse_datetime,
        default=None,
        help="Explicit ISO-8601 fixture range start.",
    )

    parser.add_argument(
        "--end",
        type=parse_datetime,
        default=None,
        help="Explicit ISO-8601 fixture range end.",
    )

    parser.add_argument(
        "--species",
        action="append",
        dest="species_values",
        help=(
            "Species to generate. May be supplied multiple times. "
            "Defaults to a small deterministic species set."
        ),
    )

    parser.add_argument(
        "--sensor-id",
        action="append",
        dest="sensor_ids",
        help=(
            "Sensor ID to generate. May be supplied multiple times. "
            "Defaults to sensors 1-4."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Random seed controlling generated fixture values "
            f"(default: {DEFAULT_SEED})."
        ),
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional fixture run identifier. "
            "Useful for deterministic test runs and targeted cleanup."
        ),
    )

    parser.add_argument(
        "--collection",
        choices=ALLOWED_COLLECTIONS,
        default=DEFAULT_COLLECTION,
        help=(
            "MongoDB collection to seed. "
            "Use events for current B1/B2 flows or detections "
            "for the REST /detections flow."
        ),
    )

    parser.add_argument(
        "--database",
        default=None,
        help=(
            f"Target MongoDB database. "
            f"Normal seeding defaults to {DEFAULT_DATABASE}. "
            "Must be explicitly supplied for cleanup."
        ),
    )

    parser.add_argument(
        "--environment",
        default="local",
        help=(
            "Fixture environment label. "
            "Must be explicitly supplied for cleanup."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate and validate fixtures without inserting them. "
            "For cleanup, report how many fixtures would be removed."
        ),
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete previously generated C13 fixtures.",
    )

    parser.add_argument(
        "--confirm-cleanup",
        action="store_true",
        help=(
            "Required confirmation flag for destructive fixture cleanup."
        ),
    )

    return parser


def validate_cleanup_safety(
    *,
    explicit_database: Optional[str],
    environment: str,
    environment_was_explicit: bool,
    confirmed: bool,
) -> None:
    """Enforce C13 destructive-operation safety requirements."""
    if not explicit_database:
        raise SeedGeneratorError(
            "Cleanup requires an explicit --database target."
        )

    if not environment_was_explicit:
        raise SeedGeneratorError(
            "Cleanup requires an explicit --environment target."
        )

    if environment.lower() in {
        "production",
        "prod",
    }:
        raise SeedGeneratorError(
            "C13 fixture cleanup is disabled for production environments."
        )

    if not confirmed:
        raise SeedGeneratorError(
            "Cleanup requires --confirm-cleanup."
        )


def environment_argument_was_supplied(argv: Sequence[str]) -> bool:
    """Detect whether --environment was explicitly provided."""
    return any(
        item == "--environment"
        or item.startswith("--environment=")
        for item in argv
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()

    effective_argv = list(
        sys.argv[1:] if argv is None else argv
    )

    args = parser.parse_args(effective_argv)

    database_name = args.database or DEFAULT_DATABASE

    if args.cleanup:
        try:
            validate_cleanup_safety(
                explicit_database=args.database,
                environment=args.environment,
                environment_was_explicit=(
                    environment_argument_was_supplied(
                        effective_argv
                    )
                ),
                confirmed=args.confirm_cleanup,
            )

            affected = cleanup_fixtures(
                database_name=database_name,
                collection_name=args.collection,
                run_id=args.run_id,
                dry_run=args.dry_run,
            )

            action = (
                "would be removed"
                if args.dry_run
                else "removed"
            )

            scope = (
                f"runId={args.run_id}"
                if args.run_id
                else "all C13 fixture runs"
            )

            print(
                f"C13 cleanup: {affected} fixture(s) {action} "
                f"from {database_name}.{args.collection} "
                f"({scope})."
            )

            return 0

        except SeedGeneratorError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        start, end = resolve_date_range(
            start=args.start,
            end=args.end,
            days=args.days,
        )

        species_values = (
            tuple(args.species_values)
            if args.species_values
            else DEFAULT_SPECIES
        )

        sensor_ids = (
            tuple(args.sensor_ids)
            if args.sensor_ids
            else DEFAULT_SENSOR_IDS
        )

        run_id = args.run_id or str(uuid.uuid4())

        fixtures = generate_fixtures(
            count=args.count,
            start=start,
            end=end,
            species_values=species_values,
            sensor_ids=sensor_ids,
            seed=args.seed,
            run_id=run_id,
            environment=args.environment,
        )

        print_generation_summary(
            fixtures=fixtures,
            database_name=database_name,
            collection_name=args.collection,
            run_id=run_id,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            print(
                "\nDry run complete: no fixture records were inserted."
            )
            return 0

        inserted = seed_database(
            documents=fixtures,
            database_name=database_name,
            collection_name=args.collection,
        )

        print(
            f"\nInserted {inserted} C13 fixture(s) into "
            f"{database_name}.{args.collection}."
        )
        print(
            f"Fixture run ID: {run_id}"
        )

        return 0

    except SeedGeneratorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    except Exception as exc:
        # Keep CLI output concise; do not print connection strings,
        # credentials or internal stack data.
        print(
            f"ERROR: Unexpected fixture generation failure: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())