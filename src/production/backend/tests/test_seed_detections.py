from datetime import datetime, timezone

import pytest

from app.schemas import DetectionCreate
from tools import seed_detections as seed


def make_range():
    return (
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 8, tzinfo=timezone.utc),
    )


def generate(count=8, seed_value=374):
    start, end = make_range()

    return seed.generate_fixtures(
        count=count,
        start=start,
        end=end,
        species_values=("Sus Scrofa", "Dingo"),
        sensor_ids=("1", "2"),
        seed=seed_value,
        run_id="test-run",
        environment="test",
    )


def detection_only(document):
    """Remove C13 metadata before deterministic comparisons."""
    result = dict(document)
    result.pop("_fixture", None)
    return result


def test_requested_fixture_count_is_generated():
    fixtures = generate(count=8)

    assert len(fixtures) == 8


def test_all_generated_fixtures_match_detection_create_schema():
    fixtures = generate()

    for fixture in fixtures:
        detection = detection_only(fixture)

        # Should not raise a Pydantic validation error.
        validated = DetectionCreate(**detection)

        assert validated.sensorId
        assert validated.species
        assert len(validated.microphoneLLA) == 3
        assert len(validated.animalEstLLA) == 3
        assert len(validated.animalTrueLLA) == 3
        assert 0 <= validated.confidence <= 100


def test_same_seed_produces_same_detection_data():
    first = generate(seed_value=374)
    second = generate(seed_value=374)

    first_data = [detection_only(item) for item in first]
    second_data = [detection_only(item) for item in second]

    assert first_data == second_data


def test_different_seed_changes_generated_data():
    first = generate(seed_value=374)
    second = generate(seed_value=999)

    first_data = [detection_only(item) for item in first]
    second_data = [detection_only(item) for item in second]

    assert first_data != second_data


def test_species_and_sensor_distribution_is_predictable():
    fixtures = generate(count=8)

    species = [fixture["species"] for fixture in fixtures]
    sensors = [fixture["sensorId"] for fixture in fixtures]

    assert species.count("Sus Scrofa") == 4
    assert species.count("Dingo") == 4

    assert sensors.count("1") == 4
    assert sensors.count("2") == 4


def test_generated_timestamps_cover_requested_range():
    fixtures = generate(count=8)
    start, end = make_range()

    timestamps = [fixture["timestamp"] for fixture in fixtures]

    assert timestamps[0] == start
    assert timestamps[-1] == end

    for timestamp in timestamps:
        assert start <= timestamp <= end


def test_fixture_metadata_marks_generated_records():
    fixtures = generate(count=1)

    metadata = fixtures[0]["_fixture"]

    assert metadata["generated"] is True
    assert metadata["generator"] == seed.GENERATOR_NAME
    assert metadata["runId"] == "test-run"
    assert metadata["seed"] == 374
    assert metadata["environment"] == "test"


def test_fixture_cleanup_filter_cannot_target_normal_records():
    fixture_filter = seed.build_fixture_filter()

    assert fixture_filter == {
        "_fixture.generated": True,
        "_fixture.generator": seed.GENERATOR_NAME,
    }


def test_targeted_cleanup_filter_contains_run_id():
    fixture_filter = seed.build_fixture_filter(
        run_id="specific-run"
    )

    assert fixture_filter["_fixture.generated"] is True
    assert fixture_filter["_fixture.generator"] == seed.GENERATOR_NAME
    assert fixture_filter["_fixture.runId"] == "specific-run"


def test_invalid_date_range_is_rejected():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    end = datetime(2026, 9, 1, tzinfo=timezone.utc)

    with pytest.raises(seed.SeedGeneratorError):
        seed.resolve_date_range(
            start=start,
            end=end,
            days=None,
        )


def test_cleanup_requires_database():
    with pytest.raises(
        seed.SeedGeneratorError,
        match="explicit --database",
    ):
        seed.validate_cleanup_safety(
            explicit_database=None,
            environment="local",
            environment_was_explicit=True,
            confirmed=True,
        )


def test_cleanup_requires_explicit_environment():
    with pytest.raises(
        seed.SeedGeneratorError,
        match="explicit --environment",
    ):
        seed.validate_cleanup_safety(
            explicit_database="EchoNet",
            environment="local",
            environment_was_explicit=False,
            confirmed=True,
        )


def test_cleanup_requires_confirmation():
    with pytest.raises(
        seed.SeedGeneratorError,
        match="--confirm-cleanup",
    ):
        seed.validate_cleanup_safety(
            explicit_database="EchoNet",
            environment="local",
            environment_was_explicit=True,
            confirmed=False,
        )


def test_cleanup_is_blocked_for_production():
    with pytest.raises(
        seed.SeedGeneratorError,
        match="disabled for production",
    ):
        seed.validate_cleanup_safety(
            explicit_database="EchoNet",
            environment="production",
            environment_was_explicit=True,
            confirmed=True,
        )


def test_dry_run_never_attempts_database_insert(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Database insertion must not occur during dry-run."
        )

    monkeypatch.setattr(
        seed,
        "seed_database",
        fail_if_called,
    )

    result = seed.main(
        [
            "--count",
            "3",
            "--start",
            "2026-09-01",
            "--end",
            "2026-09-03",
            "--run-id",
            "dry-run-test",
            "--dry-run",
        ]
    )

    assert result == 0