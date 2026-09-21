from datetime import datetime, timezone

import pytest
from pymongo.errors import PyMongoError

from app.database import client as project_mongo_client
from app.routers import insights
from tools import seed_detections as seed


FIXTURE_RUN_ID = "b2-4-analytics-fixtures"
FIXTURE_SEED = 374

TEST_EVENTS_COLLECTION = "_b2_4_test_events"
TEST_MICROPHONES_COLLECTION = "_b2_4_test_microphones"
TEST_NODES_COLLECTION = "_b2_4_test_nodes"

EXPECTED_SPECIES_COUNTS = {
    "Sus Scrofa": 4,
    "Dingo": 4,
    "Crimson Rosella": 4,
}

EXPECTED_SENSOR_COUNTS = {
    "2": 6,
    "3": 6,
}


class InsightsTestDatabase:
    """
    Minimal database proxy used only by B2.4 tests.

    insights.py requests collections using db["events"],
    db["microphones"] and db["nodes"].

    This proxy redirects those names to dedicated test collections
    inside the existing EchoNet database, so production/local data
    is never read, overwritten or deleted.
    """

    COLLECTION_MAP = {
        "events": TEST_EVENTS_COLLECTION,
        "microphones": TEST_MICROPHONES_COLLECTION,
        "nodes": TEST_NODES_COLLECTION,
    }

    def __init__(self, real_database):
        self.real_database = real_database

    def __getitem__(self, collection_name):
        mapped_name = self.COLLECTION_MAP.get(
            collection_name,
            collection_name,
        )
        return self.real_database[mapped_name]


def build_b2_fixtures():
    """
    Build the deterministic C13 dataset used by B2.4.

    Expected totals:
      12 detections
      3 unique species
      2 sensors

      Species:
        Sus Scrofa        -> 4
        Dingo             -> 4
        Crimson Rosella   -> 4

      Sensors:
        2 -> 6
        3 -> 6
    """
    return seed.generate_fixtures(
        count=12,
        start=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2026,
            9,
            12,
            tzinfo=timezone.utc,
        ),
        species_values=(
            "Sus Scrofa",
            "Dingo",
            "Crimson Rosella",
        ),
        sensor_ids=(
            "2",
            "3",
        ),
        seed=FIXTURE_SEED,
        run_id=FIXTURE_RUN_ID,
        environment="test",
        collection_name="events",
    )


def cleanup_test_collections(real_database):
    """
    Remove only B2.4 test data.

    Normal EchoNet collections are never targeted.
    """
    for collection_name in (
        TEST_EVENTS_COLLECTION,
        TEST_MICROPHONES_COLLECTION,
        TEST_NODES_COLLECTION,
    ):
        real_database[collection_name].delete_many({})


@pytest.fixture
def isolated_insights_database(monkeypatch):
    """
    Redirect the collection objects imported by insights.py
    to dedicated B2.4 collections inside EchoNet.

    The current insights module imports:
        Events
        Microphones
        Nodes

    directly from app.database, so those module globals are
    patched rather than patching an old `db` object.
    """
    try:
        project_mongo_client.admin.command("ping")
    except PyMongoError:
        pytest.skip(
            "MongoDB is not available for B2.4 integration tests."
        )

    real_database = project_mongo_client["EchoNet"]

    cleanup_test_collections(real_database)

    test_collections = {
        "events": real_database[TEST_EVENTS_COLLECTION],
        "microphones": real_database[TEST_MICROPHONES_COLLECTION],
        "nodes": real_database[TEST_NODES_COLLECTION],
    }

    monkeypatch.setattr(
        insights,
        "Events",
        test_collections["events"],
    )

    monkeypatch.setattr(
        insights,
        "Microphones",
        test_collections["microphones"],
    )

    monkeypatch.setattr(
        insights,
        "Nodes",
        test_collections["nodes"],
    )

    try:
        yield test_collections
    finally:
        cleanup_test_collections(real_database)


@pytest.fixture
def populated_insights_database(
    isolated_insights_database,
):
    fixtures = build_b2_fixtures()

    result = isolated_insights_database[
        "events"
    ].insert_many(fixtures)

    assert len(result.inserted_ids) == 12

    return isolated_insights_database


def test_c13_fixtures_have_expected_species_distribution():
    fixtures = build_b2_fixtures()

    actual = {}

    for fixture in fixtures:
        species = fixture["species"]
        actual[species] = (
            actual.get(species, 0) + 1
        )

    assert actual == EXPECTED_SPECIES_COUNTS


def test_c13_fixtures_have_expected_sensor_distribution():
    fixtures = build_b2_fixtures()

    actual = {}

    for fixture in fixtures:
        sensor_id = fixture["sensorId"]
        actual[sensor_id] = (
            actual.get(sensor_id, 0) + 1
        )

    assert actual == EXPECTED_SENSOR_COUNTS


def test_b2_fixture_generation_is_repeatable():
    first = build_b2_fixtures()
    second = build_b2_fixtures()

    def analytics_fields(fixtures):
        return [
            {
                "timestamp": item["timestamp"],
                "sensorId": item["sensorId"],
                "species": item["species"],
                "confidence": item["confidence"],
                "sampleRate": item["sampleRate"],
            }
            for item in fixtures
        ]

    assert analytics_fields(
        first
    ) == analytics_fields(
        second
    )


def test_insights_overview_returns_expected_fixture_totals(
    populated_insights_database,
):
    result = insights.insights_overview()

    assert result["counts"]["detections"] == 12
    assert result["counts"]["uniqueSpecies"] == 3
    assert (
        result["counts"]["sensorsWithDetections"]
        == 2
    )


def test_insights_overview_reports_fixture_time_range(
    populated_insights_database,
):
    result = insights.insights_overview()

    assert result["timeRange"]["start"].startswith(
        "2026-09-01"
    )

    assert result["timeRange"]["end"].startswith(
        "2026-09-12"
    )


def test_insights_species_returns_expected_aggregate_totals(
    populated_insights_database,
):
    result = insights.insights_species(
        limit=10
    )

    actual = {
        item["species"]: item["count"]
        for item in result["items"]
    }

    assert actual == EXPECTED_SPECIES_COUNTS


def test_insights_species_total_matches_overview_detection_total(
    populated_insights_database,
):
    overview = insights.insights_overview()

    species_result = insights.insights_species(
        limit=10
    )

    species_total = sum(
        item["count"]
        for item in species_result["items"]
    )

    assert species_total == 12
    assert (
        species_total
        == overview["counts"]["detections"]
    )

def test_overview_species_filter_returns_expected_totals(
    populated_insights_database,
):
    result = insights.insights_overview(
        species="Sus Scrofa"
    )

    assert result["counts"]["detections"] == 4
    assert result["counts"]["uniqueSpecies"] == 1
    assert result["counts"]["sensorsWithDetections"] == 2


def test_overview_sensor_filter_returns_expected_totals(
    populated_insights_database,
):
    result = insights.insights_overview(
        sensorId="2"
    )

    assert result["counts"]["detections"] == 6
    assert result["counts"]["uniqueSpecies"] == 3
    assert result["counts"]["sensorsWithDetections"] == 1


def test_overview_date_filter_returns_expected_totals(
    populated_insights_database,
):
    result = insights.insights_overview(
        start="2026-09-03T00:00:00Z",
        end="2026-09-06T00:00:00Z",
    )

    assert result["counts"]["detections"] == 4
    assert result["counts"]["uniqueSpecies"] == 3
    assert result["counts"]["sensorsWithDetections"] == 2

    assert result["timeRange"]["start"].startswith(
        "2026-09-03"
    )
    assert result["timeRange"]["end"].startswith(
        "2026-09-06"
    )


def test_species_endpoint_combined_filters_return_expected_total(
    populated_insights_database,
):
    result = insights.insights_species(
        species="Dingo",
        sensorId="3",
        limit=10,
    )

    assert len(result["items"]) == 1
    assert result["items"][0]["species"] == "Dingo"
    assert result["items"][0]["count"] == 2