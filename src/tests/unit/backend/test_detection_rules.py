"""Regression tests for cross-field detection rules (A3.2)."""

from datetime import datetime, timedelta

import pytest

from app.detection_rules import mutable_detection_update, validate_list_filters
from app.exceptions import DetectionRuleError


def test_valid_detection_filters_are_accepted():
    start = datetime(2026, 1, 1)
    validate_list_filters(start, start + timedelta(hours=1), -37.8, 144.9, 5)


def test_start_time_cannot_follow_end_time():
    start = datetime(2026, 1, 2)
    with pytest.raises(DetectionRuleError, match="start_time"):
        validate_list_filters(start, start - timedelta(days=1))


@pytest.mark.parametrize(
    "values",
    [
        {"lat": -37.8},
        {"lon": 144.9},
        {"radius_km": 5},
        {"lat": -37.8, "lon": 144.9},
    ],
)
def test_location_filter_must_be_complete(values):
    with pytest.raises(DetectionRuleError, match="provided together"):
        validate_list_filters(**values)


@pytest.mark.parametrize(
    "values,field",
    [
        ({"lat": -91, "lon": 0, "radius_km": 1}, "lat"),
        ({"lat": 0, "lon": 181, "radius_km": 1}, "lon"),
        ({"lat": 0, "lon": 0, "radius_km": 0}, "radius_km"),
    ],
)
def test_location_filter_ranges_are_enforced(values, field):
    with pytest.raises(DetectionRuleError, match=field):
        validate_list_filters(**values)


def test_update_removes_immutable_ids_without_mutating_input():
    supplied = {"_id": "old", "id": "old", "species": "Koala"}
    assert mutable_detection_update(supplied) == {"species": "Koala"}
    assert supplied == {"_id": "old", "id": "old", "species": "Koala"}


def test_update_with_only_immutable_fields_is_rejected():
    with pytest.raises(DetectionRuleError, match="mutable"):
        mutable_detection_update({"_id": "old"})
