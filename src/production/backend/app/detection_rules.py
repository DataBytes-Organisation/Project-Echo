"""Validation rules shared by detection service operations."""

from app.exceptions import DetectionRuleError


def validate_list_filters(start_time=None, end_time=None, lat=None, lon=None, radius_km=None):
    """Validate cross-field rules that FastAPI cannot express independently."""
    if start_time and end_time and start_time > end_time:
        raise DetectionRuleError("start_time must be earlier than or equal to end_time.")

    location_values = (lat, lon, radius_km)
    if any(value is not None for value in location_values) and not all(
        value is not None for value in location_values
    ):
        raise DetectionRuleError(
            "lat, lon and radius_km must be provided together."
        )
    if lat is not None and not -90 <= lat <= 90:
        raise DetectionRuleError("lat must be between -90 and 90.")
    if lon is not None and not -180 <= lon <= 180:
        raise DetectionRuleError("lon must be between -180 and 180.")
    if radius_km is not None and radius_km <= 0:
        raise DetectionRuleError("radius_km must be greater than zero.")


def mutable_detection_update(update_data):
    """Return allowed update fields without mutating the caller's payload."""
    safe_update = dict(update_data)
    safe_update.pop("_id", None)
    safe_update.pop("id", None)
    if not safe_update:
        raise DetectionRuleError("No mutable fields were provided for update.")
    return safe_update
