"""Validation and configurable evaluation rules for detections."""

import logging
from typing import List, Tuple

from pydantic import BaseModel, confloat

from app.config import settings
from app.exceptions import DetectionRuleError

logger = logging.getLogger(__name__)


class DetectionRules(BaseModel):
    """Configurable ingestion rules; empty allow-lists impose no restriction."""

    min_confidence: confloat(ge=0, le=100) = 0
    allowed_species: List[str] = []
    allowed_sensor_ids: List[str] = []


def get_active_rules() -> DetectionRules:
    """Build the current configured rule set for incoming detections."""
    return DetectionRules(
        min_confidence=settings.detection_min_confidence,
        allowed_species=settings.detection_allowed_species,
        allowed_sensor_ids=settings.detection_allowed_sensor_ids,
    )


def evaluate_detection(detection, rules: DetectionRules = None) -> Tuple[bool, str]:
    """Return whether a detection is accepted and its stable rule outcome."""
    if rules is None:
        rules = get_active_rules()

    if detection.confidence < rules.min_confidence:
        return False, "below_min_confidence"
    if rules.allowed_species and detection.species not in rules.allowed_species:
        return False, "species_not_allowed"
    if rules.allowed_sensor_ids and detection.sensorId not in rules.allowed_sensor_ids:
        return False, "sensor_not_allowed"
    return True, "accepted"


def log_rejected_detection(detection, reason: str) -> None:
    """Log a rejected detection without retaining binary audio data."""
    logger.info(
        "detection_rejected reason=%s sensorId=%s species=%s confidence=%s",
        reason,
        getattr(detection, "sensorId", None),
        getattr(detection, "species", None),
        getattr(detection, "confidence", None),
    )


def validate_list_filters(start_time=None, end_time=None, lat=None, lon=None, radius_km=None):
    """Validate cross-field list-query rules FastAPI cannot express alone."""
    if start_time and end_time and start_time > end_time:
        raise DetectionRuleError("start_time must be earlier than or equal to end_time.")

    location_values = (lat, lon, radius_km)
    if any(value is not None for value in location_values) and not all(
        value is not None for value in location_values
    ):
        raise DetectionRuleError("lat, lon and radius_km must be provided together.")
    if lat is not None and not -90 <= lat <= 90:
        raise DetectionRuleError("lat must be between -90 and 90.")
    if lon is not None and not -180 <= lon <= 180:
        raise DetectionRuleError("lon must be between -180 and 180.")
    if radius_km is not None and radius_km <= 0:
        raise DetectionRuleError("radius_km must be greater than zero.")


def mutable_detection_update(update_data):
    """Return permitted update fields without mutating the caller's payload."""
    safe_update = dict(update_data)
    safe_update.pop("_id", None)
    safe_update.pop("id", None)
    if not safe_update:
        raise DetectionRuleError("No mutable fields were provided for update.")
    return safe_update
