## app.detection_rules.py
# Configurable evaluation of incoming detections against operator-defined
# rules, applied before persistence/streaming. Rule values come from
# app.config.settings (see C4) rather than being hardcoded in a route, and
# the evaluator itself is transport-agnostic so both HTTP ingestion
# (app.detections.create_detection) and A1's future MQTT ingestion bridge
# can call the same logic and get the same accept/reject decision.
import logging
from typing import List, Tuple

from pydantic import BaseModel, confloat

from app.config import settings

logger = logging.getLogger(__name__)


class DetectionRules(BaseModel):
    """A single rule set. Empty allow-lists mean 'no restriction' for that rule."""

    min_confidence: confloat(ge=0, le=100) = 0
    allowed_species: List[str] = []
    allowed_sensor_ids: List[str] = []


def get_active_rules() -> DetectionRules:
    """Build the current rule set from configuration.

    Re-reads app.config.settings on every call (not cached at import time)
    so config changes take effect without restarting long-lived state, and
    so a per-request rule override (e.g. in tests) can pass its own
    DetectionRules instead of calling this at all.
    """
    return DetectionRules(
        min_confidence=settings.detection_min_confidence,
        allowed_species=settings.detection_allowed_species,
        allowed_sensor_ids=settings.detection_allowed_sensor_ids,
    )


def evaluate_detection(detection, rules: DetectionRules = None) -> Tuple[bool, str]:
    """Evaluate one detection against `rules` (or the active configured rules).

    `detection` needs .confidence (0-100 float), .species (str) and
    .sensorId (str) attributes - satisfied by both DetectionCreate/Detection
    (schemas.EventSchema) and the MQTT-side event payload A1 will parse.

    Returns (accepted, reason). `reason` is a stable machine-readable code
    ("accepted", "below_min_confidence", "species_not_allowed",
    "sensor_not_allowed") suitable for logging and for surfacing to API
    clients - never raises for a rule violation, so callers decide how to
    respond (HTTP 422, MQTT drop-and-log, etc).
    """
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
    """Log a rule-rejected detection. No audioClip/binary data.

    logging_config.JsonFormatter only serialises the formatted message (it
    does not read LogRecord.extra), so the context is embedded in the
    message itself rather than passed via `extra=`.
    """
    logger.info(
        "detection_rejected reason=%s sensorId=%s species=%s confidence=%s",
        reason,
        getattr(detection, "sensorId", None),
        getattr(detection, "species", None),
        getattr(detection, "confidence", None),
    )
