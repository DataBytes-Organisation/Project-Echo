"""
Tests for app.detection_rules.

Run inside the API container / with the backend's requirements installed:
    python -m unittest test_detection_rules -v
"""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

REQUIRED_ENV = {
    "MONGODB_URI": "mongodb://user:pass@localhost:27017/EchoNet",
    "USER_MONGODB_URI": "mongodb://user:pass@localhost:27017/UserSample",
    "JWT_SECRET": "test-secret",
}
os.environ.setdefault("MONGODB_URI", REQUIRED_ENV["MONGODB_URI"])
os.environ.setdefault("USER_MONGODB_URI", REQUIRED_ENV["USER_MONGODB_URI"])
os.environ.setdefault("JWT_SECRET", REQUIRED_ENV["JWT_SECRET"])

from pydantic import ValidationError

from app.config import Settings
from app.detection_rules import DetectionRules, evaluate_detection, get_active_rules


def _detection(confidence=90.0, species="Koala", sensorId="1"):
    return SimpleNamespace(confidence=confidence, species=species, sensorId=sensorId)


class TestEvaluateDetection(unittest.TestCase):
    def test_accepted_when_no_restrictions(self):
        rules = DetectionRules()
        accepted, reason = evaluate_detection(_detection(), rules)
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

    def test_rejected_below_min_confidence(self):
        rules = DetectionRules(min_confidence=80)
        accepted, reason = evaluate_detection(_detection(confidence=50.0), rules)
        self.assertFalse(accepted)
        self.assertEqual(reason, "below_min_confidence")

    def test_accepted_at_exactly_min_confidence(self):
        rules = DetectionRules(min_confidence=80)
        accepted, reason = evaluate_detection(_detection(confidence=80.0), rules)
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

    def test_rejected_species_not_allowed(self):
        rules = DetectionRules(allowed_species=["Sus Scrofa"])
        accepted, reason = evaluate_detection(_detection(species="Koala"), rules)
        self.assertFalse(accepted)
        self.assertEqual(reason, "species_not_allowed")

    def test_accepted_species_in_allow_list(self):
        rules = DetectionRules(allowed_species=["Koala", "Sus Scrofa"])
        accepted, reason = evaluate_detection(_detection(species="Koala"), rules)
        self.assertTrue(accepted)

    def test_rejected_sensor_not_allowed(self):
        rules = DetectionRules(allowed_sensor_ids=["2"])
        accepted, reason = evaluate_detection(_detection(sensorId="1"), rules)
        self.assertFalse(accepted)
        self.assertEqual(reason, "sensor_not_allowed")

    def test_empty_allow_lists_mean_no_restriction(self):
        rules = DetectionRules(allowed_species=[], allowed_sensor_ids=[])
        accepted, _ = evaluate_detection(_detection(species="anything", sensorId="anything"), rules)
        self.assertTrue(accepted)

    def test_rules_checked_in_order_confidence_first(self):
        # A detection that fails multiple rules should report the first one checked.
        rules = DetectionRules(min_confidence=80, allowed_species=["Sus Scrofa"])
        accepted, reason = evaluate_detection(_detection(confidence=10.0, species="Koala"), rules)
        self.assertFalse(accepted)
        self.assertEqual(reason, "below_min_confidence")


class TestDetectionRulesValidation(unittest.TestCase):
    def test_min_confidence_over_100_rejected(self):
        with self.assertRaises(ValidationError):
            DetectionRules(min_confidence=150)

    def test_min_confidence_negative_rejected(self):
        with self.assertRaises(ValidationError):
            DetectionRules(min_confidence=-1)

    def test_defaults_impose_no_restriction(self):
        rules = DetectionRules()
        self.assertEqual(rules.min_confidence, 0)
        self.assertEqual(rules.allowed_species, [])
        self.assertEqual(rules.allowed_sensor_ids, [])


class TestGetActiveRulesFromSettings(unittest.TestCase):
    def test_reads_configured_thresholds(self):
        env = {
            **REQUIRED_ENV,
            "DETECTION_MIN_CONFIDENCE": "75",
            "DETECTION_ALLOWED_SPECIES": "Koala,Sus Scrofa",
            "DETECTION_ALLOWED_SENSOR_IDS": "1, 2",
        }
        with patch.dict("os.environ", env, clear=True):
            settings = Settings()
        self.assertEqual(settings.detection_min_confidence, 75)
        self.assertEqual(settings.detection_allowed_species, ["Koala", "Sus Scrofa"])
        self.assertEqual(settings.detection_allowed_sensor_ids, ["1", "2"])

    def test_get_active_rules_uses_module_level_settings(self):
        with patch("app.detection_rules.settings") as mock_settings:
            mock_settings.detection_min_confidence = 60
            mock_settings.detection_allowed_species = ["Koala"]
            mock_settings.detection_allowed_sensor_ids = []
            rules = get_active_rules()
        self.assertEqual(rules.min_confidence, 60)
        self.assertEqual(rules.allowed_species, ["Koala"])


if __name__ == "__main__":
    unittest.main()
