"""Tests for the Sensor Health router.

Covers the fallback-heavy helpers and the GET /{sensor_id} detail endpoint
without talking to a real MongoDB. Run from src/production/backend:

    python -m unittest app.routers.test_sensors -v
"""

from __future__ import annotations

import datetime
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# sensors.py imports Mongo collections at module load. Stub them first so the
# tests never open a real database connection.
_db = MagicMock()
_db.Events = MagicMock()
_db.Nodes = MagicMock()
_db.SensorReboots = MagicMock()
_db.SensorSettings = MagicMock()
sys.modules.setdefault("app.database", _db)

from fastapi import HTTPException  # noqa: E402

from app.routers import sensors  # noqa: E402


NOW = datetime.datetime(2026, 8, 31, 5, 0, tzinfo=datetime.timezone.utc)


def _iso_minutes_ago(minutes: int) -> str:
    return (NOW - datetime.timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def _battery_node(**overrides):
    node = {
        "_id": "node_1_2",
        "name": "Alpha Sub 2",
        "type": "raspberry_pi",
        "model": "RaspberryPi Zero",
        "customProperties": {"processorSpeed": "1GHz", "memory": "512MB"},
        "connectedNodes": ["node_1"],
        "location": {"latitude": -38.7789, "longitude": 143.5705},
        "components": [
            {
                "id": "comp_bat",
                "type": "battery",
                "category": "power",
                "model": "LithiumPro 2000",
                "customProperties": {
                    "capacity": "2000mAh",
                    "health": 0.92,
                    "currentCharge": 0.75,
                },
            }
        ],
    }
    node.update(overrides)
    return node


class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, *_args, **_kwargs):
        return self

    def limit(self, _n):
        return self

    def __iter__(self):
        return iter(self._docs)


class TestAsPctAndFloat(unittest.TestCase):
    def test_fraction_scales_to_percent(self):
        self.assertEqual(sensors._as_pct(0.75), 75.0)

    def test_already_scaled_percent_is_kept(self):
        self.assertEqual(sensors._as_pct(75), 75.0)

    def test_rejects_missing_negative_and_over_100(self):
        self.assertIsNone(sensors._as_pct(None))
        self.assertIsNone(sensors._as_pct(-1))
        self.assertIsNone(sensors._as_pct(140))
        self.assertIsNone(sensors._as_pct(True))


class TestComponentPropsMerge(unittest.TestCase):
    def test_live_sensor_data_overrides_seeded_custom_properties(self):
        component = {
            "customProperties": {"currentCharge": 0.4, "health": 0.9},
            "sensorData": {"currentCharge": 0.81},
        }
        props = sensors._component_props(component)
        self.assertEqual(props["currentCharge"], 0.81)
        self.assertEqual(props["health"], 0.9)

    def test_seeded_custom_properties_used_when_no_live_telemetry(self):
        component = {"customProperties": {"currentCharge": 0.65}}
        self.assertEqual(sensors._component_props(component)["currentCharge"], 0.65)

    def test_battery_pct_reads_current_charge_from_custom_properties(self):
        self.assertEqual(sensors._extract_battery_pct(_battery_node()), 75.0)

    def test_battery_pct_none_when_device_has_no_battery(self):
        node = {
            "_id": "node_1",
            "components": [
                {
                    "type": "solar_panel",
                    "customProperties": {"wattage": 360, "currentOutput": 320},
                }
            ],
        }
        self.assertIsNone(sensors._extract_battery_pct(node))


class TestGpsFallbackChain(unittest.TestCase):
    def test_location_dict_wins(self):
        node = {
            "location": {"latitude": -38.1, "longitude": 143.5},
            "customProperties": {"lat": 0, "lon": 0},
            "microphoneLLA": [-10.0, 10.0],
        }
        self.assertEqual(sensors._extract_gps(node), {"lat": -38.1, "lon": 143.5})

    def test_falls_back_to_custom_properties(self):
        node = {"customProperties": {"lat": -38.2, "lng": 143.6}}
        self.assertEqual(sensors._extract_gps(node), {"lat": -38.2, "lon": 143.6})

    def test_falls_back_to_microphone_lla(self):
        node = {"microphoneLLA": [-38.3, 143.7, 10]}
        self.assertEqual(sensors._extract_gps(node), {"lat": -38.3, "lon": 143.7})

    def test_falls_back_to_component_gps(self):
        node = {
            "components": [
                {"sensorData": {"gps": {"lat": -38.4, "longitude": 143.8}}}
            ]
        }
        self.assertEqual(sensors._extract_gps(node), {"lat": -38.4, "lon": 143.8})

    def test_returns_none_when_no_coordinates_exist(self):
        self.assertIsNone(sensors._extract_gps({"_id": "bare"}))


class TestMetricHeuristics(unittest.TestCase):
    def test_efficiency_and_charge_render_as_percent(self):
        self.assertEqual(sensors._format_metric("efficiency", 0.85), "85.0%")
        self.assertEqual(sensors._format_metric("currentCharge", 0.75), "75.0%")

    def test_wattage_keeps_unit(self):
        self.assertEqual(sensors._format_metric("wattage", 360), "360 W")

    def test_solar_output_percentage(self):
        node = {
            "components": [
                {
                    "type": "solar_panel",
                    "customProperties": {
                        "wattage": 360,
                        "currentOutput": 320,
                        "efficiency": 0.85,
                    },
                }
            ]
        }
        power = sensors._extract_power(node)
        self.assertEqual(power["solarRatedW"], 360.0)
        self.assertEqual(power["solarOutputW"], 320.0)
        self.assertEqual(power["solarOutputPct"], 88.9)
        self.assertEqual(power["solarEfficiencyPct"], 85.0)

    def test_temperature_from_fan_component(self):
        node = {
            "components": [
                {"type": "fan", "customProperties": {"temperature": 35}}
            ]
        }
        self.assertEqual(sensors._extract_temperature_c(node), 35.0)


class TestHealthStatus(unittest.TestCase):
    def setUp(self):
        sensors.SensorSettings.find_one.return_value = None

    def test_missing_last_seen_is_unknown_not_offline(self):
        payload = sensors._sensor_health_payload(_battery_node(), None, NOW)
        self.assertEqual(payload["status"], "Unknown")
        self.assertEqual(payload["batteryPct"], 75.0)
        self.assertIsNone(payload["lastSeenMinutesAgo"])

    def test_recent_heartbeat_is_online(self):
        node = _battery_node(lastSeen=_iso_minutes_ago(2))
        payload = sensors._sensor_health_payload(node, None, NOW)
        self.assertEqual(payload["status"], "Online")

    def test_stale_heartbeat_is_offline(self):
        node = _battery_node(lastSeen=_iso_minutes_ago(40), components=[])
        payload = sensors._sensor_health_payload(node, None, NOW)
        self.assertEqual(payload["status"], "Offline")

    def test_low_battery_overrides_online_but_not_unknown(self):
        low_battery = [
            {
                "type": "battery",
                "customProperties": {"currentCharge": 0.1},
            }
        ]
        online = _battery_node(lastSeen=_iso_minutes_ago(1), components=low_battery)
        self.assertEqual(sensors._sensor_health_payload(online, None, NOW)["status"], "Low Battery")

        never_seen = _battery_node(components=low_battery)
        self.assertEqual(sensors._sensor_health_payload(never_seen, None, NOW)["status"], "Unknown")

    def test_invalid_node_id_returns_none(self):
        self.assertIsNone(sensors._sensor_health_payload({"_id": 12}, None, NOW))


class TestAlertDerivation(unittest.TestCase):
    def test_unknown_and_online_are_not_alerts(self):
        recent = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        sensors.Nodes.find.return_value.limit.return_value = [
            _battery_node(),
            _battery_node(_id="online", lastSeen=recent, components=[]),
        ]
        sensors.Events.aggregate.return_value = []
        sensors.SensorSettings.find_one.return_value = None

        result = sensors.get_sensor_alerts()
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["items"], [])

    def test_offline_becomes_critical_alert(self):
        sensors.Nodes.find.return_value.limit.return_value = [
            _battery_node(_id="gone", lastSeen=_iso_minutes_ago(40), components=[]),
        ]
        sensors.Events.aggregate.return_value = []
        sensors.SensorSettings.find_one.return_value = None

        result = sensors.get_sensor_alerts()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["severity"], "Critical")
        self.assertEqual(result["items"][0]["issue"], "Offline")


class TestSensorDetailEndpoint(unittest.TestCase):
    def setUp(self):
        sensors.SensorSettings.find_one.return_value = None
        sensors.Events.aggregate.return_value = []
        sensors.Events.find.return_value = FakeCursor([])
        sensors.Nodes.find.return_value = FakeCursor(
            [{"_id": "node_1", "name": "Node Alpha", "type": "master", "model": "RaspberryPi4"}]
        )

    def test_missing_sensor_raises_404(self):
        sensors.Nodes.find_one.return_value = None
        with self.assertRaises(HTTPException) as ctx:
            sensors.get_sensor_detail("does-not-exist")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("does-not-exist", ctx.exception.detail)

    def test_detail_includes_health_location_components_and_audio(self):
        last_audio_ts = NOW - datetime.timedelta(minutes=12)
        sensors.Nodes.find_one.return_value = _battery_node()
        sensors.Events.find_one.return_value = {
            "timestamp": last_audio_ts,
            "species": "Colluricincla harmonica",
            "confidence": 81.5,
            "sampleRate": 48000,
        }
        sensors.Events.find.return_value = FakeCursor(
            [
                {
                    "_id": "evt-1",
                    "timestamp": last_audio_ts,
                    "species": "Colluricincla harmonica",
                    "confidence": 81.5,
                    "sampleRate": 48000,
                }
            ]
        )

        payload = sensors.get_sensor_detail("node_1_2")

        self.assertEqual(payload["sensorId"], "node_1_2")
        self.assertEqual(payload["status"], "Unknown")
        self.assertEqual(payload["batteryPct"], 75.0)
        self.assertEqual(payload["gps"], {"lat": -38.7789, "lon": 143.5705})
        self.assertEqual(payload["hardware"]["memory"], "512MB")
        self.assertEqual(payload["lastAudio"]["species"], "Colluricincla harmonica")
        self.assertEqual(len(payload["recentAudio"]), 1)
        self.assertEqual(payload["components"][0]["type"], "battery")
        self.assertEqual(payload["connectedDevices"][0]["sensorId"], "node_1")
        self.assertTrue(payload["connectedDevices"][0]["known"])

    def test_detail_without_audio_still_returns_empty_history(self):
        sensors.Nodes.find_one.return_value = _battery_node()
        sensors.Events.find_one.return_value = None
        sensors.Events.find.return_value = FakeCursor([])

        payload = sensors.get_sensor_detail("node_1_2")
        self.assertIsNone(payload["lastAudio"])
        self.assertEqual(payload["recentAudio"], [])


if __name__ == "__main__":
    unittest.main()
