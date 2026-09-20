"""Tests for the Sensor Health router.

Assert what callers actually see from the public endpoints (status text,
battery %, GPS, component readings, alerts), not private helpers.
Run from src/production/backend:

    python -m unittest app.routers.test_sensors -v
"""

from __future__ import annotations

import datetime
import sys
import unittest
from pathlib import Path
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


def _recent_iso(minutes: int = 1) -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
    ).isoformat().replace("+00:00", "Z")


def _stale_iso(minutes: int = 40) -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
    ).isoformat().replace("+00:00", "Z")


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


def _solar_fan_node(**overrides):
    node = {
        "_id": "node_1",
        "name": "Node Alpha",
        "type": "master",
        "model": "RaspberryPi4",
        "customProperties": {
            "processorSpeed": "1.5GHz",
            "memory": "8GB",
            "storage": "64GB",
        },
        "connectedNodes": ["node_1_1", "node_1_2"],
        "location": {"latitude": -38.7789, "longitude": 143.5705},
        "components": [
            {
                "id": "comp_solar",
                "type": "solar_panel",
                "category": "power",
                "model": "SunPower X22-360",
                "customProperties": {
                    "wattage": 360,
                    "efficiency": 0.85,
                    "currentOutput": 320,
                },
            },
            {
                "id": "comp_fan",
                "type": "fan",
                "category": "cooling",
                "model": "CoolMaster Pro",
                "customProperties": {
                    "speed": 1200,
                    "maxSpeed": 2000,
                    "temperature": 35,
                },
            },
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


class SensorEndpointTestCase(unittest.TestCase):
    def setUp(self):
        sensors.SensorSettings.find_one.return_value = None
        sensors.Events.aggregate.return_value = []
        sensors.Events.find.return_value = FakeCursor([])
        sensors.Events.find_one.return_value = None
        sensors.Nodes.find.return_value = FakeCursor([])

    def stub_nodes(self, *nodes):
        # get_sensor_updates does list(Nodes.find(...).limit(...)); FakeCursor.limit
        # returns itself, so iterating the find result yields the full node docs.
        sensors.Nodes.find.return_value = FakeCursor(list(nodes))

    def detail_for(self, node, *, last_audio=None, recent_audio=None, neighbours=None):
        sensors.Nodes.find_one.return_value = node
        sensors.Events.find_one.return_value = last_audio
        sensors.Events.find.return_value = FakeCursor(recent_audio or [])
        if neighbours is not None:
            sensors.Nodes.find.return_value = FakeCursor(neighbours)
        elif node.get("connectedNodes"):
            sensors.Nodes.find.return_value = FakeCursor(
                [
                    {
                        "_id": neighbour_id,
                        "name": neighbour_id,
                        "type": "arduino",
                        "model": "Arduino Uno",
                    }
                    for neighbour_id in node["connectedNodes"]
                ]
            )
        else:
            sensors.Nodes.find.return_value = FakeCursor([])
        return sensors.get_sensor_detail(node["_id"])


class TestShownStatusAndAlerts(SensorEndpointTestCase):
    def test_missing_heartbeat_shows_unknown_not_offline(self):
        payload = self.detail_for(_battery_node())
        self.assertEqual(payload["status"], "Unknown")
        self.assertEqual(payload["batteryPct"], 75.0)
        self.assertIsNone(payload["lastSeenMinutesAgo"])

    def test_recent_heartbeat_shows_online(self):
        payload = self.detail_for(_battery_node(lastSeen=_recent_iso(1)))
        self.assertEqual(payload["status"], "Online")

    def test_stale_heartbeat_shows_offline(self):
        payload = self.detail_for(_battery_node(lastSeen=_stale_iso(40), components=[]))
        self.assertEqual(payload["status"], "Offline")

    def test_low_battery_shows_on_online_device_but_not_unknown(self):
        low_battery = [
            {"type": "battery", "customProperties": {"currentCharge": 0.1}}
        ]
        online = self.detail_for(
            _battery_node(lastSeen=_recent_iso(1), components=low_battery)
        )
        self.assertEqual(online["status"], "Low Battery")
        self.assertEqual(online["batteryPct"], 10.0)

        never_seen = self.detail_for(_battery_node(components=low_battery))
        self.assertEqual(never_seen["status"], "Unknown")

    def test_unknown_and_online_do_not_appear_in_alerts(self):
        self.stub_nodes(
            _battery_node(),
            _battery_node(_id="online", lastSeen=_recent_iso(1), components=[]),
        )
        result = sensors.get_sensor_alerts()
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["items"], [])

    def test_offline_device_raises_critical_alert(self):
        self.stub_nodes(_battery_node(_id="gone", lastSeen=_stale_iso(40), components=[]))
        result = sensors.get_sensor_alerts()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["severity"], "Critical")
        self.assertEqual(result["items"][0]["issue"], "Offline")


class TestShownHealthValues(SensorEndpointTestCase):
    def test_live_sensor_data_wins_over_seeded_custom_properties(self):
        node = _battery_node(
            components=[
                {
                    "id": "comp_bat",
                    "type": "battery",
                    "category": "power",
                    "model": "LithiumPro 2000",
                    "customProperties": {"currentCharge": 0.4, "health": 0.9},
                    "sensorData": {"currentCharge": 0.81},
                }
            ]
        )
        payload = self.detail_for(node)
        self.assertEqual(payload["batteryPct"], 81.0)
        self.assertEqual(payload["power"]["batteryHealthPct"], 90.0)

    def test_seeded_battery_charge_shows_as_percent(self):
        payload = self.detail_for(_battery_node())
        self.assertEqual(payload["batteryPct"], 75.0)
        charge = next(
            metric
            for metric in payload["components"][0]["metrics"]
            if metric["key"] == "currentCharge"
        )
        self.assertEqual(charge["display"], "75.0%")

    def test_device_without_battery_shows_null_battery(self):
        payload = self.detail_for(_solar_fan_node())
        self.assertIsNone(payload["batteryPct"])
        self.assertEqual(payload["temperatureC"], 35.0)
        self.assertEqual(payload["power"]["solarRatedW"], 360.0)
        self.assertEqual(payload["power"]["solarOutputW"], 320.0)
        self.assertEqual(payload["power"]["solarOutputPct"], 88.9)
        self.assertEqual(payload["power"]["solarEfficiencyPct"], 85.0)

    def test_component_readings_show_units_users_can_read(self):
        payload = self.detail_for(_solar_fan_node())
        solar = payload["components"][0]
        fan = payload["components"][1]
        solar_displays = {metric["key"]: metric["display"] for metric in solar["metrics"]}
        fan_displays = {metric["key"]: metric["display"] for metric in fan["metrics"]}

        self.assertEqual(solar_displays["wattage"], "360 W")
        self.assertEqual(solar_displays["efficiency"], "85.0%")
        self.assertEqual(fan_displays["temperature"], "35 °C")
        self.assertEqual(fan_displays["speed"], "1200 RPM")


class TestShownLocation(SensorEndpointTestCase):
    def test_location_field_is_what_the_map_uses(self):
        payload = self.detail_for(_battery_node())
        self.assertEqual(payload["gps"], {"lat": -38.7789, "lon": 143.5705})

    def test_falls_back_to_custom_properties_when_location_missing(self):
        payload = self.detail_for(
            _battery_node(
                location=None,
                customProperties={"lat": -38.2, "lng": 143.6, "memory": "512MB"},
            )
        )
        self.assertEqual(payload["gps"], {"lat": -38.2, "lon": 143.6})

    def test_falls_back_to_microphone_lla(self):
        payload = self.detail_for(
            _battery_node(location=None, customProperties={"memory": "512MB"}, microphoneLLA=[-38.3, 143.7, 10])
        )
        self.assertEqual(payload["gps"], {"lat": -38.3, "lon": 143.7})

    def test_falls_back_to_component_gps(self):
        payload = self.detail_for(
            _battery_node(
                location=None,
                customProperties={"memory": "512MB"},
                components=[
                    {"type": "battery", "sensorData": {"gps": {"lat": -38.4, "longitude": 143.8}}}
                ],
            )
        )
        self.assertEqual(payload["gps"], {"lat": -38.4, "lon": 143.8})

    def test_missing_coordinates_show_as_null(self):
        payload = self.detail_for(
            _battery_node(location=None, customProperties={"memory": "512MB"}, components=[])
        )
        self.assertIsNone(payload["gps"])


class TestSensorDetailEndpoint(SensorEndpointTestCase):
    def test_missing_sensor_raises_404(self):
        sensors.Nodes.find_one.return_value = None
        with self.assertRaises(HTTPException) as ctx:
            sensors.get_sensor_detail("does-not-exist")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("does-not-exist", ctx.exception.detail)

    def test_detail_shows_health_location_components_and_audio(self):
        last_audio_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=12)
        payload = self.detail_for(
            _battery_node(),
            last_audio={
                "timestamp": last_audio_ts,
                "species": "Colluricincla harmonica",
                "confidence": 81.5,
                "sampleRate": 48000,
            },
            recent_audio=[
                {
                    "_id": "evt-1",
                    "timestamp": last_audio_ts,
                    "species": "Colluricincla harmonica",
                    "confidence": 81.5,
                    "sampleRate": 48000,
                }
            ],
            neighbours=[
                {
                    "_id": "node_1",
                    "name": "Node Alpha",
                    "type": "master",
                    "model": "RaspberryPi4",
                }
            ],
        )

        self.assertEqual(payload["sensorId"], "node_1_2")
        self.assertEqual(payload["status"], "Unknown")
        self.assertEqual(payload["batteryPct"], 75.0)
        self.assertEqual(payload["gps"], {"lat": -38.7789, "lon": 143.5705})
        self.assertEqual(payload["hardware"]["memory"], "512MB")
        self.assertEqual(payload["lastAudio"]["species"], "Colluricincla harmonica")
        self.assertEqual(len(payload["recentAudio"]), 1)
        self.assertEqual(payload["components"][0]["type"], "battery")
        self.assertEqual(payload["connectedDevices"][0]["sensorId"], "node_1")
        self.assertEqual(payload["connectedDevices"][0]["name"], "Node Alpha")
        self.assertTrue(payload["connectedDevices"][0]["known"])

    def test_detail_without_audio_still_returns_empty_history(self):
        payload = self.detail_for(_battery_node())
        self.assertIsNone(payload["lastAudio"])
        self.assertEqual(payload["recentAudio"], [])

    def test_updates_list_shows_the_same_status_users_see_on_detail(self):
        self.stub_nodes(_battery_node(), _solar_fan_node(lastSeen=_recent_iso(1)))
        result = sensors.get_sensor_updates()
        by_id = {item["sensorId"]: item for item in result["items"]}

        self.assertEqual(by_id["node_1_2"]["status"], "Unknown")
        self.assertEqual(by_id["node_1_2"]["batteryPct"], 75.0)
        self.assertEqual(by_id["node_1"]["status"], "Online")
        self.assertEqual(by_id["node_1"]["temperatureC"], 35.0)
        self.assertIsNone(by_id["node_1"]["batteryPct"])


if __name__ == "__main__":
    unittest.main()
