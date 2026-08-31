from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from app.database import Events, Nodes, SensorReboots, SensorSettings

router = APIRouter()


DEFAULT_SETTINGS: Dict[str, Any] = {
    "recordIntervalSeconds": 60,
    "sensitivity": "Medium",
    "batteryThresholdPct": 25,
    "onlineWindowMinutes": 5,
    "degradedWindowMinutes": 15,
}


def _parse_iso_datetime(value: Any) -> Optional[datetime.datetime]:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(text)
    except Exception:
        return None


def _minutes_ago(dt: Optional[datetime.datetime], now: datetime.datetime) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=now.tzinfo)
    delta = now - dt
    return max(0, int(delta.total_seconds() // 60))


def _components(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    components = node.get("components")
    if not isinstance(components, list):
        return []
    return [c for c in components if isinstance(c, dict)]


def _component_props(component: Dict[str, Any]) -> Dict[str, Any]:
    """Readings for one component.

    Seeded topology stores values under `customProperties`; live device telemetry
    arrives as `sensorData`. Live values take precedence when both are present.
    """
    props: Dict[str, Any] = {}
    for key in ("customProperties", "sensorData"):
        value = component.get(key)
        if isinstance(value, dict):
            props.update(value)
    return props


def _iter_sensor_data(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [_component_props(component) for component in _components(node)]


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_pct(value: Any) -> Optional[float]:
    """Accepts a 0-1 fraction or an already-scaled 0-100 percentage."""
    number = _as_float(value)
    if number is None or number < 0:
        return None
    if number <= 1:
        number *= 100.0
    if number > 100:
        return None
    return round(number, 1)


def _extract_component_number(
    node: Dict[str, Any],
    keys: List[str],
    *,
    as_percent: bool = False,
) -> Optional[float]:
    candidates: List[float] = []
    for sensor_data in _iter_sensor_data(node):
        for key in keys:
            if key not in sensor_data:
                continue
            value = sensor_data.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except Exception:
                continue

            if as_percent:
                if 0 <= number <= 1:
                    candidates.append(number * 100.0)
                elif 1 < number <= 100:
                    candidates.append(number)
            elif number >= 0:
                candidates.append(number)

    if not candidates:
        return None
    value = min(candidates) if as_percent else candidates[0]
    return round(value, 1)


def _extract_battery_pct(node: Dict[str, Any]) -> Optional[float]:
    for component in _components(node):
        props = _component_props(component)
        if component.get("type") == "battery" or "currentCharge" in props:
            pct = _as_pct(props.get("currentCharge"))
            if pct is not None:
                return pct

    return _extract_component_number(
        node,
        [
            "batteryPct",
            "battery_pct",
            "battery_percent",
            "batteryPercent",
            "battery",
            "batteryLevel",
            "battery_level",
        ],
        as_percent=True,
    )


def _extract_temperature_c(node: Dict[str, Any]) -> Optional[float]:
    for props in _iter_sensor_data(node):
        for key in ("temperature", "temperatureC", "temp"):
            number = _as_float(props.get(key))
            if number is not None:
                return round(number, 1)
    return None


def _extract_power(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Solar generation and battery condition, where the device reports them."""
    power: Dict[str, Any] = {}

    for component in _components(node):
        props = _component_props(component)
        component_type = component.get("type")

        if component_type == "solar_panel":
            rated = _as_float(props.get("wattage"))
            current = _as_float(props.get("currentOutput"))
            if rated is not None:
                power["solarRatedW"] = round(rated, 1)
            if current is not None:
                power["solarOutputW"] = round(current, 1)
            if rated and current is not None and rated > 0:
                power["solarOutputPct"] = round(min(current / rated * 100.0, 100.0), 1)
            efficiency = _as_pct(props.get("efficiency"))
            if efficiency is not None:
                power["solarEfficiencyPct"] = efficiency

        elif component_type == "battery":
            health = _as_pct(props.get("health"))
            if health is not None:
                power["batteryHealthPct"] = health
            capacity = props.get("capacity")
            if capacity is not None:
                power["batteryCapacity"] = capacity

    return power or None


def _hardware_spec(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    custom = node.get("customProperties")
    if not isinstance(custom, dict):
        return None

    spec: Dict[str, Any] = {}
    for source, target in (
        ("processor", "processor"),
        ("processorSpeed", "clockSpeed"),
        ("clockSpeed", "clockSpeed"),
        ("memory", "memory"),
        ("storage", "storage"),
    ):
        value = custom.get(source)
        if value is not None and target not in spec:
            spec[target] = value

    return spec or None


_METRIC_LABELS: Dict[str, str] = {
    "wattage": "Rated output",
    "currentOutput": "Current output",
    "efficiency": "Efficiency",
    "capacity": "Capacity",
    "health": "Health",
    "currentCharge": "Charge",
    "speed": "Speed",
    "maxSpeed": "Max speed",
    "temperature": "Temperature",
    "gain": "Gain",
    "volume": "Volume",
    "frequency": "Frequency",
    "sensitivity": "Sensitivity",
    "maxTemp": "Max temperature",
    "minTemp": "Min temperature",
    "maxHumidity": "Max humidity",
    "minHumidity": "Min humidity",
    "maxAcc": "Max acceleration",
    "maxGyro": "Max gyro",
}

# Keys whose values are 0-1 fractions and read better as percentages.
_FRACTION_KEYS = {"efficiency", "health", "currentCharge"}

_METRIC_UNITS: Dict[str, str] = {
    "wattage": "W",
    "currentOutput": "W",
    "speed": "RPM",
    "maxSpeed": "RPM",
    "temperature": "\u00b0C",
    "maxTemp": "\u00b0C",
    "minTemp": "\u00b0C",
    "maxHumidity": "%",
    "minHumidity": "%",
    "gain": "dB",
    "frequency": "Hz",
    "maxAcc": "g",
    "maxGyro": "\u00b0/s",
}


def _format_metric(key: str, value: Any) -> str:
    if key in _FRACTION_KEYS:
        pct = _as_pct(value)
        if pct is not None:
            return f"{pct}%"

    number = _as_float(value)
    if number is None:
        return str(value)

    text = f"{number:g}"
    unit = _METRIC_UNITS.get(key)
    return f"{text} {unit}" if unit else text


def _normalised_components(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten each component into a display-ready set of labelled readings."""
    items: List[Dict[str, Any]] = []

    for component in _components(node):
        props = _component_props(component)
        metrics = [
            {
                "key": key,
                "label": _METRIC_LABELS.get(key, key),
                "value": value,
                "display": _format_metric(key, value),
            }
            for key, value in props.items()
            if value is not None
        ]

        items.append(
            {
                "componentId": component.get("id"),
                "type": component.get("type"),
                "category": component.get("category"),
                "model": component.get("model"),
                "metrics": metrics,
            }
        )

    return items


def _extract_gps(node: Dict[str, Any]) -> Optional[Dict[str, float]]:
    location = node.get("location")
    if isinstance(location, dict):
        lat = location.get("latitude", location.get("lat"))
        lon = location.get("longitude", location.get("lon", location.get("lng")))
        try:
            if lat is not None and lon is not None:
                return {"lat": float(lat), "lon": float(lon)}
        except Exception:
            pass

    custom = node.get("customProperties")
    if isinstance(custom, dict):
        lat = custom.get("latitude", custom.get("lat"))
        lon = custom.get("longitude", custom.get("lon", custom.get("lng")))
        try:
            if lat is not None and lon is not None:
                return {"lat": float(lat), "lon": float(lon)}
        except Exception:
            pass

    microphone_lla = node.get("microphoneLLA")
    if isinstance(microphone_lla, (list, tuple)) and len(microphone_lla) >= 2:
        try:
            return {"lat": float(microphone_lla[0]), "lon": float(microphone_lla[1])}
        except Exception:
            pass

    for sensor_data in _iter_sensor_data(node):
        gps = sensor_data.get("gps")
        if isinstance(gps, dict):
            lat = gps.get("lat", gps.get("latitude"))
            lon = gps.get("lon", gps.get("lng", gps.get("longitude")))
            try:
                if lat is not None and lon is not None:
                    return {"lat": float(lat), "lon": float(lon)}
            except Exception:
                continue
        lat = sensor_data.get("lat", sensor_data.get("latitude"))
        lon = sensor_data.get("lon", sensor_data.get("lng", sensor_data.get("longitude")))
        try:
            if lat is not None and lon is not None:
                return {"lat": float(lat), "lon": float(lon)}
        except Exception:
            continue

    return None


def _connected_devices(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve linked node ids to labels so the detail view can navigate the mesh."""
    connected = node.get("connectedNodes")
    if not isinstance(connected, list):
        return []

    ids = [n for n in connected if isinstance(n, str) and n]
    if not ids:
        return []

    found = {
        doc["_id"]: doc
        for doc in Nodes.find({"_id": {"$in": ids}}, {"_id": 1, "name": 1, "type": 1, "model": 1})
        if isinstance(doc.get("_id"), str)
    }

    return [
        {
            "sensorId": node_id,
            "name": found.get(node_id, {}).get("name"),
            "type": found.get(node_id, {}).get("type"),
            "model": found.get(node_id, {}).get("model"),
            "known": node_id in found,
        }
        for node_id in ids
    ]


def _recent_audio_events(sensor_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    cursor = (
        Events.find(
            {"sensorId": sensor_id},
            {
                "timestamp": 1,
                "species": 1,
                "confidence": 1,
                "sampleRate": 1,
                "microphoneLLA": 1,
            },
        )
        .sort("timestamp", -1)
        .limit(limit)
    )

    items: List[Dict[str, Any]] = []
    for doc in cursor:
        event_id = doc.get("_id")
        items.append(
            {
                "eventId": str(event_id) if event_id is not None else None,
                "timestamp": doc.get("timestamp"),
                "species": doc.get("species"),
                "confidence": doc.get("confidence"),
                "sampleRate": doc.get("sampleRate"),
                "microphoneLLA": doc.get("microphoneLLA"),
            }
        )
    return items


def _sensor_health_payload(
    node: Dict[str, Any],
    last_audio_ts: Optional[datetime.datetime],
    now: datetime.datetime,
) -> Optional[Dict[str, Any]]:
    sensor_id = node.get("_id")
    if not isinstance(sensor_id, str) or not sensor_id:
        return None

    settings = _get_settings_for_sensor(sensor_id)
    online_window = int(settings.get("onlineWindowMinutes", DEFAULT_SETTINGS["onlineWindowMinutes"]))
    degraded_window = int(settings.get("degradedWindowMinutes", DEFAULT_SETTINGS["degradedWindowMinutes"]))
    battery_threshold = float(settings.get("batteryThresholdPct", DEFAULT_SETTINGS["batteryThresholdPct"]))

    last_seen_dt = _parse_iso_datetime(node.get("lastSeen"))
    last_seen_mins = _minutes_ago(last_seen_dt, now)
    battery_pct = _extract_battery_pct(node)

    # "Unknown" (never sent a heartbeat) is deliberately distinct from "Offline"
    # (used to report, now silent). Live heartbeat ingestion is not connected yet,
    # so treating an absent lastSeen as a failure raises false critical alerts.
    if last_seen_mins is None:
        status = "Unknown"
    elif last_seen_mins <= online_window:
        status = "Online"
    elif last_seen_mins <= degraded_window:
        status = "Degraded"
    else:
        status = "Offline"

    if status != "Unknown" and battery_pct is not None and battery_pct <= battery_threshold:
        status = "Low Battery"

    connected = node.get("connectedNodes")
    if not isinstance(connected, list):
        connected = []

    return {
        "sensorId": sensor_id,
        "name": node.get("name"),
        "project": _project_name(node),
        "status": status,
        "type": node.get("type"),
        "model": node.get("model"),
        "hardware": _hardware_spec(node),
        "temperatureC": _extract_temperature_c(node),
        "power": _extract_power(node),
        "connectedNodes": [n for n in connected if isinstance(n, str)],
        "componentCount": len(_components(node)),
        "batteryPct": battery_pct,
        "cpu": _extract_component_number(node, ["cpu", "cpuPct", "cpu_percent", "cpuUsage"], as_percent=True),
        "ram": _extract_component_number(node, ["ram", "ramPct", "memory", "memoryPct"], as_percent=True),
        "disk": _extract_component_number(node, ["disk", "diskPct", "storage", "storagePct"], as_percent=True),
        "uptime": _extract_component_number(node, ["uptime", "uptimeSeconds", "uptime_seconds"], as_percent=False),
        "gps": _extract_gps(node),
        "lastSeen": node.get("lastSeen"),
        "lastSeenMinutesAgo": last_seen_mins,
        "lastAudioTs": last_audio_ts,
        "lastAudioMinutesAgo": _minutes_ago(last_audio_ts, now),
    }


def _project_name(node: Dict[str, Any]) -> Optional[str]:
    custom = node.get("customProperties")
    if isinstance(custom, dict):
        for key in ("project", "deployment", "site", "location"):
            val = custom.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _get_settings_for_sensor(sensor_id: str) -> Dict[str, Any]:
    doc = SensorSettings.find_one({"_id": sensor_id})
    if not doc:
        doc = SensorSettings.find_one({"_id": "__default__"})

    settings = dict(DEFAULT_SETTINGS)
    if doc and isinstance(doc.get("settings"), dict):
        settings.update(doc["settings"])
    return settings


@router.get("/updates", response_description="Derived sensor health updates")
def get_sensor_updates(
    limit: int = Query(500, ge=1, le=5000),
):
    try:
        nodes = list(
            Nodes.find(
                {},
                {
                    "_id": 1,
                    "name": 1,
                    "type": 1,
                    "model": 1,
                    "customProperties": 1,
                    "lastSeen": 1,
                    "components": 1,
                    "connectedNodes": 1,
                    "location": 1,
                    "microphoneLLA": 1,
                },
            ).limit(limit)
        )
        node_ids = [n.get("_id") for n in nodes if isinstance(n.get("_id"), str) and n.get("_id")]

        last_audio_by_sensor: Dict[str, datetime.datetime] = {}
        if node_ids:
            pipeline = [
                {"$match": {"sensorId": {"$in": node_ids}}},
                {"$sort": {"timestamp": -1}},
                {"$group": {"_id": "$sensorId", "lastAudioTs": {"$first": "$timestamp"}}},
            ]
            for row in Events.aggregate(pipeline):
                sensor_id = row.get("_id")
                ts = row.get("lastAudioTs")
                if isinstance(sensor_id, str) and isinstance(ts, datetime.datetime):
                    last_audio_by_sensor[sensor_id] = ts

        now = datetime.datetime.now(datetime.timezone.utc)

        updates: List[Dict[str, Any]] = []
        for node in nodes:
            sensor_id = node.get("_id")
            if not isinstance(sensor_id, str) or not sensor_id:
                continue
            payload = _sensor_health_payload(node, last_audio_by_sensor.get(sensor_id), now)
            if payload:
                updates.append(payload)

        return jsonable_encoder({"items": updates, "count": len(updates)})
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error deriving sensor updates: {str(error)}")


@router.get("/alerts", response_description="Derived alerts from sensor health")
def get_sensor_alerts(
    limit: int = Query(500, ge=1, le=5000),
):
    try:
        updates_payload = get_sensor_updates(limit=limit)
        items = updates_payload.get("items", []) if isinstance(updates_payload, dict) else []

        alerts: List[Dict[str, Any]] = []
        for item in items:
            status = item.get("status")
            sensor_id = item.get("sensorId")
            # "Unknown" means the device has never checked in, which is a data-pipeline
            # gap rather than a device fault, so it is not raised as an alert.
            if not sensor_id or status in ("Online", "Unknown"):
                continue

            severity = "Medium"
            issue = status
            details = ""

            if status == "Offline":
                severity = "Critical"
                mins = item.get("lastSeenMinutesAgo")
                details = "No contact" if mins is None else f"No contact for {mins} minutes"
            elif status == "Low Battery":
                severity = "High"
                battery = item.get("batteryPct")
                details = "Battery low" if battery is None else f"Battery at {battery}%"
            elif status == "Degraded":
                severity = "Medium"
                mins = item.get("lastSeenMinutesAgo")
                details = "Irregular heartbeat" if mins is None else f"Last contact {mins} minutes ago"

            alerts.append(
                {
                    "sensorId": sensor_id,
                    "severity": severity,
                    "issue": issue,
                    "details": details,
                    "lastAudioTs": item.get("lastAudioTs"),
                    "lastAudioMinutesAgo": item.get("lastAudioMinutesAgo"),
                }
            )

        return jsonable_encoder({"items": alerts, "count": len(alerts)})
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error deriving alerts: {str(error)}")


@router.get("/{sensor_id}/settings", response_description="Get sensor settings (sensor-specific or global defaults)")
def get_sensor_settings(sensor_id: str):
    try:
        settings = _get_settings_for_sensor(sensor_id)
        return jsonable_encoder({"sensorId": sensor_id, "settings": settings})
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error retrieving settings: {str(error)}")


@router.put("/{sensor_id}/settings", response_description="Upsert sensor settings")
def put_sensor_settings(sensor_id: str, payload: Dict[str, Any] = Body(...)):
    try:
        # Accept either { settings: {...} } or a raw settings object
        settings_update = payload.get("settings") if isinstance(payload.get("settings"), dict) else payload
        if not isinstance(settings_update, dict):
            raise HTTPException(status_code=400, detail="Settings payload must be an object")

        SensorSettings.update_one(
            {"_id": sensor_id},
            {
                "$set": {
                    "settings": settings_update,
                    "updatedAt": datetime.datetime.now(datetime.timezone.utc),
                }
            },
            upsert=True,
        )
        merged = _get_settings_for_sensor(sensor_id)
        return jsonable_encoder({"sensorId": sensor_id, "settings": merged})
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error saving settings: {str(error)}")


@router.post("/{sensor_id}/reboot", response_description="Queue a reboot command (records intent only)")
def queue_reboot(sensor_id: str, payload: Dict[str, Any] = Body(default={})):  # type: ignore[assignment]
    try:
        reason = None
        if isinstance(payload, dict):
            reason_val = payload.get("reason")
            if isinstance(reason_val, str) and reason_val.strip():
                reason = reason_val.strip()

        doc = {
            "sensorId": sensor_id,
            "reason": reason,
            "status": "Queued",
            "requestedAt": datetime.datetime.now(datetime.timezone.utc),
        }
        result = SensorReboots.insert_one(doc)
        return jsonable_encoder({"rebootId": str(result.inserted_id), "sensorId": sensor_id, "status": "Queued"})
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error queuing reboot: {str(error)}")


@router.get("/{sensor_id}/reboots", response_description="Reboot history for a sensor")
def get_reboot_history(sensor_id: str, limit: int = Query(50, ge=1, le=200)):
    try:
        cursor = SensorReboots.find({"sensorId": sensor_id}).sort("requestedAt", -1).limit(limit)
        items: List[Dict[str, Any]] = []
        for doc in cursor:
            doc_id = doc.get("_id")
            if isinstance(doc_id, ObjectId):
                doc["_id"] = str(doc_id)
            items.append(doc)
        return jsonable_encoder({"items": items, "count": len(items)})
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error retrieving reboot history: {str(error)}")


@router.get("/reboots/recent", response_description="Recent reboot history across all sensors")
def get_recent_reboots(limit: int = Query(50, ge=1, le=200)):
    try:
        cursor = SensorReboots.find({}).sort("requestedAt", -1).limit(limit)
        items: List[Dict[str, Any]] = []
        for doc in cursor:
            doc_id = doc.get("_id")
            if isinstance(doc_id, ObjectId):
                doc["_id"] = str(doc_id)
            items.append(doc)
        return jsonable_encoder({"items": items, "count": len(items)})
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error retrieving recent reboots: {str(error)}")


@router.get("/{sensor_id}", response_description="Get a single sensor's current health, location, and recent audio metadata")
def get_sensor_detail(
    sensor_id: str,
    history_limit: int = Query(10, ge=1, le=50),
):
    try:
        node = Nodes.find_one({"_id": sensor_id})
        if not node:
            raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' was not found")

        last_audio_doc = Events.find_one(
            {"sensorId": sensor_id},
            {"timestamp": 1, "species": 1, "confidence": 1, "sampleRate": 1},
            sort=[("timestamp", -1)],
        )
        last_audio_ts = last_audio_doc.get("timestamp") if last_audio_doc else None
        if last_audio_ts is not None and not isinstance(last_audio_ts, datetime.datetime):
            last_audio_ts = None

        now = datetime.datetime.now(datetime.timezone.utc)
        payload = _sensor_health_payload(node, last_audio_ts, now)
        if not payload:
            raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' was not found")

        last_audio_meta = None
        if last_audio_doc:
            last_audio_meta = {
                "timestamp": last_audio_doc.get("timestamp"),
                "species": last_audio_doc.get("species"),
                "confidence": last_audio_doc.get("confidence"),
                "sampleRate": last_audio_doc.get("sampleRate"),
            }

        payload["lastAudio"] = last_audio_meta
        payload["recentAudio"] = _recent_audio_events(sensor_id, history_limit)
        payload["components"] = _normalised_components(node)
        payload["connectedDevices"] = _connected_devices(node)
        return jsonable_encoder(payload)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error retrieving sensor detail: {str(error)}")
