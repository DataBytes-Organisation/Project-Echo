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


def _extract_battery_pct(node: Dict[str, Any]) -> Optional[float]:
    components = node.get("components")
    if not isinstance(components, list):
        return None

    candidates: List[float] = []
    keys = [
        "batteryPct",
        "battery_pct",
        "battery_percent",
        "batteryPercent",
        "battery",
        "batteryLevel",
        "battery_level",
    ]

    for component in components:
        if not isinstance(component, dict):
            continue
        sensor_data = component.get("sensorData")
        if not isinstance(sensor_data, dict):
            continue
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

            # Heuristics: 0..1 => fraction, 1..100 => percent
            if 0 <= number <= 1:
                candidates.append(number * 100.0)
            elif 1 < number <= 100:
                candidates.append(number)

    if not candidates:
        return None
    return round(min(candidates), 1)


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
        nodes = list(Nodes.find({}, {"_id": 1, "name": 1, "customProperties": 1, "lastSeen": 1, "components": 1}).limit(limit))
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

            settings = _get_settings_for_sensor(sensor_id)
            online_window = int(settings.get("onlineWindowMinutes", DEFAULT_SETTINGS["onlineWindowMinutes"]))
            degraded_window = int(settings.get("degradedWindowMinutes", DEFAULT_SETTINGS["degradedWindowMinutes"]))
            battery_threshold = float(settings.get("batteryThresholdPct", DEFAULT_SETTINGS["batteryThresholdPct"]))

            last_seen_dt = _parse_iso_datetime(node.get("lastSeen"))
            last_seen_mins = _minutes_ago(last_seen_dt, now)

            battery_pct = _extract_battery_pct(node)
            last_audio_ts = last_audio_by_sensor.get(sensor_id)

            status = "Offline"
            if last_seen_mins is not None:
                if last_seen_mins <= online_window:
                    status = "Online"
                elif last_seen_mins <= degraded_window:
                    status = "Degraded"
                else:
                    status = "Offline"

            if battery_pct is not None and battery_pct <= battery_threshold:
                status = "Low Battery"

            updates.append(
                {
                    "sensorId": sensor_id,
                    "name": node.get("name"),
                    "project": _project_name(node),
                    "status": status,
                    "batteryPct": battery_pct,
                    "lastSeen": node.get("lastSeen"),
                    "lastSeenMinutesAgo": last_seen_mins,
                    "lastAudioTs": last_audio_ts,
                    "lastAudioMinutesAgo": _minutes_ago(last_audio_ts, now),
                }
            )

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
            if not sensor_id or status == "Online":
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
