import json
import os

import paho.mqtt.client as mqtt
from pydantic import ValidationError

from app.schemas import EventSchema
from app.services.event_ingestion import persist_event


connection_state = "disconnected"
latest_events = {}


MQTT_BROKER_URL = os.environ.get(
    "MQTT_BROKER_URL",
    "ts-mqtt-server-cont",
)

MQTT_BROKER_PORT = int(
    os.environ.get("MQTT_BROKER_PORT", 1883)
)

MQTT_DETECTION_TOPIC = os.environ.get(
    "MQTT_DETECTION_TOPIC",
    "projectecho/backend/detections",
)

MQTT_TOPICS = [
    topic.strip()
    for topic in os.environ.get(
        "MQTT_TOPICS",
        "projectecho/engine/2,projectecho/movement,iot/data/test",
    ).split(",")
    if topic.strip()
]

if MQTT_DETECTION_TOPIC not in MQTT_TOPICS:
    MQTT_TOPICS.append(MQTT_DETECTION_TOPIC)


def on_connect(client, userdata, flags, rc):
    global connection_state

    connection_state = "connected"
    print(f"[MQTT] Connected, rc={rc}")

    for topic in MQTT_TOPICS:
        client.subscribe(topic)
        print(f"[MQTT] Subscribed to {topic}")


def on_disconnect(client, userdata, rc):
    global connection_state

    connection_state = "reconnecting"
    print(
        f"[MQTT] Disconnected, rc={rc} - attempting reconnect"
    )


def persist_detection_payload(raw_payload):
    """
    Validate a classified detection MQTT message against EventSchema
    and persist it through the shared Backend persistence service.
    """
    try:
        data = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        print(
            f"[MQTT] Detection rejected: invalid JSON ({exc})"
        )
        return None

    try:
        event = EventSchema(**data)
    except ValidationError as exc:
        print(
            f"[MQTT] Detection rejected by EventSchema: {exc}"
        )
        return None

    try:
        inserted_id = persist_event(event)
    except Exception as exc:
        print(
            f"[MQTT] Detection persistence failed: {exc}"
        )
        return None

    print(
        "[MQTT] Detection persisted successfully "
        f"eventId={inserted_id} "
        f"sensorId={event.sensorId} "
        f"species={event.species}"
    )

    return str(inserted_id)


def on_message(client, userdata, msg):
    if msg.topic == MQTT_DETECTION_TOPIC:
        persist_detection_payload(msg.payload)
        return

    normalized = normalize_payload(
        msg.payload,
        msg.topic,
    )

    if normalized:
        print(
            f"[MQTT] Normalized event: {normalized}"
        )

        key = normalized.get(
            "_id",
            "unknown",
        )

        latest_events[key] = normalized


def normalize_payload(raw_payload, topic):
    """
    Convert existing MQTT messages into shapes used by
    the frontend/dashboard.
    """
    try:
        data = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError):
        print(
            f"[MQTT] Could not parse payload on "
            f"{topic} as JSON"
        )
        return None

    if "animalEstLLA" in data or "audioClip" in data:
        return {
            "eventType": "vocalization",
            "_id": (
                f"{data.get('sensorId', 'unknown')}_"
                f"{data.get('timestamp', '')}"
            ),
            "timestamp": data.get("timestamp"),
            "confidence": data.get(
                "animalLLAUncertainty"
            ),
            "species": "unclassified",
            "commonName": "unclassified",
            "type": "mammal",
            "status": "normal",
            "diet": "unknown",
            "animalLLAUncertainty": data.get(
                "animalLLAUncertainty"
            ),
            "animalEstLLA": data.get(
                "animalEstLLA"
            ),
            "animalTrueLLA": data.get(
                "animalTrueLLA"
            ),
            "sensorId": data.get(
                "sensorId"
            ),
            "microphoneLLA": data.get(
                "microphoneLLA"
            ),
        }

    if "animalId" in data and "species" in data:
        return {
            "eventType": "movement",
            "_id": (
                f"{data.get('animalId', 'unknown')}_"
                f"{data.get('timestamp', '')}"
            ),
            "timestamp": data.get("timestamp"),
            "animalId": data.get("animalId"),
            "species": data.get("species"),
            "animalTrueLLA": data.get(
                "animalTrueLLA"
            ),
            "type": "mammal",
            "status": "normal",
            "diet": "omnivore",
        }

    if "cpu" in data or "batteryPct" in data:
        return {
            "eventType": "sensor_health",
            "_id": (
                f"{data.get('sensorId', 'unknown')}_"
                f"{data.get('timestamp', '')}"
            ),
            "timestamp": data.get("timestamp"),
            "sensorId": data.get("sensorId"),
            "status": data.get("status"),
            "batteryPct": data.get(
                "batteryPct"
            ),
            "cpu": data.get("cpu"),
            "ram": data.get("ram"),
        }

    if "nodeId" in data:
        return {
            "eventType": "iot_node",
            "_id": (
                f"{data.get('nodeId', 'unknown')}_"
                f"{data.get('timestamp', '')}"
            ),
            "timestamp": data.get("timestamp"),
            "nodeId": data.get("nodeId"),
            "status": data.get("status"),
        }

    print(
        f"[MQTT] Unrecognized payload shape on "
        f"{topic}: {list(data.keys())}"
    )

    return {
        "eventType": "unknown",
        "raw": data,
    }


def start_mqtt_client():
    global connection_state

    connection_state = "connecting"

    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    client.reconnect_delay_set(
        min_delay=1,
        max_delay=30,
    )

    try:
        client.connect(
            MQTT_BROKER_URL,
            MQTT_BROKER_PORT,
        )
    except Exception as exc:
        connection_state = "reconnecting"

        print(
            f"[MQTT] Initial connect failed: {exc} "
            "- will keep retrying in background"
        )

    client.loop_start()

    return client


def get_connection_state():
    return connection_state


def get_latest_events():
    return list(latest_events.values())