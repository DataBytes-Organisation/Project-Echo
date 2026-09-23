import logging
import os
import json

import paho.mqtt.client as mqtt
from pydantic import ValidationError

from app.schemas import EventSchema
from app.services.event_ingestion import persist_event


logger = logging.getLogger(__name__)

connection_state = "disconnected"  # disconnected | connecting | connected | reconnecting
latest_events = {}

# The single paho client (and its network-loop thread) owned by this process.
# Set by start_mqtt_client() and released by stop_mqtt_client() on shutdown.
_client = None

MQTT_BROKER_URL = os.environ.get("MQTT_BROKER_URL", "ts-mqtt-server-cont")
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", 1883))
MQTT_TOPICS = os.environ.get(
    "MQTT_TOPICS",
    "projectecho/engine/2,projectecho/movement,iot/data/test",
).split(",")

MQTT_DETECTION_TOPIC = os.environ.get(
    "MQTT_DETECTION_TOPIC",
    "projectecho/backend/detections",
)

MQTT_TOPICS = [
    topic.strip()
    for topic in MQTT_TOPICS
    if topic.strip()
]

if MQTT_DETECTION_TOPIC not in MQTT_TOPICS:
    MQTT_TOPICS.append(MQTT_DETECTION_TOPIC)


def on_connect(client, userdata, flags, rc):
    global connection_state
    connection_state = "connected"
    logger.info("MQTT client connected, rc=%s", rc)
    for topic in MQTT_TOPICS:
        client.subscribe(topic.strip())

def on_disconnect(client, userdata, rc):
    global connection_state
    if rc == mqtt.MQTT_ERR_SUCCESS:
        # rc 0 means we asked to disconnect (stop_mqtt_client); paho will not reconnect.
        connection_state = "disconnected"
        logger.info("MQTT client disconnected cleanly")
        return
    connection_state = "reconnecting"
    logger.warning("MQTT client disconnected, rc=%s; attempting reconnect", rc)

def persist_detection_payload(raw_payload):
    """
    Validate a classified detection MQTT message using EventSchema
    and persist it through the shared Backend event path.
    """
    try:
        data = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        logger.warning("MQTT detection rejected: invalid JSON (%s)", exc)
        return None

    try:
        event = EventSchema(**data)
    except ValidationError as exc:
        logger.warning(
            "MQTT detection rejected by EventSchema: %s",
            exc,
        )
        return None

    try:
        inserted_id = persist_event(event)
    except Exception as exc:
        logger.exception(
            "MQTT detection persistence failed: %s",
            exc,
        )
        return None

    logger.info(
        "MQTT detection persisted successfully "
        "eventId=%s sensorId=%s species=%s",
        inserted_id,
        event.sensorId,
        event.species,
    )

    return str(inserted_id)


def on_message(client, userdata, msg):
    if msg.topic == MQTT_DETECTION_TOPIC:
        persist_detection_payload(msg.payload)
        return

    normalized = normalize_payload(msg.payload, msg.topic)
    if normalized:
        logger.debug(
            "MQTT event received, event_type=%s",
            normalized.get("eventType", "unknown"),
        )
        key = normalized.get("_id", "unknown")
        latest_events[key] = normalized
    # TODO: forward `normalized` to wherever the frontend/dashboard reads from

def normalize_payload(raw_payload, topic):
    """
    Converts a raw MQTT message into a shape the frontend can use directly.
    Routes by event type: vocalization, movement, sensor_health, iot_node.
    """
    try:
        data = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError):
        logger.warning("MQTT payload could not be parsed as JSON")
        return None

    # Vocalization / recording events (from comms_manager.py's
    # mqtt_send_random_audio_msg and mqtt_send_recording_msg).
    # NOTE: species classification (species/commonName/status/diet) is NOT
    # available at this stage — it only exists after the engine classifies
    # the audio and posts the result to MongoDB via HTTP, not over MQTT.
    # We use clear placeholders here rather than fabricate values.
    if "animalEstLLA" in data or "audioClip" in data:
        return {
            "eventType": "vocalization",
            "_id": f"{data.get('sensorId', 'unknown')}_{data.get('timestamp', '')}",
            "timestamp": data.get("timestamp"),
            "confidence": data.get("animalLLAUncertainty"),
            "species": "unclassified",
            "commonName": "unclassified",
            "type": "mammal",
            "status": "normal",
            "diet": "unknown",
            "animalLLAUncertainty": data.get("animalLLAUncertainty"),
            "animalEstLLA": data.get("animalEstLLA"),
            "animalTrueLLA": data.get("animalTrueLLA"),
            "sensorId": data.get("sensorId"),
            "microphoneLLA": data.get("microphoneLLA"),
        }

    # Movement events
    if "animalId" in data and "species" in data:
        return {
            "eventType": "movement",
            "_id": f"{data.get('animalId', 'unknown')}_{data.get('timestamp', '')}",
            "timestamp": data.get("timestamp"),
            "animalId": data.get("animalId"),
            "species": data.get("species"),
            "animalTrueLLA": data.get("animalTrueLLA"),
            # Placeholders: real species metadata isn't available on this
            # MQTT payload, only after DB enrichment (see vocalization
            # branch above for the same limitation).
            "type": "mammal",
            "status": "normal",
            "diet": "omnivore",
        }

    # Sensor health events
    if "cpu" in data or "batteryPct" in data:
        return {
            "eventType": "sensor_health",
            "_id": f"{data.get('sensorId', 'unknown')}_{data.get('timestamp', '')}",
            "timestamp": data.get("timestamp"),
            "sensorId": data.get("sensorId"),
            "status": data.get("status"),
            "batteryPct": data.get("batteryPct"),
            "cpu": data.get("cpu"),
            "ram": data.get("ram"),
        }

    # IoT node updates
    if "nodeId" in data:
        return {
            "eventType": "iot_node",
            "_id": f"{data.get('nodeId', 'unknown')}_{data.get('timestamp', '')}",
            "timestamp": data.get("timestamp"),
            "nodeId": data.get("nodeId"),
            "status": data.get("status"),
        }

    logger.warning("MQTT payload shape not recognized")
    return {"eventType": "unknown", "raw": data}

def start_mqtt_client():
    global connection_state, _client
    if _client is not None:
        return _client
    connection_state = "connecting"
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    try:
        client.connect(MQTT_BROKER_URL, MQTT_BROKER_PORT)
    except Exception:
        connection_state = "reconnecting"
        logger.warning(
            "MQTT initial connection failed; will keep retrying in background"
        )
    client.loop_start()
    _client = client
    return client

def stop_mqtt_client():
    """
    Disconnect the owned MQTT client and join its network-loop thread.
    Safe to call when the client was never started or is already stopped.
    """
    global connection_state, _client
    client, _client = _client, None
    if client is None:
        return
    try:
        # Sends DISCONNECT when connected; when still retrying it just tells
        # the loop thread to stop reconnecting.
        client.disconnect()
    except Exception:
        logger.warning("MQTT disconnect failed during shutdown", exc_info=True)
    finally:
        client.loop_stop()
        connection_state = "disconnected"
        logger.info("MQTT client stopped and network loop joined")

def get_connection_state():
    return connection_state

def get_latest_events():
    return list(latest_events.values())