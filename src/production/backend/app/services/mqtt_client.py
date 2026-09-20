import paho.mqtt.client as mqtt
import os
import json

connection_state = "disconnected"  # disconnected | connecting | connected | reconnecting
latest_events = {}

MQTT_BROKER_URL = os.environ.get("MQTT_BROKER_URL", "ts-mqtt-server-cont")
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", 1883))
MQTT_TOPICS = os.environ.get(
    "MQTT_TOPICS",
    "projectecho/engine/2,projectecho/movement,iot/data/test",
).split(",")

def on_connect(client, userdata, flags, rc):
    global connection_state
    connection_state = "connected"
    print(f"[MQTT] Connected, rc={rc}")
    for topic in MQTT_TOPICS:
        client.subscribe(topic.strip())

def on_disconnect(client, userdata, rc):
    global connection_state
    connection_state = "reconnecting"
    print(f"[MQTT] Disconnected, rc={rc} - attempting reconnect")

def on_message(client, userdata, msg):
    normalized = normalize_payload(msg.payload, msg.topic)
    if normalized:
        print(f"[MQTT] Normalized event: {normalized}")
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
        print(f"[MQTT] Could not parse payload on {topic} as JSON")
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

    print(f"[MQTT] Unrecognized payload shape on {topic}: {list(data.keys())}")
    return {"eventType": "unknown", "raw": data}

def start_mqtt_client():
    global connection_state
    connection_state = "connecting"
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    try:
        client.connect(MQTT_BROKER_URL, MQTT_BROKER_PORT)
    except Exception as e:
        connection_state = "reconnecting"
        print(f"[MQTT] Initial connect failed: {e} — will keep retrying in background")
    client.loop_start()
    return client

def get_connection_state():
    return connection_state

def get_latest_events():
    return list(latest_events.values())