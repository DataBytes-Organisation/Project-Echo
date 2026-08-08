import paho.mqtt.client as mqtt
import os
import json

connection_state = "disconnected"  # disconnected | connecting | connected | reconnecting
latest_events = {}

MQTT_BROKER_URL = os.environ.get("MQTT_BROKER_URL", "ts-mqtt-server-cont")
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", 1883))
MQTT_TOPIC = os.environ.get("MQTT_PUBLISH_URL", "projectecho/engine/2")

def on_connect(client, userdata, flags, rc):
    global connection_state
    connection_state = "connected"
    print(f"[MQTT] Connected, rc={rc}")
    client.subscribe(MQTT_TOPIC)

def on_disconnect(client, userdata, rc):
    global connection_state
    connection_state = "reconnecting"
    print(f"[MQTT] Disconnected, rc={rc} - attempting reconnect")

def on_message(client, userdata, msg):
    normalized = normalize_payload(msg.payload, msg.topic)
    if normalized:
        print(f"[MQTT] Normalized event: {normalized}")
        sensor_id = normalized.get("sensorId", "unknown")
        latest_events[sensor_id] = normalized
    # TODO: forward `normalized` to wherever the frontend/dashboard reads from

def normalize_payload(raw_payload, topic):
    """
    Converts a raw MQTT message into a consistent shape for
    downstream use (map rendering, dashboards, notifications).
    """
    try:
        data = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError):
        print(f"[MQTT] Could not parse payload on {topic} as JSON")
        return None

    # Vocalization / recording events (from comms_manager.py's
    # mqtt_send_random_audio_msg and mqtt_send_recording_msg)
    if "animalEstLLA" in data or "audioClip" in data:
        est_lla = data.get("animalEstLLA", [None, None, None])
        return {
            "type": "vocalization",
            "timestamp": data.get("timestamp"),
            "sensorId": data.get("sensorId"),
            "location": {
                "lat": est_lla[0] if len(est_lla) > 0 else None,
                "lon": est_lla[1] if len(est_lla) > 1 else None,
                "alt": est_lla[2] if len(est_lla) > 2 else None,
            },
            "confidence": data.get("animalLLAUncertainty"),
        }

    # Fallback for any unrecognized event shape
    print(f"[MQTT] Unrecognized payload shape on {topic}: {list(data.keys())}")
    return {
        "type": "unknown",
        "raw": data,
    }

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