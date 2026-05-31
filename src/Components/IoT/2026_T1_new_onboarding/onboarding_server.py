import json
import os
import paho.mqtt.client as mqtt

# Broker connection settings
BROKER = "a44b74302a2b4739adb489de4f26aa03.s1.eu.hivemq.cloud"
PORT = 8883  # Standard MQTT over TLS port
USERNAME = "EchoUsername"
PASSWORD = "ProjectEcho123"

ONBOARD_REQUEST_TOPIC = "devices/onboard/request"

# Response topic uses device_id so each Pi gets its own response
ONBOARD_RESPONSE_TOPIC_TEMPLATE = "devices/onboard/response/{}"

# Device registry file path, relative to this script's location
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "devices.json")


def load_devices():
    # Load device registry from disk, returns empty list if file doesn't exist
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return []


def save_devices(devices):
    # Save device registry to disk, creating the config directory if needed
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(devices, f, indent=2)


def on_connect(client, userdata, flags, rc):
    # Subscribe to onboarding request topic once connected
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(ONBOARD_REQUEST_TOPIC)
        print(f"Subscribed to {ONBOARD_REQUEST_TOPIC}")
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    # Handle incoming onboarding requests - Registers new devices or updates IP for existing ones
    print(f"Received onboarding request on {msg.topic}")
    try:
        data = json.loads(msg.payload.decode())
        device_id = data.get("device_id")
        ip = data.get("ip")
        username = data.get("username", "pi")

        # Validate required fields before proceeding
        if not device_id or not ip:
            print("Invalid onboarding request - missing device_id or ip")
            return

        print(f"Processing onboarding for device: {device_id} at {ip}")

        devices = load_devices()

        # Check if device already exists in registry
        existing = next((d for d in devices if d.get("device_id") == device_id), None)

        if existing:
            # Update IP in case it changed since last onboarding
            existing["ip"] = ip
            existing["username"] = username
            save_devices(devices)
            print(f"Updated existing device: {device_id}")
            status_message = f"Device {device_id} updated successfully"
        else:
            # Register new device
            new_device = {
                "device_id": device_id,
                "ip": ip,
                "username": username,
                "key_path": "~/.ssh/id_rsa"
            }
            devices.append(new_device)
            save_devices(devices)
            print(f"Registered new device: {device_id}")
            status_message = f"Device {device_id} onboarded successfully"

        # Send confirmation back to the device on its specific response topic
        response = {
            "status": "success",
            "device_id": device_id,
            "message": status_message,
        }
        response_topic = ONBOARD_RESPONSE_TOPIC_TEMPLATE.format(device_id)
        client.publish(response_topic, json.dumps(response), qos=1)
        print(f"Sent onboarding response to {response_topic}")

    except json.JSONDecodeError:
        print("Failed to parse onboarding request payload")
    except Exception as e:
        # Catch all errors so the server keeps running
        print(f"Error processing onboarding request: {e}")


def main():
    # Initialise client - CallbackAPIVersion.VERSION1 required in newer paho versions
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()  # Enable TLS for secure connection

    client.on_connect = on_connect
    client.on_message = on_message

    print("Starting onboarding server...")
    client.connect(BROKER, PORT)

    # loop_forever keeps the server running to handle incoming requests
    client.loop_forever()


if __name__ == "__main__":
    main()