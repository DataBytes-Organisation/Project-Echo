import json
import time
import socket
import uuid
import os
import paho.mqtt.client as mqtt

# Path where the device ID is saved on the Pi
ID_FILE = os.path.expanduser("~/.echo_device_id")

# Broker settings aligned with T3 prototype
BROKER = "broker.hivemq.com"
PORT = 1883
ONBOARD_REQUEST_TOPIC = "devices/onboard/request"


def get_or_create_device_id():
    # Load saved device ID or generate a new one from the Pi's MAC address
    if os.path.exists(ID_FILE):
        with open(ID_FILE, "r") as f:
            device_id = f.read().strip()
            print(f"Loaded existing device ID: {device_id}")
            return device_id

    # Generate ID from MAC address and save it for future runs
    mac = uuid.getnode()
    device_id = f"EchoPi-{mac:012x}"
    with open(ID_FILE, "w") as f:
        f.write(device_id)
    print(f"Generated new device ID: {device_id}")
    return device_id


def get_local_ip():
    # Get the Pi's local IP address using a UDP socket trick
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to external address to determine outbound IP - no data is sent
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


device_id = get_or_create_device_id()

# Device specific response topic so the server can reply to this Pi only
response_topic = f"devices/onboard/response/{device_id}"

onboarding_complete = False


def on_connect(client, userdata, flags, rc):
    # Subscribe to response topic once connected
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(response_topic)
        print(f"Subscribed to {response_topic}")
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    # Handle onboarding response from server
    global onboarding_complete
    print(f"Received response on {msg.topic}: {msg.payload.decode()}")
    response = json.loads(msg.payload.decode())
    if response.get("status") == "success":
        print(f"Onboarding successful! Device ID: {response.get('device_id')}")
        onboarding_complete = True
    else:
        print(f"Onboarding failed: {response.get('message', 'Unknown error')}")


# Initialise client - CallbackAPIVersion.VERSION1 required in newer paho versions
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_connect = on_connect
client.on_message = on_message

print("Connecting to MQTT Broker...")
client.connect(BROKER, PORT, 60)
client.loop_start()
print("Connected.")

# Wait for connection and subscription to be confirmed before publishing
time.sleep(2)

actual_ip = get_local_ip()

# Build onboarding payload with device identity details
onboarding_data = {
    "device_id": device_id,
    "ip": actual_ip,
    "username": "pi",
}

print(f"Publishing onboarding request for device {device_id} with IP {actual_ip}")
client.publish(ONBOARD_REQUEST_TOPIC, json.dumps(onboarding_data), qos=1)

# Wait up to 15 seconds for a response before timing out
timeout = 15
elapsed = 0
while not onboarding_complete and elapsed < timeout:
    time.sleep(1)
    elapsed += 1

if not onboarding_complete:
    print("Onboarding timed out - no response received from server")

client.loop_stop()
client.disconnect()