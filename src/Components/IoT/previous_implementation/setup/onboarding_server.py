import json
import os
import paho.mqtt.client as mqtt

BROKER = "a44b74302a2b4739adb489de4f26aa03.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "EchoUsername"
PASSWORD = "ProjectEcho123"

ONBOARD_REQUEST_TOPIC = "devices/onboard/request"
ONBOARD_RESPONSE_TOPIC_TEMPLATE = "devices/onboard/response/{}"
CONFIG_FILE = "../config/devices.json"

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(ONBOARD_REQUEST_TOPIC)
    print(f"Subscribed to {ONBOARD_REQUEST_TOPIC}")

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}: {msg.payload.decode()}")
    try:
        data = json.loads(msg.payload.decode())
        device_id = data.get("device_id")
        ip = data.get("ip")
        username = data.get("username", "pi")
        key_path = "~/.ssh/id_rsa"

        print(f"Onboarding request from device: {device_id}")

        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                devices = json.load(f)
        else:
            devices = []

        if not any(d.get("name") == device_id for d in devices):
            new_device = {
                "name": device_id,
                "ip": ip,
                "username": username,
                "key_path": key_path
            }
            devices.append(new_device)

            with open(CONFIG_FILE, "w") as f:
                json.dump(devices, f, indent=2)
            print(f"Added new device to config: {new_device}")
        else:
            print(f"Device {device_id} already exists in config")

        response = {
            "status": "success",
            "message": f"Device {device_id} onboarded successfully",
        }

        response_topic = ONBOARD_RESPONSE_TOPIC_TEMPLATE.format(device_id)
        client.publish(response_topic, json.dumps(response))
        print(f"Sent onboarding response to {response_topic}")

    except Exception as e:
        print(f"Error processing onboarding request: {e}")
def main():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT)
    client.loop_forever()

if __name__ == "__main__":
    main()
