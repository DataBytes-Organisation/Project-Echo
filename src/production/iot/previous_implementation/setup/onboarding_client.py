import json
import time
import socket
import paho.mqtt.client as mqtt

# Define this accordingly
device_id = "EchoPi-01"

BROKER = "a44b74302a2b4739adb489de4f26aa03.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "EchoUsername"
PASSWORD = "ProjectEcho123"

ONBOARD_REQUEST_TOPIC = "devices/onboard/request"
ONBOARD_RESPONSE_TOPIC = f"devices/onboard/response/{device_id}"

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(ONBOARD_RESPONSE_TOPIC)
    print(f"Subscribed to {ONBOARD_RESPONSE_TOPIC}")

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}: {msg.payload.decode()}")
    response = json.loads(msg.payload.decode())
    if response.get("status") == "success":
        print("Onboarding successful!")
    else:
        print("Onboarding failed or unknown status.")


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def main():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT)
    client.loop_start()

    actual_ip = get_local_ip()

    onboarding_data = {
        "device_id": device_id,
        "ip": actual_ip,
        "username": "pi",
    }

    print(f"Publishing onboarding request for device {device_id} with IP {actual_ip}")
    client.publish(ONBOARD_REQUEST_TOPIC, json.dumps(onboarding_data))

    time.sleep(10)

    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()
