import json
import paho.mqtt.client as mqtt
import os

JSON_DIR = 'jsonLocal'
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "iot/data/test"

os.makedirs(JSON_DIR, exist_ok=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Subscriber connected successfully")
        client.subscribe(TOPIC, qos=1)
    else:
        print(f"Subscriber failed to connect, code {rc}")

def on_subscribe(client, userdata, mid, granted_qos):
    print(f"Subscribed with QoS {granted_qos}")

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}")

    payload = json.loads(msg.payload)

    print("Writing json data")
    subfolder = os.path.join(JSON_DIR, f"{payload["sensor_id"]}")
    os.makedirs(subfolder, exist_ok=True)
    json_filename = f"data_{payload["timestamp"]}"
    with open(os.path.join(subfolder, json_filename), "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved JSON: {json_filename}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_forever()