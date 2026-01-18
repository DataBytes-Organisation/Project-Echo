import paho.mqtt.client as mqtt
import os
import time
import json
import base64

AUDIO_DIR = "audioLocal"
JSON_DIR = "jsonLocal"
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "iot/data/test"

os.makedirs(AUDIO_DIR, exist_ok=True)
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
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    json_filename = f"data_{timestamp}.json"
    with open(os.path.join(JSON_DIR, json_filename), "w") as f:
        json.dump(payload["health_data"], f)
    print(f"Saved JSON: {json_filename}")

    print("Writing audio data")
    audio_bytes = base64.b64decode(payload["audio_file"])
    audio_file = f"audio_{timestamp}.wav"
    with open(os.path.join(AUDIO_DIR, audio_file), "wb") as f:
        f.write(audio_bytes)
    print(f"Saved audio: {audio_file}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_forever()