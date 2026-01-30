import paho.mqtt.client as mqtt
import os
import time

SAVE_DIR = "audioLocal"
os.makedirs(SAVE_DIR, exist_ok=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Subscriber connected successfully")
        client.subscribe(topic, qos=1)
    else:
        print(f"Subscriber failed to connect, code {rc}")

def on_subscribe(client, userdata, mid, granted_qos):
    print(f"Subscribed with QoS {granted_qos}")

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}")
    print("Writing")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"audio_{timestamp}.wav"
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(msg.payload)

client = mqtt.Client()
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
topic = "iot/data/test"
client.subscribe(topic, qos=1)
print(f"Subscribed to {topic}")
client.loop_forever()