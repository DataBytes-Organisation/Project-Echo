import paho.mqtt.client as mqtt
import json

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}: {msg.payload.decode()}")
    try:
        data = json.loads(msg.payload.decode())
        print(f"Parsed JSON data: {data}")
    except json.JSONDecodeError:    
        print("Failed to decode JSON")

client = mqtt.Client()
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
topic = "iot/data/test"
client.subscribe(topic)
print(f"Subscribed to {topic}")
client.loop_forever()