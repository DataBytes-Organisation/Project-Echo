import os
import paho.mqtt.client as mqtt
import json
from gps3 import gps3
import time

print("Connecting to MQTT Broker...")
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)
topic = "iot/data/test"
print("Connected.")

def get_gps_reading():
    gps_socket = gps3.GPSDSocket()
    data_stream = gps3.DataStream()

    gps_socket.connect()
    gps_socket.watch()

    for new_data in gps_socket:
        if new_data:
            data_stream.unpack(new_data)

            lat = data_stream.TPV['lat']
            lon = data_stream.TPV['lon']

    return {"lat": lat, "lon": lon}

def send_data():
    gps_data = get_gps_reading()
    
    json_payload = json.dumps(gps_data)

    client.publish(topic, json_payload)
    print(f"Published to {topic}: {json_payload}")

if __name__ == "__main__":
    while True:
        send_data()
        time.sleep(20)