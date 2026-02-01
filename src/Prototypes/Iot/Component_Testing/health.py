import psutil
import paho.mqtt.client as mqtt
import json
import time

print("Connecting to MQTT Broker...")
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)
topic = "iot/data/test"
print("Connected.")

def get_health_report():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "uptime": psutil.boot_time()
    }

def send_data():
    health_data = get_health_report()
    
    json_payload = json.dumps(health_data)

    client.publish(topic, json_payload)
    print(f"Published to {topic}: {json_payload}")

if __name__ == "__main__":
    while True:
        send_data()
        time.sleep(20)


