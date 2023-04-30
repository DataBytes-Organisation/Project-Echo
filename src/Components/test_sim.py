import paho.mqtt.publish as publish
import time

# Set the MQTT broker URL and port
MQTT_BROKER_URL = "localhost"
MQTT_BROKER_PORT = 1883

publish.single("Simulator_Controls", "Start", hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)

publish.single("Simulator_Controls", "Stop", hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)