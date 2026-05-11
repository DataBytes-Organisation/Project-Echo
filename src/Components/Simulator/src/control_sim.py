import paho.mqtt.publish as publish
import time

# Set the MQTT broker URL and port
MQTT_BROKER_URL = "echo_mqtt"
MQTT_BROKER_PORT = 1883

publish.single("Simulator_Controls", "Animal_Mode", hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)

time.sleep(30)

publish.single("Simulator_Controls", "Stop", hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)
