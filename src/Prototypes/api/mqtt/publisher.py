import paho.mqtt.client as paho
import time

def on_publish(client, userdata, mid):
    print("mid: "+str(mid))
 
client = paho.Client()
client.on_publish = on_publish
client.connect('broker.mqttdashboard.com', 1883)
client.loop_start()

while True:
    temperature = 15
    (rc, mid) = client.publish('projectecho/1', str(temperature), qos=1)
    print(temperature)
    time.sleep(5)