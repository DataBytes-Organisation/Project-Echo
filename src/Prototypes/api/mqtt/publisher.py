import paho.mqtt.client as paho
import time
import json


f = open('events.json')
data = json.load(f)



def on_publish(client, userdata, mid):
    print("mid: "+str(mid))
 
client = paho.Client()
client.on_publish = on_publish
client.connect('broker.mqttdashboard.com', 1883)
client.loop_start()

i = 0
while True:
    MQTT_MSG = json.dumps(data[i])

    (rc, mid) = client.publish('projectecho/1', MQTT_MSG, qos=1)
    print(MQTT_MSG)
    time.sleep(5)
    i += 1
    
    
