
import paho.mqtt.client as paho
import json


def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: "+str(mid)+" "+str(granted_qos))

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))    
    #mycol = mydb["movements"]
    json_object = json.loads(msg.payload)
    print(json_object)
    #mycol.insert_one(json_object)

client = paho.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect('broker.mqttdashboard.com', 1883)
client.subscribe('projectecho/engine/1', qos=1)

client.loop_forever()