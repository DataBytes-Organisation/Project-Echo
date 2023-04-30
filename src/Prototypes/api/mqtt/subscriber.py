import paho.mqtt.client as paho
import pymongo
import json
import datetime
import os


def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: "+str(mid)+" "+str(granted_qos))

def on_message(client, userdata, msg):
    # print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))    
    
    json_object = json.loads(msg.payload)
    print(json_object['timestamp'])


client = paho.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect('localhost', 1883)
client.subscribe('projectecho/engine/2', qos=1)

client.loop_forever()