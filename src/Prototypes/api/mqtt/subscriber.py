import paho.mqtt.client as paho
import pymongo
import json
import datetime
import dotenv
import os



dotenv.load_dotenv()

id = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
connection_string=f"mongodb+srv://projectechodeakin:YjGHDjNsEijhytYG@cluster0.gu2idc8.mongodb.net/test"
myclient = pymongo.MongoClient(connection_string)
mydb = myclient["mydatabase"]
print(myclient.list_database_names())
mydb["events"].drop()
mycol = mydb["events"]

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: "+str(mid)+" "+str(granted_qos))

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))    
    
    json_object = json.loads(msg.payload)
    print(json_object)
    mycol.insert_one(json_object)

client = paho.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect('broker.mqttdashboard.com', 1883)
client.subscribe('projectecho/1', qos=1)

client.loop_forever()