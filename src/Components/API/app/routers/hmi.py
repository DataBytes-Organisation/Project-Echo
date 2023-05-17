from fastapi import status, APIRouter
from bson import ObjectId
import datetime
from app import serializers
from app import schemas
from app.database import Events, Movements, Microphones
import paho.mqtt.publish as publish


router = APIRouter()

MQTT_BROKER_URL = "ts-mqtt-server-cont"
MQTT_BROKER_PORT = 1883


@router.get("/events_time", response_description="Get detection events within certain duration")
def show_event_from_time(start: str, end: str):
    datetime_start = datetime.datetime.fromtimestamp(float(start))
    datetime_end = datetime.datetime.fromtimestamp(float(end))
    # print(f'we think query date is {datetime_start}', flush=True)
    # print(datetime_end)
    aggregate = [
        {
            '$match':{'timestamp': { '$gte' : datetime_start, '$lt' : datetime_end}}
            
        },
        {
            '$lookup': {
                'from': 'species',
                'localField': 'species',
                'foreignField': '_id',
                'as': 'info'
            }
        },
        {
            "$replaceRoot": { "newRoot": { "$mergeObjects": [ { "$arrayElemAt": [ "$info", 0 ] }, "$$ROOT" ] } }
        },
        {
            '$project': { "audioClip": 0, "sampleRate": 0}
        },
        {
            "$addFields": {
            "timestamp": { "$toLong": "$timestamp" }}
        }       

    ]
    events = serializers.eventSpeciesListEntity(Events.aggregate(aggregate))
    return events

@router.get("/audio", response_description="returns audio given ID")
def show_audio(id: str):
    aggregate = [
        {
            '$match':{'_id': ObjectId(id)}
        },
        {
            '$project': { "audioClip": 1, "sampleRate": 1}
        }
    ]
    results = list(Events.aggregate(aggregate))
    audio = serializers.audioListEntity(results)[0]
    return audio

@router.get("/movement_time", response_description="Get true animal movement data within certain duration")
def show_event_from_time(start: str, end: str):
    datetime_start = datetime.datetime.fromtimestamp(float(start))
    datetime_end = datetime.datetime.fromtimestamp(float(end))
    
    print(f'movement range: {datetime_start}  {datetime_end}')
    
    aggregate = [
        {
            '$match':{'timestamp': {'$gte' : datetime_start ,'$lt' : datetime_end}}
        },
        {
            '$lookup': {
                'from': 'species',
                'localField': 'species',
                'foreignField': '_id',
                'as': 'info'
            }
        },
        {
            "$replaceRoot": { "newRoot": { "$mergeObjects": [ { "$arrayElemAt": [ "$info", 0 ] }, "$$ROOT" ] } }
        },
        { "$project": { "info": 0 } },
        {
            "$addFields":
            {
            "timestamp": { "$toLong": "$timestamp" }
            }
        }       

    ]
    events = serializers.movementSpeciesListEntity(Movements.aggregate(aggregate))
    return events

@router.get("/microphones", response_description="returns location of all microphones")
def list_microphones():
    results = Microphones.find()
    microphones = serializers.microphoneListEntity(results)
    return microphones

@router.get("/latest_movement", response_description="returns the latest simluated movement message")
def latest_movememnt():

    aggregate = [
    { "$sort" : { "timestamp" : -1 } },
    { "$limit": 1 },
    { "$project": { "timestamp": 1 } }]

    result = list(Movements.aggregate(aggregate))
    timestamp = serializers.timestampListEntity(result)[0]
    print(timestamp)
    return timestamp



@router.post("/sim_control", status_code=status.HTTP_201_CREATED)
def post_control(control: str):

    publish.single("Simulator_Controls", control, hostname=MQTT_BROKER_URL, port=MQTT_BROKER_PORT)

    return control

