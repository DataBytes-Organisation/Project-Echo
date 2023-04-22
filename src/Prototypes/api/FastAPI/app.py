import os
from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import datetime
import serializers
import models
import pymongo
import dotenv

app = FastAPI()

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Replace with your own allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

dotenv.load_dotenv()
id = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
connection_string="mongodb+srv://projectechodeakin:YjGHDjNsEijhytYG@cluster0.gu2idc8.mongodb.net/test"
client = pymongo.MongoClient(connection_string)
db = client.mydatabase


@app.get("/events_time", response_description="Get detection events within certain duration")
def show_event_from_time(start: str, end: str):
    datetime_start = datetime.datetime.fromtimestamp(int(start), datetime.timezone.utc)
    datetime_end = datetime.datetime.fromtimestamp(int(end), datetime.timezone.utc)
    print(datetime_start)
    print(datetime_end)
    aggregate = [
        {
            '$match':{'timestamp': {'$lt' : datetime_end, '$gte' : datetime_start }}
            
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
            '$project': { "audioClip": 0, "sampleRate": 0, "fileFormat": 0}
        },
        {
            "$addFields": {
            "timestamp": { "$toLong": "$timestamp" }}
        }       

    ]
    events = serializers.eventSpeciesListEntity(db["events"].aggregate(aggregate))
    return events

@app.get("/audio", response_description="returns audio given ID")
def show_audio(id: str):
    aggregate = [
        {
            '$match':{'_id': ObjectId(id)}
        },
        {
            '$project': { "audioClip": 1}
        }
    ]
    results = list(db["events"].aggregate(aggregate))
    audio = serializers.audioListEntity(results)[0]
    return audio

@app.get("/movement_time", response_description="Get true animal movement data within certain duration")
def show_event_from_time(start: str, end: str):
    datetime_start = datetime.datetime.fromtimestamp(int(start), datetime.timezone.utc)
    datetime_end = datetime.datetime.fromtimestamp(int(end), datetime.timezone.utc)
    aggregate = [
        {
            '$match':{'timestamp': {'$lt' : datetime_end, '$gte' : datetime_start }}
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
    events = serializers.movementListEntity(db["movements"].aggregate(aggregate))
    return events

@app.get("/microphones", response_description="returns location of all microphones")
def list_microphones():
    aggregate = [
        {
            "$group":
            {
            "_id": "$sensorId",
            "microphoneLLA": { "$first": "$microphoneLLA" }
            }
        }
    ]
    results = list(db["events"].aggregate(aggregate))
    microphones = serializers.microphoneListEntity(results)
    return microphones
