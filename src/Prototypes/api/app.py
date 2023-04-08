import os
from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import motor.motor_asyncio
import datetime
import serializers
import models
import pymongo

app = FastAPI()
router = APIRouter()
client = pymongo.MongoClient("mongodb://localhost:27017")
#client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = client.mydatabase


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


@app.get("/event", response_description="List all events")
def list_events():
    
    events = serializers.eventListEntity(db["events"].find())
    return events


@app.get("/species", response_description="Get events with certain species")
def show_event(species: str):
    events = serializers.eventListEntity(db["events"].find({"species": species}))
    return events

@app.get("/time", response_description="Get events within certain duration")
def show_event_from_time(start: str, end: str):
    datetime_start = datetime.datetime.fromtimestamp(int(start))
    datetime_end = datetime.datetime.fromtimestamp(int(end))
    aggregate = [
        {
            '$lookup': {
                'from': 'species',
                'localField': 'species',
                'foreignField': '_id',
                'as': 'info'
            }
        },
        {
            '$match':{'timestamp': {'$lt' : datetime_end, '$gte' : datetime_start }}
            
        },
        {
            '$project': { "audioClip": 0}
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