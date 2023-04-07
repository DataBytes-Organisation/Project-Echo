import os
from fastapi import FastAPI, Body, HTTPException, status, APIRouter
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import motor.motor_asyncio
import datetime
from serializers import eventEntity, eventListEntity, eventSpeciesListEntity, eventSpeciesEntity
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


class StudentModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str = Field(...)
    email: EmailStr = Field(...)
    course: str = Field(...)
    gpa: float = Field(..., le=4.0)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "name": "Jane Doe",
                "email": "jdoe@example.com",
                "course": "Experiments, Science, and Fashion in Nanophotonics",
                "gpa": "3.0",
            }
        }


@app.get("/event", response_description="List all events")
def list_events():
    
    events = eventListEntity(db["events"].find())
    print(events)
    return events


@app.get("/species", response_description="Get events with certain species")
def show_event(species: str):
    print(species)
    events = eventListEntity(db["events"].find({"species": species}))
    print(events)
    return events

@app.get("/time", response_description="Get events within certain duration")
def show_event_from_time(start: str, end: str):
    print(start, end)
    datetime_start = datetime.datetime.strptime(start, '%Y-%m-%d-%H-%M-%S')
    datetime_end = datetime.datetime.strptime(end, '%Y-%m-%d-%H-%M-%S')
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
        }

    ]
    events = eventSpeciesListEntity(db["events"].aggregate(aggregate))
    print(events)
    return events

'''start = datetime.datetime(2023, 3, 22, 13, 44)
end = datetime.datetime(2023, 3, 23, 13, 44)'''